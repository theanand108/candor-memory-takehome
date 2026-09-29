from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

from .models import MemoryUnit

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_@.\-']*")
_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "at",
    "is", "are", "was", "were", "what", "when", "who", "did", "do", "does",
    "how", "why", "about", "with", "our", "their", "this", "that", "it", "i",
    "we", "they", "he", "she", "from", "by", "be", "as", "has", "have", "had",
    "my", "me", "you", "your", "going", "can", "could", "would", "will",
}
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def tokenize(text: str) -> list[str]:
    tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
    expanded: list[str] = []
    for token in tokens:
        expanded.append(_NUMBER_WORDS.get(token, token))
    return expanded


def _stem(token: str) -> str:
    """Tiny morphology normalizer; conservative for names/codes."""
    if len(token) <= 4 or any(ch.isdigit() for ch in token) or "@" in token:
        return token
    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _norm_tokens(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text)]


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    if not text:
        return set()
    padded = f"  {text}  "
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


def _phrases(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


@dataclass(slots=True)
class SearchHit:
    unit: MemoryUnit
    score: float


class LexicalIndex:
    """Deterministic multi-stage lexical retriever.

    Stage 1 scores individual units with BM25-like lexical evidence and rich
    source metadata. Stage 2 adds record-level evidence. Stage 3 performs a
    narrow contextual expansion around strong transcript/thread hits instead
    of spreading one good match across every neighbouring filler segment.

    This mirrors a common RAG pattern: retrieve broadly, then use document
    context to improve the final ranking. citeturn1search0turn1search5
    """

    def __init__(self, units: list[MemoryUnit]):
        self.units = units
        self.by_id = {u.id: u for u in units}
        self.tokens = {u.id: _norm_tokens(u.searchable_text()) for u in units}
        self.raw_tokens = {u.id: set(tokenize(u.searchable_text())) for u in units}
        self.doc_len = {u.id: len(tokens) for u, tokens in ((u, self.tokens[u.id]) for u in units)}
        self.avgdl = sum(self.doc_len.values()) / max(1, len(self.doc_len))
        self.df: Counter[str] = Counter()
        for tokens in self.tokens.values():
            self.df.update(set(tokens))
        self.n = len(units)

        self.record_units: dict[str, list[MemoryUnit]] = defaultdict(list)
        for unit in units:
            self.record_units[unit.record_id].append(unit)
        for members in self.record_units.values():
            members.sort(key=lambda u: (u.available_at, u.id))

        self.record_tokens = {
            rid: _norm_tokens(" ".join(m.searchable_text() for m in members))
            for rid, members in self.record_units.items()
        }

    def _idf(self, term: str) -> float:
        docs = self.df.get(term, 0)
        if not docs:
            return 0.0
        return math.log(1 + (self.n - docs + 0.5) / (docs + 0.5))

    def _bm25(self, uid: str, q: list[str]) -> float:
        tokens = self.tokens[uid]
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        dl = self.doc_len[uid]
        k1, b = 1.5, 0.75
        score = 0.0
        for term in set(q):
            tf = counts.get(term, 0)
            if not tf:
                continue
            denom = tf + k1 * (1 - b + b * dl / max(self.avgdl, 1))
            score += self._idf(term) * (tf * (k1 + 1)) / max(denom, 1e-9)
        return score

    def _record_bm25(self, record_id: str, q: list[str]) -> float:
        tokens = self.record_tokens.get(record_id, [])
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        dl = len(tokens)
        avg = max(1.0, sum(map(len, self.record_tokens.values())) / max(1, len(self.record_tokens)))
        score = 0.0
        record_sets = {rid: set(values) for rid, values in self.record_tokens.items()}
        for term in set(q):
            tf = counts.get(term, 0)
            if not tf:
                continue
            rdocs = sum(term in values for values in record_sets.values())
            idf = math.log(1 + (len(record_sets) - rdocs + 0.5) / (rdocs + 0.5))
            denom = tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg)
            score += idf * (tf * 2.5) / max(denom, 1e-9)
        return score

    def search(self, query: str, limit: int = 20, source: str | None = None) -> list[SearchHit]:
        q_raw = tokenize(query)
        q = [_stem(t) for t in q_raw]
        if not q:
            return []

        query_lower = query.lower().strip()
        query_chars = _char_ngrams(query_lower)
        query_raw_set = set(q_raw)
        query_stem_set = set(q)
        query_phrases = _phrases(q)
        scores: dict[str, float] = defaultdict(float)
        direct: dict[str, float] = defaultdict(float)
        candidates = self.units if source is None else [u for u in self.units if u.source == source]

        # ---------- Stage 1: direct unit retrieval ----------
        for unit in candidates:
            uid = unit.id
            searchable = unit.searchable_text().lower()
            tokens = self.tokens[uid]
            scores[uid] += self._bm25(uid, q)

            if len(query_lower) >= 5 and query_lower in searchable:
                scores[uid] += 5.0

            # Exact multi-word phrase matches are much stronger than isolated
            # words. This helps questions containing product names and dates.
            unit_phrases = _phrases(tokens)
            phrase_hits = len(query_phrases & unit_phrases)
            if phrase_hits:
                scores[uid] += min(3.0, 1.5 * phrase_hits)

            exact_terms = len(query_raw_set & self.raw_tokens[uid])
            scores[uid] += 0.55 * exact_terms

            if query_chars:
                doc_chars = _char_ngrams(searchable)
                if doc_chars:
                    overlap = len(query_chars & doc_chars) / len(query_chars)
                    scores[uid] += min(1.2, overlap * 1.6)

            for key, weight in (
                ("title", 2.25), ("subject", 2.25), ("speaker", 1.15),
                ("author", 1.15), ("channel", 0.8), ("repo", 1.2),
                ("target_context", 1.5), ("mode", 0.35), ("delivery_state", 0.45),
            ):
                value = unit.metadata.get(key)
                if not value:
                    continue
                value_tokens = set(_norm_tokens(str(value)))
                matched = len(query_stem_set & value_tokens)
                if matched:
                    scores[uid] += weight * matched

            direct[uid] = scores[uid]

        # ---------- Stage 2: record-level context ----------
        # A question may mention a topic once while the answer is another
        # segment in the same meeting/thread. Give the record a signal, but do
        # not blindly boost every member.
        record_scores = {rid: self._record_bm25(rid, q) for rid in self.record_units}
        for unit in candidates:
            rs = record_scores.get(unit.record_id, 0.0)
            if rs <= 0:
                continue
            own = direct.get(unit.id, 0.0)
            # Strong direct hits get a little record context. Weak/filler units
            # only get a contextual path in Stage 3.
            if own > 1.0:
                scores[unit.id] += min(2.0, rs * 0.45)

        # ---------- Stage 3: narrow local context expansion ----------
        # Meetings are diarized into tiny segments. The question may retrieve
        # the setup line while the answer is 1-6 segments away. Expand around
        # strong seeds, but require the neighbour to contain either query terms
        # or meaningful overlap with the seed; this avoids flooding top-10 with
        # "Yeah", "Okay", "And?" filler.
        for rid, members in self.record_units.items():
            eligible = [m for m in members if m.id in direct]
            seeds = sorted(eligible, key=lambda m: direct[m.id], reverse=True)[:3]
            for seed in seeds:
                seed_score = direct[seed.id]
                if seed_score < 1.8:
                    continue
                seed_tokens = set(self.tokens[seed.id])
                if not seed_tokens:
                    continue
                seed_index = members.index(seed)
                for distance in range(1, 7):
                    for idx in (seed_index - distance, seed_index + distance):
                        if idx < 0 or idx >= len(members):
                            continue
                        neighbour = members[idx]
                        if neighbour.id not in direct:
                            continue
                        if neighbour.id == seed.id:
                            continue
                        neighbour_tokens = set(self.tokens[neighbour.id])
                        query_overlap = len(query_stem_set & neighbour_tokens)
                        seed_overlap = len(seed_tokens & neighbour_tokens)
                        if query_overlap == 0 and seed_overlap == 0:
                            continue
                        distance_factor = 1.0 / (1.0 + 0.35 * distance)
                        context_signal = min(seed_score, 5.0) * distance_factor
                        if query_overlap:
                            scores[neighbour.id] += min(1.6, context_signal * 0.42 + 0.25 * query_overlap)
                        else:
                            scores[neighbour.id] += min(0.9, context_signal * 0.22)

        # Short filler segments should not win merely from generic overlap.
        for unit in candidates:
            if len(unit.text.strip()) < 24 and direct.get(unit.id, 0.0) < 1.5:
                scores[unit.id] *= 0.58

        hits = [SearchHit(self.by_id[uid], score) for uid, score in scores.items() if score > 0]
        hits.sort(key=lambda hit: (-hit.score, hit.unit.available_at, hit.unit.id))
        return hits[:limit]


def retrieve(units: list[MemoryUnit], query: str, as_of: datetime, limit: int = 20) -> list[SearchHit]:
    visible = [u for u in units if u.available_at <= as_of]
    return LexicalIndex(visible).search(query, limit=limit)
