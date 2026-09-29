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


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


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


@dataclass(slots=True)
class SearchHit:
    unit: MemoryUnit
    score: float


class LexicalIndex:
    """Deterministic hybrid lexical retriever for the take-home corpus.

    Besides unit-level BM25-style matching, this index uses weighted metadata,
    conservative morphology, character overlap, record-level topic evidence,
    and local transcript context. The latter matters because a meeting answer
    is often split across several short diarized segments.
    """

    def __init__(self, units: list[MemoryUnit]):
        self.units = units
        self.by_id = {u.id: u for u in units}
        self.tokens = {u.id: _norm_tokens(u.searchable_text()) for u in units}
        self.raw_tokens = {u.id: set(tokenize(u.searchable_text())) for u in units}
        self.doc_len = {uid: len(tokens) for uid, tokens in self.tokens.items()}
        self.avgdl = sum(self.doc_len.values()) / max(1, len(self.doc_len))
        self.df: Counter[str] = Counter()
        for tokens in self.tokens.values():
            self.df.update(set(tokens))
        self.n = len(units)

        self.record_units: dict[str, list[MemoryUnit]] = defaultdict(list)
        for unit in units:
            self.record_units[unit.record_id].append(unit)
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
        for term in set(q):
            tf = counts.get(term, 0)
            if not tf:
                continue
            rdocs = sum(term in set(values) for values in self.record_tokens.values())
            idf = math.log(1 + (len(self.record_tokens) - rdocs + 0.5) / (rdocs + 0.5))
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
        scores: dict[str, float] = defaultdict(float)
        strong_local: dict[str, float] = defaultdict(float)
        candidates = self.units if source is None else [u for u in self.units if u.source == source]

        for unit in candidates:
            uid = unit.id
            searchable = unit.searchable_text().lower()
            scores[uid] += self._bm25(uid, q)

            if len(query_lower) >= 5 and query_lower in searchable:
                scores[uid] += 5.0

            exact_terms = len(query_raw_set & self.raw_tokens[uid])
            scores[uid] += 0.55 * exact_terms

            if query_chars:
                doc_chars = _char_ngrams(searchable)
                if doc_chars:
                    overlap = len(query_chars & doc_chars) / len(query_chars)
                    scores[uid] += min(1.5, overlap * 2.0)

            # Titles/subjects are strong topic anchors; speaker/author/channel
            # help attribution and source-specific questions.
            for key, weight in (
                ("title", 2.25), ("subject", 2.25), ("speaker", 1.15),
                ("author", 1.15), ("channel", 0.8), ("repo", 0.8),
                ("target_context", 0.9),
            ):
                value = unit.metadata.get(key)
                if not value:
                    continue
                matched = len(set(q) & set(_norm_tokens(str(value))))
                if matched:
                    scores[uid] += weight * matched
                    if key in {"title", "subject", "target_context"}:
                        strong_local[uid] += weight * matched

        # A query can match the topic of a whole meeting while the exact answer
        # lives in a different segment. Spread a modest record-level signal.
        record_scores = {rid: self._record_bm25(rid, q) for rid in self.record_units}
        for unit in candidates:
            rs = record_scores.get(unit.record_id, 0.0)
            if rs > 0:
                scores[unit.id] += min(2.5, rs * 0.65)

        # Nearby transcript segments frequently contain the setup/answer pair.
        for members in self.record_units.values():
            members = sorted(members, key=lambda u: (u.available_at, u.id))
            for i, unit in enumerate(members):
                sibling_best = 0.0
                for j in range(max(0, i - 3), min(len(members), i + 4)):
                    if i != j:
                        sibling_best = max(sibling_best, scores.get(members[j].id, 0.0))
                if sibling_best > 2.0:
                    scores[unit.id] += min(1.25, sibling_best * 0.10)

        # Short filler segments ("Yeah", "And?", "Perfect") should not win
        # merely because they share a common word with the question.
        for unit in candidates:
            if len(unit.text.strip()) < 24 and strong_local.get(unit.id, 0.0) == 0:
                scores[unit.id] *= 0.72

        hits = [SearchHit(self.by_id[uid], score) for uid, score in scores.items() if score > 0]
        hits.sort(key=lambda hit: (-hit.score, hit.unit.available_at, hit.unit.id))
        return hits[:limit]


def retrieve(units: list[MemoryUnit], query: str, as_of: datetime, limit: int = 20) -> list[SearchHit]:
    visible = [u for u in units if u.available_at <= as_of]
    return LexicalIndex(visible).search(query, limit=limit)
