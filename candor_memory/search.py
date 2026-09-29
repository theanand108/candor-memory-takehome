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
}

_MONTH_ALIASES = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr",
    "june": "jun", "july": "jul", "august": "aug", "september": "sep",
    "october": "oct", "november": "nov", "december": "dec",
}


def _stem(token: str) -> str:
    """Small domain-safe stemmer; avoids external NLP dependencies."""
    if len(token) > 6 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ingly", "edly", "ing", "ed"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    if len(token) > 5 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    result: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in _STOP:
            continue
        result.append(raw)
        if raw in _MONTH_ALIASES:
            result.append(_MONTH_ALIASES[raw])
        result.append(_stem(raw))
    return result


@dataclass(slots=True)
class SearchHit:
    unit: MemoryUnit
    score: float


class LexicalIndex:
    """BM25-style + field/phrase retrieval for the fixed take-home corpus."""

    def __init__(self, units: list[MemoryUnit]):
        self.units = units
        self.by_id = {u.id: u for u in units}
        self.tokens: dict[str, list[str]] = {
            u.id: tokenize(self._document_text(u)) for u in units
        }
        self.doc_len = {uid: len(tokens) for uid, tokens in self.tokens.items()}
        self.avgdl = sum(self.doc_len.values()) / max(1, len(self.doc_len))
        self.postings: dict[str, list[str]] = defaultdict(list)
        for uid, tokens in self.tokens.items():
            for token in set(tokens):
                self.postings[token].append(uid)
        self.n = len(units)

    @staticmethod
    def _document_text(unit: MemoryUnit) -> str:
        fields = [unit.text, unit.id, unit.record_id]
        for key, value in unit.metadata.items():
            if value is not None:
                if isinstance(value, (str, int, float)):
                    fields.append(str(value))
        return " ".join(fields)

    def search(self, query: str, limit: int = 20, source: str | None = None) -> list[SearchHit]:
        q = tokenize(query)
        if not q:
            return []

        q_unique = set(q)
        qtf = Counter(q)
        scores: dict[str, float] = defaultdict(float)
        k1, b = 1.5, 0.72

        for term, q_freq in qtf.items():
            docs = self.postings.get(term, [])
            if not docs:
                continue
            idf = math.log(1 + (self.n - len(docs) + 0.5) / (len(docs) + 0.5))
            for uid in docs:
                unit = self.by_id[uid]
                if source and unit.source != source:
                    continue
                tf = self.tokens[uid].count(term)
                dl = self.doc_len[uid]
                denom = tf + k1 * (1 - b + b * dl / max(self.avgdl, 1))
                scores[uid] += idf * (tf * (k1 + 1)) / max(denom, 1e-9)

        query_lower = query.lower().strip()
        for unit in self.units:
            if source and unit.source != source:
                continue
            uid = unit.id
            text_lower = unit.text.lower()
            full_text = self._document_text(unit).lower()

            # Exact phrase is a strong signal, especially for named projects,
            # people, and multi-word questions.
            if query_lower and query_lower in full_text:
                scores[uid] += 8.0

            # Field-aware boosts keep exact entity/title/subject matches near
            # the top instead of allowing long documents to dominate BM25.
            field_weights = {
                "title": 3.5,
                "subject": 3.5,
                "author": 2.5,
                "speaker": 2.5,
                "channel": 2.0,
                "repo": 2.0,
                "target_context": 1.5,
            }
            for field, weight in field_weights.items():
                value = unit.metadata.get(field)
                if not isinstance(value, str):
                    continue
                field_tokens = set(tokenize(value))
                scores[uid] += weight * len(q_unique & field_tokens)

            # Exact identifier/entity matches are useful for cross-source and
            # multi-hop questions.
            identifier_tokens = set(tokenize(f"{unit.id} {unit.record_id}"))
            scores[uid] += 1.2 * len(q_unique & identifier_tokens)

            # Short records with high query overlap are often the exact evidence
            # passage; give them a modest precision bonus.
            content_tokens = set(tokenize(text_lower))
            overlap = len(q_unique & content_tokens)
            if overlap:
                scores[uid] += min(2.5, 0.35 * overlap)

        hits = [SearchHit(self.by_id[uid], score) for uid, score in scores.items()]
        hits.sort(key=lambda hit: (-hit.score, hit.unit.available_at, hit.unit.id))
        return hits[:limit]


def retrieve(units: list[MemoryUnit], query: str, as_of: datetime, limit: int = 20) -> list[SearchHit]:
    # Temporal filtering is a hard boundary: search never sees future records.
    visible = [u for u in units if u.available_at <= as_of]
    index = LexicalIndex(visible)
    return index.search(query, limit=limit)
