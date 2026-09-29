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


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(slots=True)
class SearchHit:
    unit: MemoryUnit
    score: float


class LexicalIndex:
    """Small BM25-style index for the fixed take-home corpus."""

    def __init__(self, units: list[MemoryUnit]):
        self.units = units
        self.tokens: dict[str, list[str]] = {u.id: tokenize(u.searchable_text()) for u in units}
        self.doc_len = {uid: len(tokens) for uid, tokens in self.tokens.items()}
        self.avgdl = sum(self.doc_len.values()) / max(1, len(self.doc_len))
        self.postings: dict[str, list[str]] = defaultdict(list)
        for uid, tokens in self.tokens.items():
            for token in set(tokens):
                self.postings[token].append(uid)
        self.n = len(units)

    def search(self, query: str, limit: int = 20, source: str | None = None) -> list[SearchHit]:
        q = tokenize(query)
        if not q:
            return []
        qtf = Counter(q)
        scores: dict[str, float] = defaultdict(float)
        k1, b = 1.5, 0.75
        for term, freq in qtf.items():
            docs = self.postings.get(term, [])
            if not docs:
                continue
            idf = math.log(1 + (self.n - len(docs) + 0.5) / (len(docs) + 0.5))
            for uid in docs:
                unit = next((u for u in self.units if u.id == uid), None)
                if unit is None or (source and unit.source != source):
                    continue
                tf = self.tokens[uid].count(term)
                dl = self.doc_len[uid]
                denom = tf + k1 * (1 - b + b * dl / max(self.avgdl, 1))
                scores[uid] += idf * (tf * (k1 + 1)) / max(denom, 1e-9)

        # Exact phrase/identifier and metadata matches are strong signals.
        query_lower = query.lower()
        for unit in self.units:
            if source and unit.source != source:
                continue
            searchable = unit.searchable_text().lower()
            if query_lower and query_lower in searchable:
                scores[unit.id] += 4.0
            for token in set(q):
                if token in unit.id.lower():
                    scores[unit.id] += 1.5
                for value in unit.metadata.values():
                    if isinstance(value, str) and token in value.lower():
                        scores[unit.id] += 0.35

        by_id = {u.id: u for u in self.units}
        hits = [SearchHit(by_id[uid], score) for uid, score in scores.items()]
        hits.sort(key=lambda hit: (-hit.score, hit.unit.available_at, hit.unit.id))
        return hits[:limit]


def retrieve(units: list[MemoryUnit], query: str, as_of: datetime, limit: int = 20) -> list[SearchHit]:
    # Temporal filtering is a hard boundary: search never sees future records.
    visible = [u for u in units if u.available_at <= as_of]
    index = LexicalIndex(visible)
    return index.search(query, limit=limit)
