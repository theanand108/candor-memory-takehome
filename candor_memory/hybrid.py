from __future__ import annotations

from dataclasses import dataclass

from .models import MemoryUnit
from .search import LexicalIndex, SearchHit
from .semantic import SemanticIndex, SemanticHit


@dataclass(slots=True)
class HybridHit:
    unit: MemoryUnit
    score: float


class HybridIndex:
    """Fuse lexical and semantic retrieval without weakening temporal filtering.

    BM25/lexical retrieval is retained because exact IDs, names, dates and jargon
    are important. Dense retrieval adds semantic recall for paraphrases and
    natural-language questions. Reciprocal Rank Fusion avoids putting the two
    raw score scales on the same numeric axis.
    """

    RRF_K = 60.0

    def __init__(self, units: list[MemoryUnit]):
        self.units = units
        self.lexical = LexicalIndex(units)
        self.semantic: SemanticIndex | None = None
        try:
            self.semantic = SemanticIndex(units)
        except (ImportError, RuntimeError):
            # The take-home must remain runnable with the standard-library
            # baseline. If the optional model is unavailable, lexical retrieval
            # is the deterministic fallback.
            self.semantic = None

    def search(self, query: str, limit: int = 20) -> list[HybridHit]:
        lexical_hits = self.lexical.search(query, limit=max(50, limit * 3))
        if self.semantic is None:
            return [HybridHit(hit.unit, hit.score) for hit in lexical_hits[:limit]]

        semantic_hits = self.semantic.search(query, limit=max(50, limit * 3))
        fused: dict[str, float] = {}
        units: dict[str, MemoryUnit] = {}

        for rank, hit in enumerate(lexical_hits, start=1):
            fused[hit.unit.id] = fused.get(hit.unit.id, 0.0) + 1.0 / (self.RRF_K + rank)
            units[hit.unit.id] = hit.unit

        for rank, hit in enumerate(semantic_hits, start=1):
            fused[hit.unit.id] = fused.get(hit.unit.id, 0.0) + 1.0 / (self.RRF_K + rank)
            units[hit.unit.id] = hit.unit

        ranked = sorted(
            fused.items(),
            key=lambda item: (-item[1], units[item[0]].available_at, item[0]),
        )
        return [HybridHit(units[uid], score) for uid, score in ranked[:limit]]
