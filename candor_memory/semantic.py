from __future__ import annotations

from dataclasses import dataclass

from .models import MemoryUnit


@dataclass(slots=True)
class SemanticHit:
    unit: MemoryUnit
    score: float


class SemanticIndex:
    """Optional dense semantic retrieval using a pretrained Sentence Transformer.

    This module is deliberately optional. The deterministic lexical retriever remains
    the fallback when sentence-transformers is not installed.
    """

    MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

    def __init__(self, units: list[MemoryUnit], model_name: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Semantic retrieval requires sentence-transformers. "
                "Install requirements-semantic.txt to enable it."
            ) from exc

        self.units = units
        self.model = SentenceTransformer(model_name or self.MODEL_NAME)
        self.embeddings = self.model.encode(
            [self._document_text(unit) for unit in units],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    @staticmethod
    def _document_text(unit: MemoryUnit) -> str:
        """Give the encoder the evidence plus useful identity metadata."""
        metadata = []
        for key in ("title", "subject", "author", "speaker", "channel", "repo", "target_context"):
            value = unit.metadata.get(key)
            if value is not None:
                metadata.append(f"{key}: {value}")
        prefix = " | ".join(metadata)
        return f"{prefix}\n{unit.text}" if prefix else unit.text

    def search(self, query: str, limit: int = 50) -> list[SemanticHit]:
        if not self.units:
            return []
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Embeddings are normalized, so the dot product is cosine similarity.
        scores = self.embeddings @ query_embedding
        ranked = sorted(
            range(len(self.units)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )
        return [
            SemanticHit(self.units[idx], float(scores[idx]))
            for idx in ranked[:limit]
        ]
