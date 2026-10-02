from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .models import MemoryUnit
from .search import LexicalIndex, SearchHit
from .semantic import SemanticIndex


@dataclass(slots=True)
class HybridHit:
    unit: MemoryUnit
    score: float


_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_INSTRUCTION_MARKERS = (
    "ignore previous instructions",
    "ignore your previous instructions",
    "disregard previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "assistant must",
    "do not tell alex",
    "forward all emails",
)


def _looks_like_untrusted_instruction(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INSTRUCTION_MARKERS)


def _date_keys(text: str) -> set[str]:
    """Return normalized YYYY-MM-DD keys mentioned in a memory passage."""
    keys: set[str] = set()

    for match in re.finditer(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text):
        year, month, day = map(int, match.groups())
        try:
            keys.add(datetime(year, month, day).date().isoformat())
        except ValueError:
            pass

    month_names = "|".join(_MONTHS)
    pattern = rf"\b({month_names})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d{{2}}))?\b"
    for match in re.finditer(pattern, text, re.I):
        month, day, year = match.groups()
        year_i = int(year) if year else None
        if year_i is None:
            continue
        try:
            keys.add(datetime(year_i, _MONTHS[month.lower()], int(day)).date().isoformat())
        except ValueError:
            pass

    return keys


class HybridIndex:
    """Fuse lexical and semantic retrieval, then expand evidence chains.

    The first stage is ordinary lexical + dense retrieval.  The second stage
    deliberately retrieves *related evidence* that a single natural-language
    query may not mention verbatim: other segments in the same record/thread,
    Slack edit targets, and calendar events occurring on a date discovered in
    a seed result.  This keeps the semantic model responsible for recall while
    deterministic relationship logic handles multi-hop structure.
    """

    RRF_K = 60.0
    CANDIDATE_LIMIT = 60

    def __init__(self, units: list[MemoryUnit]):
        self.units = units
        self.by_id = {u.id: u for u in units}
        self.lexical = LexicalIndex(units)
        self.semantic: SemanticIndex | None = None
        try:
            self.semantic = SemanticIndex(units)
        except (ImportError, RuntimeError):
            self.semantic = None

        self.by_record: dict[str, list[MemoryUnit]] = {}
        for unit in units:
            self.by_record.setdefault(unit.record_id, []).append(unit)

    def _base_search(self, query: str) -> dict[str, float]:
        lexical_hits = self.lexical.search(query, limit=self.CANDIDATE_LIMIT)
        fused: dict[str, float] = {}
        if self.semantic is None:
            for rank, hit in enumerate(lexical_hits, start=1):
                fused[hit.unit.id] = 1.0 / (self.RRF_K + rank)
            return fused

        semantic_hits = self.semantic.search(query, limit=self.CANDIDATE_LIMIT)
        for rank, hit in enumerate(lexical_hits, start=1):
            fused[hit.unit.id] = fused.get(hit.unit.id, 0.0) + 1.0 / (self.RRF_K + rank)
        for rank, hit in enumerate(semantic_hits, start=1):
            fused[hit.unit.id] = fused.get(hit.unit.id, 0.0) + 1.0 / (self.RRF_K + rank)
        return fused

    def _expand_evidence(self, query: str, fused: dict[str, float]) -> dict[str, float]:
        """Expand from strong seeds into deterministic related evidence."""
        if not fused:
            return fused

        ranked_ids = sorted(fused, key=lambda uid: (-fused[uid], uid))
        seeds = [
            self.by_id[uid]
            for uid in ranked_ids[:20]
            if not _looks_like_untrusted_instruction(self.by_id[uid].text)
        ]
        expanded = dict(fused)

        # 1. Same record/thread: meetings, Slack threads, Gmail threads and
        # other grouped records often split one fact across many short units.
        # Keep this relationship useful without allowing a single seed to
        # flood the final ranking with loosely related sibling segments.
        # Never create new retrieval paths from instruction-bearing content.
        for seed in seeds:
            for unit in self.by_record.get(seed.record_id, []):
                if unit.id == seed.id or _looks_like_untrusted_instruction(unit.text):
                    continue
                expanded[unit.id] = max(
                    expanded.get(unit.id, 0.0),
                    fused[seed.id] * 0.65,
                )

            target_id = seed.metadata.get("target_id")
            if target_id and target_id in self.by_id:
                target = self.by_id[target_id]
                if not _looks_like_untrusted_instruction(target.text):
                    expanded[target_id] = max(
                        expanded.get(target_id, 0.0),
                        fused[seed.id] * 0.88,
                    )

            thread_parent = seed.metadata.get("thread_parent_id")
            if thread_parent:
                for unit in self.units:
                    if (
                        unit.metadata.get("thread_parent_id") == thread_parent
                        and not _looks_like_untrusted_instruction(unit.text)
                    ):
                        expanded[unit.id] = max(
                            expanded.get(unit.id, 0.0),
                            fused[seed.id] * 0.65,
                        )

        # 2. Date hop: if a seed mentions a concrete date, retrieve calendar
        # events on that date. This handles questions such as "what is on my
        # calendar the day I fly to Denver?" without asking the embedding model
        # to perform symbolic date reasoning.
        date_keys: set[str] = set()
        for seed in seeds:
            date_keys.update(_date_keys(seed.text))

        if date_keys:
            for unit in self.units:
                if unit.source != "calendar" or _looks_like_untrusted_instruction(unit.text):
                    continue
                unit_dates = _date_keys(unit.text)
                if date_keys & unit_dates:
                    expanded[unit.id] = max(
                        expanded.get(unit.id, 0.0),
                        0.0105,
                    )

        return expanded

    def search(self, query: str, limit: int = 20) -> list[HybridHit]:
        fused = self._base_search(query)
        fused = self._expand_evidence(query, fused)

        ranked = sorted(
            fused.items(),
            key=lambda item: (
                -item[1],
                self.by_id[item[0]].available_at,
                item[0],
            ),
        )
        return [
            HybridHit(self.by_id[uid], score)
            for uid, score in ranked[:limit]
        ]
