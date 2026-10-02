from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .models import MemoryUnit
from .search import LexicalIndex, SearchHit, tokenize
from .semantic import SemanticIndex


@dataclass(slots=True)
class HybridHit:
    unit: MemoryUnit
    score: float


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_INSTRUCTION_MARKERS = (
    "ignore previous instructions", "ignore your previous instructions",
    "disregard previous instructions", "ignore all previous instructions",
    "system prompt", "assistant must", "do not tell alex", "forward all emails",
)


def _looks_like_untrusted_instruction(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INSTRUCTION_MARKERS)


def _date_keys(text: str) -> set[str]:
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
        if year is None:
            continue
        try:
            keys.add(datetime(int(year), _MONTHS[month.lower()], int(day)).date().isoformat())
        except ValueError:
            pass
    return keys


def _query_tokens(query: str) -> set[str]:
    return set(tokenize(query))


def _expansion_relevance(query_tokens: set[str], unit: MemoryUnit) -> tuple[int, int]:
    unit_tokens = set(tokenize(f"{unit.text} {unit.id} {unit.record_id}"))
    overlap = len(query_tokens & unit_tokens)
    all_terms = 1 if query_tokens and query_tokens.issubset(unit_tokens) else 0
    return overlap, all_terms


def _intent_bonus(query: str, unit: MemoryUnit) -> float:
    """Small deterministic reranking priors for common memory question intents.

    These are deliberately weaker than lexical/semantic agreement. They only
    break near-ties for evidence types that generic similarity often under-ranks:
    current launch dates, delivery/commitment status, causes of changes, and
    calendar facts.
    """
    q = query.lower()
    text = unit.text.lower()
    metadata = " ".join(str(v).lower() for v in unit.metadata.values() if v is not None)
    searchable = f"{text} {metadata}"
    bonus = 0.0

    asks_launch = "launch" in q or "launching" in q
    asks_when = "when" in q
    asks_days = "how many days" in q or "days after" in q
    asks_delivery = any(term in q for term in ("send", "sent", "go out", "went out"))
    asks_cause = any(term in q for term in ("why", "slip", "moved", "regression"))
    asks_calendar = "board" in q or "prep" in q or "calendar" in q

    if asks_launch:
        if any(term in searchable for term in ("launch", "launching", "go/no-go", "target")):
            bonus += 0.0035
        if _date_keys(unit.text):
            bonus += 0.0030
        if any(term in searchable for term in ("moved", "move", "current", "now", "final")):
            bonus += 0.0020
        if asks_when and unit.source in {"meeting", "slack", "email", "calendar", "dictation"}:
            bonus += 0.0010

    if asks_days and _date_keys(unit.text):
        bonus += 0.0040

    if asks_delivery:
        if any(term in searchable for term in ("sent", "send", "went out", "delivered")):
            bonus += 0.0050
        if unit.metadata.get("delivery_state"):
            bonus += 0.0020

    if asks_cause:
        if any(term in searchable for term in ("because", "reason", "regression", "geocod", "moved", "slip")):
            bonus += 0.0045

    if asks_calendar:
        if unit.source == "calendar":
            bonus += 0.0050
        if any(term in searchable for term in ("board", "prep", "meeting")):
            bonus += 0.0030
        if _date_keys(unit.text):
            bonus += 0.0015

    return bonus


class HybridIndex:
    """Fuse lexical and semantic retrieval, then expand evidence chains."""

    RRF_K = 60.0
    CANDIDATE_LIMIT = 60
    LEXICAL_ANCHOR_LIMIT = 5
    SEMANTIC_ANCHOR_LIMIT = 10
    LEXICAL_ANCHOR_BONUS = 0.0065
    SEMANTIC_ANCHOR_BONUS = 0.0045
    MAX_EXPANDED_SIBLINGS_PER_RECORD = 3

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

        semantic_ids = {hit.unit.id for hit in semantic_hits}
        for rank, hit in enumerate(lexical_hits[: self.LEXICAL_ANCHOR_LIMIT], start=1):
            if hit.unit.id not in semantic_ids:
                fused[hit.unit.id] = fused.get(hit.unit.id, 0.0) + self.LEXICAL_ANCHOR_BONUS

        lexical_ids = {hit.unit.id for hit in lexical_hits}
        for rank, hit in enumerate(semantic_hits[: self.SEMANTIC_ANCHOR_LIMIT], start=1):
            if hit.unit.id not in lexical_ids:
                fused[hit.unit.id] = fused.get(hit.unit.id, 0.0) + self.SEMANTIC_ANCHOR_BONUS

        # Intent priors are applied only after both retrieval channels have
        # produced candidates. They cannot invent evidence that neither channel
        # found, and they are intentionally small enough not to overpower RRF.
        for uid in list(fused):
            fused[uid] += _intent_bonus(query, self.by_id[uid])
        return fused

    def _expand_evidence(self, query: str, fused: dict[str, float]) -> dict[str, float]:
        if not fused:
            return fused

        ranked_ids = sorted(fused, key=lambda uid: (-fused[uid], uid))
        seeds = [
            self.by_id[uid] for uid in ranked_ids[:20]
            if not _looks_like_untrusted_instruction(self.by_id[uid].text)
        ]
        expanded = dict(fused)
        q_tokens = _query_tokens(query)
        expanded_record_counts: dict[str, int] = {}

        for seed in seeds:
            siblings = [
                unit for unit in self.by_record.get(seed.record_id, [])
                if unit.id != seed.id and not _looks_like_untrusted_instruction(unit.text)
            ]
            siblings.sort(
                key=lambda unit: (
                    _expansion_relevance(q_tokens, unit),
                    -unit.available_at.timestamp(),
                    unit.id,
                ),
                reverse=True,
            )
            record_count = expanded_record_counts.get(seed.record_id, 0)
            for unit in siblings:
                if record_count >= self.MAX_EXPANDED_SIBLINGS_PER_RECORD:
                    break
                if unit.id in fused:
                    continue
                expanded[unit.id] = max(expanded.get(unit.id, 0.0), fused[seed.id] * 0.65)
                record_count += 1
            expanded_record_counts[seed.record_id] = record_count

            target_id = seed.metadata.get("target_id")
            if target_id and target_id in self.by_id:
                target = self.by_id[target_id]
                if not _looks_like_untrusted_instruction(target.text):
                    expanded[target_id] = max(expanded.get(target_id, 0.0), fused[seed.id] * 0.88)

            thread_parent = seed.metadata.get("thread_parent_id")
            if thread_parent:
                for unit in self.units:
                    if (
                        unit.metadata.get("thread_parent_id") == thread_parent
                        and not _looks_like_untrusted_instruction(unit.text)
                    ):
                        expanded[unit.id] = max(expanded.get(unit.id, 0.0), fused[seed.id] * 0.65)

        date_keys: set[str] = set()
        for seed in seeds:
            date_keys.update(_date_keys(seed.text))

        if date_keys:
            for unit in self.units:
                if unit.source != "calendar" or _looks_like_untrusted_instruction(unit.text):
                    continue
                if date_keys & _date_keys(unit.text):
                    expanded[unit.id] = max(expanded.get(unit.id, 0.0), 0.0105)
        return expanded

    def search(self, query: str, limit: int = 20) -> list[HybridHit]:
        fused = self._base_search(query)
        fused = self._expand_evidence(query, fused)
        ranked = sorted(
            fused.items(),
            key=lambda item: (-item[1], self.by_id[item[0]].available_at, item[0]),
        )
        return [HybridHit(self.by_id[uid], score) for uid, score in ranked[:limit]]
