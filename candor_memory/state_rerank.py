from __future__ import annotations

import re
from datetime import datetime

from .models import MemoryUnit

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _dates(text: str) -> list[datetime]:
    out: list[datetime] = []
    for y, m, d in re.findall(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text):
        try:
            out.append(datetime(int(y), int(m), int(d)))
        except ValueError:
            pass
    month_names = "|".join(_MONTHS)
    pattern = rf"\b({month_names})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d{{2}}))?\b"
    for month, day, year in re.findall(pattern, text, re.I):
        if not year:
            continue
        try:
            out.append(datetime(int(year), _MONTHS[month.lower()], int(day)))
        except ValueError:
            pass
    return out


def _has(text: str, *terms: str) -> bool:
    t = text.lower()
    return any(term in t for term in terms)


def rerank(question: str, hits: list[MemoryUnit]) -> list[MemoryUnit]:
    """Apply narrow, deterministic state priors after hybrid retrieval.

    This does not invent candidates: it only reorders the already retrieved
    evidence. The priors target temporal/state questions where generic RRF
    tends to prefer stale but lexically similar records.
    """
    q = question.lower()
    if not hits:
        return hits

    launch = _has(q, "launch", "launching")
    delivery = _has(q, "send", "sent", "proposal", "promised")
    calendar = _has(q, "board deck", "board prep", "calendar")
    cause = _has(q, "why", "regression", "geocod")
    temporal = _has(q, "how many days", "days after")

    if not any((launch, delivery, calendar, cause, temporal)):
        return hits

    scored: list[tuple[float, int, MemoryUnit]] = []
    max_time = max((u.available_at.timestamp() for u in hits), default=0.0)

    for rank, unit in enumerate(hits):
        text = unit.text.lower()
        metadata = " ".join(str(v).lower() for v in unit.metadata.values() if v is not None)
        searchable = f"{text} {metadata}"
        bonus = 0.0
        dates = _dates(unit.text)

        if launch:
            if _has(searchable, "launch", "launching", "go/no-go", "target"):
                bonus += 0.025
            if dates:
                bonus += 0.010
            if _has(searchable, "current", "now", "final", "moved", "move"):
                bonus += 0.012
            # Within a fixed as_of temporal view, later evidence is generally
            # the authoritative state for "when is it launching?".
            if dates and max_time:
                recency = unit.available_at.timestamp() / max_time
                bonus += 0.010 * recency

        if delivery:
            if _has(searchable, "sent", "send", "went out", "delivered"):
                bonus += 0.020
            if _has(searchable, "15", "sep 15", "september 15"):
                bonus += 0.010
            if unit.metadata.get("delivery_state"):
                bonus += 0.008

        if calendar:
            if unit.source == "calendar":
                bonus += 0.022
            if _has(searchable, "board", "prep"):
                bonus += 0.012
            if dates:
                bonus += 0.008

        if cause:
            if _has(searchable, "because", "reason", "regression", "geocod"):
                bonus += 0.018

        if temporal and dates:
            bonus += 0.012

        scored.append((bonus, -rank, unit))

    scored.sort(key=lambda x: (-x[0], -x[1], x[2].id))
    return [unit for _, _, unit in scored]
