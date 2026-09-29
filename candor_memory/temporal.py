from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable

from .models import MemoryUnit


def build_temporal_view(units: Iterable[MemoryUnit], as_of: datetime) -> list[MemoryUnit]:
    """Build the exact Slack state visible at ``as_of``.

    Slack edit/delete records are audit events, not user-visible messages. An
    edit replaces the original text from its delivery time onward; a deletion
    removes the original message entirely from that point onward.
    """
    units = list(units)
    deleted_at: dict[str, datetime] = {}
    edits: dict[str, list[tuple[datetime, str]]] = defaultdict(list)

    for unit in units:
        if unit.source != "slack" or unit.available_at > as_of:
            continue
        target = unit.metadata.get("target_id")
        if not target:
            continue
        if unit.unit_type == "deletion":
            deleted_at[target] = min(deleted_at.get(target, unit.available_at), unit.available_at)
        elif unit.unit_type == "edit":
            edits[target].append((unit.available_at, unit.text))

    visible: list[MemoryUnit] = []
    for unit in units:
        if unit.available_at > as_of:
            continue
        if unit.source == "slack" and unit.unit_type in {"edit", "deletion"}:
            continue
        deleted = deleted_at.get(unit.id)
        if deleted is not None and deleted <= as_of:
            continue

        text = unit.text
        if unit.source == "slack" and unit.id in edits:
            applicable = [text for timestamp, text in edits[unit.id] if timestamp <= as_of]
            if applicable:
                text = applicable[-1]

        if text != unit.text:
            unit = MemoryUnit(
                unit.id, unit.record_id, unit.source, unit.unit_type,
                unit.available_at, text, dict(unit.metadata),
            )
        visible.append(unit)

    return sorted(visible, key=lambda unit: (unit.available_at, unit.id))
