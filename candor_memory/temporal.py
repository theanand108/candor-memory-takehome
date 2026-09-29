from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable

from .models import MemoryUnit


def build_temporal_view(units: Iterable[MemoryUnit], as_of: datetime) -> list[MemoryUnit]:
    """Return the units that existed at `as_of`, with Slack edits applied.

    Deletion events remove their target from the visible view from the deletion
    timestamp onward. Edit events replace the target message text from the edit
    timestamp onward. Event records themselves remain citable evidence.
    """
    units = list(units)
    deleted_at: dict[str, datetime] = {}
    edits: dict[str, list[tuple[datetime, str]]] = defaultdict(list)

    for unit in units:
        if unit.available_at > as_of:
            continue
        if unit.source != "slack":
            continue
        if unit.unit_type == "deletion":
            target = unit.metadata.get("target_id")
            if target:
                deleted_at[target] = min(as_of, unit.available_at) if target not in deleted_at else min(deleted_at[target], unit.available_at)
        elif unit.unit_type == "edit":
            target = unit.metadata.get("target_id")
            if target:
                edits[target].append((unit.available_at, unit.text))

    visible: list[MemoryUnit] = []
    for unit in units:
        if unit.available_at > as_of:
            continue
        deleted = deleted_at.get(unit.id)
        if deleted is not None and deleted <= as_of:
            continue
        if unit.unit_type in {"edit", "deletion"}:
            # Event records are useful for auditability but should not compete
            # with the actual message for ordinary semantic retrieval.
            visible.append(unit)
            continue
        text = unit.text
        if unit.source == "slack" and unit.id in edits:
            applicable = [text for timestamp, text in edits[unit.id] if timestamp <= as_of]
            if applicable:
                text = applicable[-1]
        if text != unit.text:
            unit = MemoryUnit(unit.id, unit.record_id, unit.source, unit.unit_type,
                              unit.available_at, text, dict(unit.metadata))
        visible.append(unit)
    return visible
