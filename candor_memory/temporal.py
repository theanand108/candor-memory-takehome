from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable

from .models import MemoryUnit


def build_temporal_view(units: Iterable[MemoryUnit], as_of: datetime) -> list[MemoryUnit]:
    """Build the temporal retrieval view visible at ``as_of``.

    Normal Slack messages are reconstructed to their latest visible text.
    Edit events are also retained as *retrieval evidence* because questions
    can explicitly ask what was changed.  Deletion events remain excluded so
    deleted content is not resurrected by retrieval.
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
            deleted_at[target] = min(
                deleted_at.get(target, unit.available_at), unit.available_at
            )
        elif unit.unit_type == "edit":
            edits[target].append((unit.available_at, unit.text))

    visible: list[MemoryUnit] = []
    audit_evidence: list[MemoryUnit] = []

    for unit in units:
        if unit.available_at > as_of:
            continue

        if unit.source == "slack" and unit.unit_type == "deletion":
            continue

        # Edit events are kept separately as retrieval evidence.  The original
        # message is still reconstructed below so ordinary state queries see
        # the edited text rather than an obsolete version.
        if unit.source == "slack" and unit.unit_type == "edit":
            audit_evidence.append(unit)
            continue

        deleted = deleted_at.get(unit.id)
        if deleted is not None and deleted <= as_of:
            continue

        text = unit.text
        if unit.source == "slack" and unit.id in edits:
            applicable = [
                text for timestamp, text in edits[unit.id]
                if timestamp <= as_of
            ]
            if applicable:
                text = applicable[-1]

        if text != unit.text:
            unit = MemoryUnit(
                unit.id,
                unit.record_id,
                unit.source,
                unit.unit_type,
                unit.available_at,
                text,
                dict(unit.metadata),
            )
        visible.append(unit)

    # Audit evidence is deliberately appended after the reconstructed state;
    # the hybrid ranker can surface it when a question explicitly asks about
    # an edit/change while normal state queries continue to prefer the current
    # message text.
    visible.extend(audit_evidence)
    return sorted(visible, key=lambda unit: (unit.available_at, unit.id))
