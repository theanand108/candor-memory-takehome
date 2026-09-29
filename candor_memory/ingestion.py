from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from .models import MemoryUnit


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_units(data_dir: str | Path) -> list[MemoryUnit]:
    """Normalize every citable item in the supplied challenge data.

    Delivery/availability time is deliberately preserved separately from event
    semantics because the evaluator answers questions at an `as_of` moment.
    """
    root = Path(data_dir)
    units: list[MemoryUnit] = []

    for path in sorted((root / "native/meetings").glob("*.json")):
        meeting = json.loads(path.read_text(encoding="utf-8"))
        start = parse_dt(meeting["start"])
        for segment in meeting.get("segments", []):
            speaker = segment.get("speaker_name") or segment.get("speaker_label") or "Unknown speaker"
            units.append(MemoryUnit(
                id=segment["seg_id"], record_id=meeting["id"], source="meeting",
                unit_type="segment",
                available_at=start + timedelta(seconds=float(segment["end_s"])),
                text=segment.get("text", ""),
                metadata={
                    "title": meeting.get("title"), "speaker": speaker,
                    "speaker_label": segment.get("speaker_label"),
                    "speaker_confidence": segment.get("speaker_confidence"),
                    "channel": segment.get("channel"),
                    "meeting_start": meeting.get("start"),
                },
            ))

    for item in _jsonl(root / "native/dictation/dictations.jsonl"):
        units.append(MemoryUnit(
            id=item["id"], record_id=item["id"], source="dictation", unit_type="dictation",
            available_at=parse_dt(item["timestamp"]),
            text=item.get("cleaned_text", ""),
            metadata={
                "mode": item.get("mode"), "target_app": item.get("target_app"),
                "target_context": item.get("target_context"),
                "delivery_state": item.get("delivery_state"),
                "raw_transcript": item.get("raw_transcript"),
            },
        ))

    users = {u["id"]: u for u in json.loads((root / "connectors/slack/users.json").read_text(encoding="utf-8"))}
    channels = {c["id"]: c for c in json.loads((root / "connectors/slack/channels.json").read_text(encoding="utf-8"))}
    for item in _jsonl(root / "connectors/slack/messages.jsonl"):
        channel = channels.get(item.get("channel_id"), {})
        subtype = item.get("subtype")
        if subtype == "message_deleted":
            text = f"Slack message {item.get('target_id')} was deleted."
            unit_type = "deletion"
        elif subtype == "message_changed":
            text = item.get("text", "")
            unit_type = "edit"
        else:
            user = users.get(item.get("user"), {})
            text = item.get("text", "")
            unit_type = "message"
        user = users.get(item.get("user"), {})
        units.append(MemoryUnit(
            id=item["id"], record_id=item["id"], source="slack", unit_type=unit_type,
            available_at=parse_dt(item["ts"]), text=text,
            metadata={
                "author": user.get("real_name") or user.get("name") or item.get("user") or item.get("bot_name"),
                "author_id": item.get("user"), "channel": channel.get("name") or item.get("channel_id"),
                "channel_id": item.get("channel_id"), "subtype": subtype,
                "target_id": item.get("target_id"), "thread_parent_id": item.get("thread_parent_id"),
            },
        ))

    for item in _jsonl(root / "connectors/gmail/messages.jsonl"):
        body = item.get("body", "")
        header = f"From {item.get('from', '')} To {', '.join(item.get('to', []))}"
        if item.get("cc"):
            header += f" Cc {', '.join(item['cc'])}"
        units.append(MemoryUnit(
            id=item["id"], record_id=item["id"], source="gmail", unit_type="email",
            available_at=parse_dt(item["date"]), text=f"{header} | {item.get('subject', '')}\n{body}",
            metadata={"subject": item.get("subject"), "author": item.get("from"), "thread_id": item.get("thread_id")},
        ))

    for item in _jsonl(root / "connectors/google_calendar/events.jsonl"):
        start = item.get("start", {})
        end = item.get("end", {})
        when = f"{start.get('dateTime') or start.get('date')} to {end.get('dateTime') or end.get('date')}"
        attendees = ", ".join(a.get("email", "") for a in item.get("attendees", []))
        text = f"{item.get('summary', '')} | {when} | {item.get('location') or ''} | attendees: {attendees} | {item.get('description') or ''}"
        units.append(MemoryUnit(
            id=item["id"], record_id=item["id"], source="calendar", unit_type="event",
            available_at=parse_dt(item["updated"]), text=text,
            metadata={"title": item.get("summary"), "status": item.get("status"), "organizer": item.get("organizer")},
        ))

    for path in sorted((root / "connectors/codex/sessions").glob("*.jsonl")):
        events = list(_jsonl(path))
        if not events:
            continue
        meta, body = events[0], events[1:]
        text = "\n".join(
            f"{event.get('role', event.get('tool', event.get('type', 'event')))}: {event.get('content') or event.get('input') or event.get('output') or ''}"
            for event in body
        )
        last_time = body[-1].get("timestamp") if body else meta.get("started_at")
        units.append(MemoryUnit(
            id=meta["id"], record_id=meta["id"], source="codex", unit_type="session",
            available_at=parse_dt(last_time), text=text,
            metadata={"repo": meta.get("repo"), "cwd": meta.get("cwd")},
        ))

    conversations = json.loads((root / "connectors/chatgpt/conversations.json").read_text(encoding="utf-8"))
    for conversation in conversations:
        for message in conversation.get("messages", []):
            units.append(MemoryUnit(
                id=message["id"], record_id=conversation["id"], source="chatgpt", unit_type="message",
                available_at=parse_dt(message["create_time"]), text=message.get("content", ""),
                metadata={"title": conversation.get("title"), "author": message.get("role")},
            ))

    return sorted(units, key=lambda unit: (unit.available_at, unit.id))
