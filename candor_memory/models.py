from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class MemoryUnit:
    id: str
    record_id: str
    source: str
    unit_type: str
    available_at: datetime
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def searchable_text(self) -> str:
        # Keep source-specific context in the lexical corpus. In particular,
        # dictation target_context often contains the recipient/app that the
        # actual dictated text omits, while title/subject identifies records
        # whose useful evidence is split across short transcript segments.
        parts = [self.text]
        for key in (
            "title", "subject", "speaker", "author", "channel", "repo",
            "target_context", "mode", "delivery_state",
        ):
            value = self.metadata.get(key)
            if value:
                parts.append(str(value))
        return " ".join(parts)
