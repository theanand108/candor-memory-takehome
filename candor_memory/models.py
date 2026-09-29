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
        parts = [self.text]
        for key in ("title", "subject", "speaker", "author", "channel", "repo"):
            value = self.metadata.get(key)
            if value:
                parts.append(str(value))
        return " ".join(parts)
