from __future__ import annotations

import re
from datetime import datetime

from .models import MemoryUnit

_SECRET_PATTERNS = [
    re.compile(r"(?:sk|pk|api)[-_]?[a-z0-9]{12,}", re.I),
    re.compile(r"(?:password|passwd|secret|api[_ -]?key)\s*[:=]\s*\S+", re.I),
]
_INSTRUCTION_WORDS = ("ignore previous instructions", "system prompt", "assistant must", "do not tell alex")


def sanitize(text: str) -> str:
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def looks_like_untrusted_instruction(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INSTRUCTION_WORDS)


def answer_from_evidence(question: str, hits: list[MemoryUnit]) -> tuple[str, list[str], bool]:
    """Conservative baseline answerer.

    This is intentionally extractive. A later LLM layer can improve synthesis,
    but it must receive only temporally valid evidence and remain subject to the
    same safety/abstention rules.
    """
    safe_hits = [h for h in hits if not looks_like_untrusted_instruction(h.text)]
    if not safe_hits:
        return "I don't know.", [], True

    qtokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    scored: list[tuple[float, MemoryUnit]] = []
    for unit in safe_hits:
        words = set(re.findall(r"[a-z0-9]+", unit.text.lower()))
        overlap = len(qtokens & words)
        score = overlap / max(1, len(qtokens))
        scored.append((score, unit))
    scored.sort(key=lambda x: (-x[0], x[1].available_at))
    chosen = [unit for score, unit in scored[:3] if score > 0]
    if not chosen:
        return "I don't know.", [], True

    primary = chosen[0]
    snippet = sanitize(primary.text).strip().replace("\n", " ")
    if len(snippet) > 500:
        snippet = snippet[:497].rsplit(" ", 1)[0] + "..."
    answer = snippet
    return answer, [u.id for u in chosen], False
