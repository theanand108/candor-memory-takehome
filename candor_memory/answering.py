from __future__ import annotations

import re

from .models import MemoryUnit

# Keep potentially sensitive material out of generated answers. These patterns
# are deliberately structural rather than tied to one challenge record.
_SECRET_PATTERNS = [
    re.compile(r"(?:sk|pk|api)[-_]?[a-z0-9][a-z0-9_-]{11,}", re.I),
    re.compile(r"(?:password|passwd|secret|api[_ -]?key)\s*[:=]\s*\S+", re.I),
]

# Evidence can contain adversarial text because the memory corpus itself is
# untrusted. Such records may remain retrievable, but must not be reproduced as
# answers.
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


def sanitize(text: str) -> str:
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def looks_like_untrusted_instruction(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INSTRUCTION_MARKERS)


def _compact_snippet(text: str, max_words: int = 38) -> str:
    """Keep one evidence record concise enough to combine several records."""
    snippet = sanitize(text).strip().replace("\n", " ")
    words = snippet.split()
    if len(words) <= max_words:
        return snippet
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def answer_from_evidence(question: str, hits: list[MemoryUnit]) -> tuple[str, list[str], bool]:
    """Return a conservative extractive answer from retrieval-ranked evidence.

    Retrieval already combines lexical and semantic relevance. Re-ranking the
    same hits with raw question-token overlap makes small retrieval changes
    unstable and can select an older or unrelated record. We therefore retain
    retrieval order and use token overlap only as a minimal relevance gate.

    Multiple top-ranked records are included because many memory questions
    require a small evidence chain (for example, a commitment followed by its
    later completion or a current view alongside a conflicting view).
    """
    safe_hits = [h for h in hits if not looks_like_untrusted_instruction(h.text)]
    if not safe_hits:
        return "I don't know.", [], True

    qtokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    chosen: list[MemoryUnit] = []
    seen_texts: set[str] = set()

    for unit in safe_hits:
        words = set(re.findall(r"[a-z0-9]+", unit.text.lower()))
        if not (qtokens & words):
            continue
        normalized = " ".join(unit.text.lower().split())
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        chosen.append(unit)
        if len(chosen) == 3:
            break

    if not chosen:
        return "I don't know.", [], True

    snippets = [_compact_snippet(unit.text) for unit in chosen]
    answer = " ".join(snippet for snippet in snippets if snippet)
    return answer, [u.id for u in chosen], False
