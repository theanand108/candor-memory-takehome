from __future__ import annotations

import re

from .models import MemoryUnit

_SECRET_PATTERNS = [
    re.compile(r"(?:sk|pk|api)[-_]?[a-z0-9][a-z0-9_-]{11,}", re.I),
    re.compile(r"(?:password|passwd|secret|api[_ -]?key)\s*[:=]\s*\S+", re.I),
]

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

_QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "be", "did", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your", "we", "our", "they", "their",
    "he", "she", "them", "his", "her", "than", "then", "still", "just",
}


def sanitize(text: str) -> str:
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def _safe_evidence_text(text: str) -> str:
    """Remove instruction-bearing sentences while retaining nearby facts."""
    text = sanitize(text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    safe = [part.strip() for part in parts if part.strip() and not looks_like_untrusted_instruction(part)]
    return " ".join(safe)


def looks_like_untrusted_instruction(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INSTRUCTION_MARKERS)


def _compact_snippet(text: str, max_words: int = 24) -> str:
    snippet = _safe_evidence_text(text).strip().replace("\n", " ")
    words = snippet.split()
    if len(words) <= max_words:
        return snippet
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def _content_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _unit_search_text(unit: MemoryUnit) -> str:
    metadata = " ".join(str(v) for v in unit.metadata.values() if isinstance(v, (str, int, float)))
    return f"{unit.text} {metadata}"


def _question_content_tokens(question: str) -> set[str]:
    return {
        token
        for token in _content_tokens(question)
        if token not in _QUESTION_STOPWORDS and len(token) >= 3
    }


def _has_structured_anchor(question: str, text: str) -> bool:
    lowered = question.lower()
    if " about " not in lowered:
        return True

    before, after = lowered.split(" about ", 1)
    before_tokens = {t for t in _question_content_tokens(before) if len(t) >= 4}
    after_tokens = _question_content_tokens(after)
    text_tokens = _content_tokens(text)
    return bool(before_tokens & text_tokens) and bool(after_tokens & text_tokens)


def answer_from_evidence(question: str, hits: list[MemoryUnit]) -> tuple[str, list[str], bool]:
    """Produce a concise, grounded answer from ranked evidence.

    Retrieval remains the primary signal. We score evidence for topical overlap,
    but use metadata as well as body text so speaker/author/entity questions do
    not abstain merely because a name lives in metadata. Instruction-bearing
    sentences are removed rather than discarding an otherwise useful record.
    """
    safe_units: list[tuple[MemoryUnit, str]] = []
    for unit in hits:
        safe_text = _safe_evidence_text(unit.text)
        if safe_text:
            safe_units.append((unit, safe_text))
    if not safe_units:
        return "I don't know.", [], True

    qtokens = _question_content_tokens(question)
    candidates: list[tuple[int, int, MemoryUnit, str]] = []

    for rank, (unit, safe_text) in enumerate(safe_units):
        search_text = f"{safe_text} {_unit_search_text(unit)}"
        words = _content_tokens(search_text)
        overlap = len(qtokens & words)
        if overlap == 0 or not _has_structured_anchor(question, search_text):
            continue

        strong_single = any(token in words for token in qtokens if len(token) >= 7)
        if overlap < 2 and not strong_single:
            # A named entity in metadata is enough for direct entity questions.
            entity_only = any(
                token in words
                for token in qtokens
                if len(token) >= 4 and token in _content_tokens(" ".join(str(v) for v in unit.metadata.values()))
            )
            if not entity_only:
                continue

        candidates.append((overlap, -rank, unit, safe_text))

    if not candidates:
        return "I don't know.", [], True

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    chosen: list[tuple[MemoryUnit, str]] = []
    seen_texts: set[str] = set()
    for _, _, unit, safe_text in candidates:
        normalized = " ".join(safe_text.lower().split())
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        chosen.append((unit, safe_text))
        if len(chosen) == 5:
            break

    snippets = [_compact_snippet(text) for _, text in chosen]
    answer = " ".join(snippet for snippet in snippets if snippet)
    return answer, [unit.id for unit, _ in chosen], False
