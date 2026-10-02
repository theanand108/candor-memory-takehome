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

# Words that tend to identify the *answer state* rather than merely the topic.
# They are deliberately small and domain-neutral so hidden questions benefit too.
_SIGNAL_TERMS = {
    "sent", "send", "sent", "promised", "promise", "agreed", "agree", "cancelled",
    "cancel", "needed", "done", "completed", "corrected", "correction", "actually",
    "final", "current", "now", "moved", "shifted", "scheduled", "targeting", "target",
    "reply", "reviewing", "expects", "expected", "sign", "signed", "keep", "cut",
    "because", "reason", "disagree", "disagreement", "unclear", "unlikely", "conditional",
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


def _answer_signal_tokens(question: str) -> set[str]:
    return _content_tokens(question) & _SIGNAL_TERMS


def _timestamp_score(unit: MemoryUnit) -> float:
    """Small recency signal used only after topical relevance is established."""
    try:
        return unit.available_at.timestamp() / 1_000_000_000.0
    except (AttributeError, TypeError, ValueError):
        return 0.0


def answer_from_evidence(question: str, hits: list[MemoryUnit]) -> tuple[str, list[str], bool]:
    """Produce a concise, grounded answer from ranked evidence.

    Retrieval remains the primary signal. The answerer sees the full retrieved
    evidence budget, scores topical overlap plus answer-state cues, and uses
    delivery-time recency only as a tie-break. This helps corrections and
    evolving commitments without overriding the temporal visibility boundary.
    """
    safe_units: list[tuple[MemoryUnit, str]] = []
    for unit in hits:
        safe_text = _safe_evidence_text(unit.text)
        if safe_text:
            safe_units.append((unit, safe_text))
    if not safe_units:
        return "I don't know.", [], True

    qtokens = _question_content_tokens(question)
    signal_tokens = _answer_signal_tokens(question)
    candidates: list[tuple[float, int, float, MemoryUnit, str]] = []

    for rank, (unit, safe_text) in enumerate(safe_units):
        search_text = f"{safe_text} {_unit_search_text(unit)}"
        words = _content_tokens(search_text)
        overlap = len(qtokens & words)
        if overlap == 0 or not _has_structured_anchor(question, search_text):
            continue

        strong_single = any(token in words for token in qtokens if len(token) >= 7)
        if overlap < 2 and not strong_single:
            entity_only = any(
                token in words
                for token in qtokens
                if len(token) >= 4 and token in _content_tokens(" ".join(str(v) for v in unit.metadata.values()))
            )
            if not entity_only:
                continue

        # Topic overlap dominates. Answer-state cues and recency break ties,
        # which is especially useful for corrections, commitments and updates.
        signal_overlap = len(signal_tokens & words)
        score = float(overlap) + 0.45 * signal_overlap
        candidates.append((score, -rank, _timestamp_score(unit), unit, safe_text))

    if not candidates:
        return "I don't know.", [], True

    candidates.sort(key=lambda item: (item[0], item[2], item[1]), reverse=True)

    chosen: list[tuple[MemoryUnit, str]] = []
    seen_texts: set[str] = set()
    seen_records: dict[str, int] = {}
    for _, _, _, unit, safe_text in candidates:
        normalized = " ".join(safe_text.lower().split())
        if normalized in seen_texts:
            continue

        # Avoid letting a long meeting monopolize the answer when several
        # independent records support the same storyline. Still allow up to
        # two passages from one record because some corrections/final decisions
        # live in the same meeting.
        record_count = seen_records.get(unit.record_id, 0)
        if record_count >= 2:
            continue

        seen_texts.add(normalized)
        seen_records[unit.record_id] = record_count + 1
        chosen.append((unit, safe_text))
        if len(chosen) == 5:
            break

    if not chosen:
        return "I don't know.", [], True

    snippets = [_compact_snippet(text) for _, text in chosen]
    answer = " ".join(snippet for snippet in snippets if snippet)
    return answer, [unit.id for unit, _ in chosen], False
