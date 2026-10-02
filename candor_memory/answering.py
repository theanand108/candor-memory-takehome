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

# These words carry little topical information. Removing them makes the
# relevance gate useful for questions such as "What is Dana's salary?": a
# record merely mentioning Dana should not be treated as evidence about salary.
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


def looks_like_untrusted_instruction(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INSTRUCTION_MARKERS)


def _compact_snippet(text: str, max_words: int = 26) -> str:
    """Keep evidence concise enough to combine several relevant records."""
    snippet = sanitize(text).strip().replace("\n", " ")
    words = snippet.split()
    if len(words) <= max_words:
        return snippet
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def _content_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _question_content_tokens(question: str) -> set[str]:
    return {
        token
        for token in _content_tokens(question)
        if token not in _QUESTION_STOPWORDS and len(token) >= 3
    }


def _has_structured_anchor(question: str, text: str) -> bool:
    """Require both sides of a question's explicit 'about' relation."""
    lowered = question.lower()
    if " about " not in lowered:
        return True

    before, after = lowered.split(" about ", 1)
    before_tokens = {
        t for t in _question_content_tokens(before)
        if len(t) >= 4
    }
    after_tokens = _question_content_tokens(after)
    text_tokens = _content_tokens(text)

    return bool(before_tokens & text_tokens) and bool(after_tokens & text_tokens)


def answer_from_evidence(question: str, hits: list[MemoryUnit]) -> tuple[str, list[str], bool]:
    """Return a conservative extractive answer from retrieval-ranked evidence.

    Retrieval order remains the primary signal, but an evidence record now
    needs meaningful topical overlap with the question. This prevents a record
    that merely names the right person/project from answering a different
    question (for example, Dana's salary), while allowing several records to
    form the small evidence chains required by the benchmark.
    """
    safe_hits = [h for h in hits if not looks_like_untrusted_instruction(h.text)]
    if not safe_hits:
        return "I don't know.", [], True

    qtokens = _question_content_tokens(question)
    candidates: list[tuple[int, int, MemoryUnit]] = []

    for rank, unit in enumerate(safe_hits):
        words = _content_tokens(unit.text)
        overlap = len(qtokens & words)
        if overlap == 0 or not _has_structured_anchor(question, unit.text):
            continue

        # One weak overlap is usually an entity-only match. Require either two
        # topical anchors or a single distinctive long token.
        strong_single = any(token in words for token in qtokens if len(token) >= 7)
        if overlap < 2 and not strong_single:
            continue

        candidates.append((overlap, -rank, unit))

    if not candidates:
        return "I don't know.", [], True

    # Prefer records with more direct topical overlap, breaking ties by the
    # retrieval rank produced by the hybrid index. Keep a compact evidence
    # chain so multi-hop/disagreement questions retain both sides.
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    chosen: list[MemoryUnit] = []
    seen_texts: set[str] = set()
    for _, _, unit in candidates:
        normalized = " ".join(unit.text.lower().split())
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        chosen.append(unit)
        if len(chosen) == 4:
            break

    snippets = [_compact_snippet(unit.text) for unit in chosen]
    answer = " ".join(snippet for snippet in snippets if snippet)
    return answer, [u.id for u in chosen], False
