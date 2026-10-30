
"""Extract structured memory candidates from text.

This module does not call AI services, write to SQLite, or modify bot.py.
It validates and normalizes candidates produced by a separate AI adapter.
"""

import re
from dataclasses import dataclass
from typing import Iterable

VALID_KINDS = frozenset({"fact", "preference", "hypothesis"})
MAX_CONTENT_LENGTH = 2000
MAX_SOURCE_LENGTH = 500
MAX_CANDIDATES = 30


def has_statement_for_extraction(message_text):
    """Reject explicit question-only messages before any candidate can be saved.

    This is a conservative punctuation guard, not a semantic fact classifier.
    Mixed statements/questions still go through the normal extraction checks.
    """
    if not isinstance(message_text, str):
        raise ValueError("Message must be text")
    text = message_text.strip()
    if not text or text.startswith("/"):
        return False
    quoted = text.rstrip(".! ")
    if re.fullmatch(r'«[^»]*»|"[^"\r\n]*"', quoted):
        text = quoted[1:-1]
    # A question mark in a quoted title is not a question from the user.
    text = re.sub(
        r'«[^»]*»|"[^"\r\n]*"',
        lambda match: re.sub(r"[.!?\r\n]", " ", match.group()),
        text,
    )
    for match in re.finditer(r"([^.!?\r\n]+)([.!?]*)", text):
        clause, punctuation = match.groups()
        if "?" in punctuation:
            # A declaration can precede a question within one sentence.
            before_question = re.split(
                r"[,;]\s*(?=(?:как|что|какие|какой|где|when|what|how)\b)",
                clause, maxsplit=1, flags=re.IGNORECASE,
            )
            if len(before_question) == 2 and re.match(
                r"\s*(?:я|мне|мой|моя|моё|мои|i|my)\b", before_question[0],
                flags=re.IGNORECASE,
            ):
                return True
            continue
        if re.fullmatch(
            r"\W*(?:спасибо|благодарю|пожалуйста|привет|здравствуйте|thank you|thanks)\W*",
            clause, flags=re.IGNORECASE,
        ):
            continue
        if re.search(r"[^\W\d_]", clause):
            return True
    return False


@dataclass(frozen=True)
class MemoryCandidate:
    content: str
    kind: str
    source: str
    confidence: float


def _clean_text(value, field_name, max_length):
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")

    cleaned = " ".join(value.split())

    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")

    if len(cleaned) > max_length:
        raise ValueError(
            f"{field_name} exceeds {max_length} characters"
        )

    return cleaned


def validate_candidate(data):
    """Validate one candidate returned by an extraction service."""
    if not isinstance(data, dict):
        raise ValueError("Candidate must be a dictionary")

    content = _clean_text(
        data.get("content"),
        "content",
        MAX_CONTENT_LENGTH,
    )

    kind = data.get("kind")
    if not isinstance(kind, str) or kind not in VALID_KINDS:
        raise ValueError("Invalid memory kind")

    source = _clean_text(
        data.get("source"),
        "source",
        MAX_SOURCE_LENGTH,
    )

    confidence = data.get("confidence")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError("confidence must be between 0 and 1")

    return MemoryCandidate(
        content=content,
        kind=kind,
        source=source,
        confidence=float(confidence),
    )


def extract_candidates(
    raw_candidates: Iterable[dict],
    min_confidence=0.75,
):
    """Validate, filter, and deduplicate AI-generated candidates.

    This function does not independently understand a conversation.
    An AI extraction adapter will supply raw_candidates later.
    """
    if (
        isinstance(min_confidence, bool)
        or not isinstance(min_confidence, (int, float))
        or not 0.0 <= min_confidence <= 1.0
    ):
        raise ValueError("Invalid minimum confidence")

    if isinstance(raw_candidates, (str, bytes, dict)):
        raise ValueError("Candidates must be a collection")

    try:
        candidates = list(raw_candidates)
    except TypeError as exc:
        raise ValueError("Candidates must be iterable") from exc

    if len(candidates) > MAX_CANDIDATES:
        raise ValueError("Too many candidates")

    result = []
    seen = set()

    for item in candidates:
        candidate = validate_candidate(item)

        if candidate.confidence < min_confidence:
            continue

        key = (
            candidate.kind,
            candidate.content.casefold(),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(candidate)

    return result


def candidate_to_storage_record(candidate):
    """Prepare a candidate for storage.add_memory().

    Confidence is intentionally excluded because the current
    SQLite schema does not have a confidence column.
    """
    if not isinstance(candidate, MemoryCandidate):
        raise ValueError("Expected MemoryCandidate")

    return {
        "content": candidate.content,
        "kind": candidate.kind,
        "source": candidate.source,
    }
