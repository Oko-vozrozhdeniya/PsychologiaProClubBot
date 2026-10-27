
"""Extract structured memory candidates from text.

This module does not call AI services, write to SQLite, or modify bot.py.
It validates and normalizes candidates produced by a separate AI adapter.
"""

from dataclasses import dataclass
from typing import Iterable

VALID_KINDS = frozenset({"fact", "preference", "hypothesis"})
MAX_CONTENT_LENGTH = 2000
MAX_SOURCE_LENGTH = 500
MAX_CANDIDATES = 30


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
