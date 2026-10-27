
"""Retrieve relevant user-scoped memories from SQLite.

This module reads existing memories only.
It does not initialize the database or modify stored records.
"""

import re
from datetime import datetime, timezone

from memory.storage import list_memories


MAX_CANDIDATES = 500
MAX_RESULTS = 20
DEFAULT_LIMIT = 5

STOP_WORDS = frozenset({
    "а", "без", "бы", "в", "во", "вы", "да", "для", "до",
    "его", "ее", "её", "же", "за", "и", "из", "или",
    "их", "к", "как", "какие", "какой", "какая",
    "какое", "каких", "который", "которая", "которые",
    "ко", "ли", "мне", "мой", "моя",
    "мы", "на", "над", "не", "но", "о", "об", "он",
    "она", "они", "от", "по", "под", "при", "с", "со",
    "так", "то", "у", "что", "это", "я",
})


def _validate_user_id(user_id):
    if (
        isinstance(user_id, bool)
        or not isinstance(user_id, int)
        or user_id <= 0
    ):
        raise ValueError("user_id must be a positive integer")


def _validate_limit(limit):
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_RESULTS
    ):
        raise ValueError("limit must be between 1 and 20")


def _tokens(text):
    """Return meaningful lowercase words from text."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    words = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)

    return {
        word
        for word in words
        if len(word) >= 2 and word not in STOP_WORDS
    }


def _parse_timestamp(value):
    """Parse an ISO timestamp, returning None for invalid values."""
    if not isinstance(value, str):
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _score_memory(memory, query_tokens):
    """Calculate a simple, explainable relevance score."""
    content = memory.get("content", "")
    memory_tokens = _tokens(content)

    if not query_tokens.issubset(memory_tokens):
        return 0

    score = len(query_tokens) * 10 + 10

    kind = memory.get("kind")

    if kind == "fact":
        score += 3
    elif kind == "preference":
        score += 2

    return score


def rank_memories(query, memories, limit=DEFAULT_LIMIT):
    """Rank supplied memory records without modifying them.

    Only records containing matching query words are returned.
    This function does not perform user authorization; use
    retrieve_memories() for user-scoped database retrieval.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be non-empty text")

    _validate_limit(limit)

    if not isinstance(memories, (list, tuple)):
        raise ValueError("memories must be a list or tuple")

    query_tokens = _tokens(query)

    if not query_tokens:
        return []

    ranked = []

    for memory in memories:
        if not isinstance(memory, dict):
            raise ValueError("each memory must be a dictionary")

        score = _score_memory(memory, query_tokens)

        if score <= 0:
            continue

        result = dict(memory)
        result["relevance_score"] = score
        ranked.append(result)

    ranked.sort(
        key=lambda item: (
            -item["relevance_score"],
            -int(item.get("id", 0)),
        )
    )

    return ranked[:limit]


def retrieve_memories(
    user_id,
    query,
    limit=DEFAULT_LIMIT,
    db_path=None,
):
    """Retrieve relevant memories belonging only to user_id.

    The database must already be initialized.
    Returns an empty list when no matching memories exist.

    db_path is optional and intended for isolated testing.
    """
    _validate_user_id(user_id)
    _validate_limit(limit)

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be non-empty text")

    if db_path is None:
        candidates = list_memories(
            user_id,
            limit=MAX_CANDIDATES,
        )
    else:
        candidates = list_memories(
            user_id,
            limit=MAX_CANDIDATES,
            db_path=db_path,
        )

    return rank_memories(
        query=query,
        memories=candidates,
        limit=limit,
    )
