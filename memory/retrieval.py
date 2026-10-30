
"""Retrieve relevant user-scoped memories from SQLite.

This module reads existing memories only.
It does not initialize the database or modify stored records.
Search supports bounded Russian word-form matching and common preference
phrases. It is still lexical retrieval, not a general semantic search engine.
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
    "ты", "тебе", "тебя", "меня", "мои", "мое", "моего", "мою", "моих",
    "чем", "чему", "кого", "кому", "кем", "кто", "где", "когда", "куда",
    "почему", "зачем", "какого", "какому", "какую", "каком", "какими",
    "помнишь", "помните", "вспомни", "вспомните", "расскажи", "расскажите",
    "пожалуйста",
})

# These phrases describe the same preference relation. Topic words must still
# match independently: liking dogs must never retrieve a preference for cats.
PREFERENCE_WORDS = frozenset({
    "люблю", "любишь", "любит", "любим", "любите", "любят",
    "любил", "любила", "любили",
    "нравится", "нравятся", "нравилось", "нравились", "нравился", "нравилась",
    "увлекаюсь", "увлекаешься", "увлекается", "увлекаемся", "увлекаетесь",
    "увлекаются", "увлекался", "увлекалась", "увлекались",
    "предпочитаю", "предпочитаешь", "предпочитает", "предпочитаем",
    "предпочитаете", "предпочитают",
})
PREFERENCE_TOKEN = "__preference__"


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
    """Return normalized topic words and explicit preference relations."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    words = re.findall(r"[^\W_]+", text.casefold().replace("ё", "е"), flags=re.UNICODE)

    return {
        PREFERENCE_TOKEN if word in PREFERENCE_WORDS else word
        for word in words
        if len(word) >= 2 and word not in STOP_WORDS
    }


def _same_russian_word_form(left, right):
    """Accept a long shared stem with short endings, never arbitrary substrings.

    Exact matches are handled first by the caller. Approximation is restricted
    to Russian words with at least four shared initial letters; at least 75%
    of the shorter word must match, with at most four remaining letters.
    This covers common forms such as книги/книгах and выращивать/выращиванием
    without enabling fuzzy matching for IDs, numbers or Latin words.
    """
    common = 0
    for first, second in zip(left, right):
        if first != second:
            break
        common += 1
    return (
        common >= 4
        and common * 4 >= min(len(left), len(right)) * 3
        and max(len(left), len(right)) - common <= 4
    )


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
    """Require every topic term, preferring exact words over word-form matches."""
    content = memory.get("content", "")
    memory_tokens = _tokens(content)

    word_forms = {}
    for word in memory_tokens:
        if len(word) >= 4 and re.fullmatch(r"[а-я]+", word):
            word_forms.setdefault(word[:4], []).append(word)

    match_quality = []
    for word in query_tokens:
        if word in memory_tokens:
            match_quality.append(2)
        elif re.fullmatch(r"[а-я]+", word) and any(
            _same_russian_word_form(word, candidate)
            for candidate in word_forms.get(word[:4], ())
        ):
            match_quality.append(1)
        else:
            return 0

    score = sum(match_quality) * 10 + 10

    kind = memory.get("kind")

    if kind == "fact":
        score += 3
    elif kind == "preference":
        score += 2

    return score


def rank_memories(query, memories, limit=DEFAULT_LIMIT):
    """Rank supplied memory records without modifying them.

    Every meaningful query term must have an exact, normalized preference,
    or bounded Russian word-form match. Unrelated records are not a fallback.
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
