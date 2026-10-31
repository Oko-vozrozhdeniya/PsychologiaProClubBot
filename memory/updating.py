
"""Rules for safely updating long-term memory records.

This module prepares update decisions. It does not modify the database
automatically and does not call Gemini.
"""

from dataclasses import dataclass
from typing import Optional
import re

VALID_ACTIONS = frozenset({
    "keep",
    "update",
    "conflict",
    "review",
})


def _preference_clause(text, *, self_statement=False, past=False):
    """Parse one unambiguous, positive preference, preserving object order."""
    text = text.strip()
    subject = None
    for _ in range(2):
        match = re.match(r"^(пользователь|я|мне)\s+", text)
        if match:
            if subject is not None and subject != match[1]:
                return None
            subject, text = match[1], text[match.end():]
        text = re.sub(r"^(?:теперь|сейчас)\s+", "", text)
    forms = (r"нравилось|нравились|предпочитал|предпочитала|любил|любила|"
             r"увлекался|увлекалась") if past else (
             r"предпочитаю|предпочитает|нравится|нравятся|люблю|любит|"
             r"увлекаюсь|увлекается")
    match = re.match(r"^(" + forms + r")\s+(.+)$", text)
    if not match:
        return None
    if self_statement and not (
        subject in ("я", "мне")
        or subject is None and match[1] in ("предпочитаю", "люблю", "увлекаюсь")
    ):
        return None
    obj = match[2]
    if (not re.fullmatch(r"[а-яa-z0-9\s-]+", obj)
            or re.search(r"\b(не|ни|без|кроме|если|или|и|возможно|наверное|"
                         r"больше|меньше|чем|раньше|теперь|сейчас|вместо)\b", obj)):
        return None
    activity = {
        "выращивать": "выращивание", "выращиванием": "выращивание",
        "собирать": "сборка", "сборкой": "сборка", "сборку": "сборка",
        "читать": "чтение", "чтением": "чтение",
    }
    words = []
    for word in obj.split():
        if word in activity:
            word = activity[word]
        elif len(word) > 5 and re.fullmatch(r"[а-я]+", word):
            # Only explicit plural inflections; no fuzzy prefix/substring match.
            word = re.sub(r"(?:ыми|ими|ами|ями|ые|ие|ых|их|ей|ов|ев|ах|ях|ы|и)$", "", word)
        words.append(word)
    return tuple(words) if words else None


def _resolve_preference_ellipsis(previous, current, *, self_statement=False):
    """Reuse an explicit noun only in a same-activity old/current comparison.

    For example, 'grow purple cacti, now grow yellow [cacti]'. Do not infer
    a topic from an isolated colour, another actor, or a different activity.
    """
    before = _preference_clause(previous.casefold().replace("ё", "е"),
                                self_statement=self_statement, past=True)
    after = _preference_clause(current.casefold().replace("ё", "е"),
                               self_statement=self_statement)
    if (before is None or after is None or len(after) < 2
            or before[0] != after[0] or before[0] not in ("выращивание", "сборка", "чтение")):
        return current
    old_words, new_words = previous.split()[-len(before):], current.split()[-len(after):]
    def adjective(word):
        return re.fullmatch(r"[а-яё]+(?:ые|ие|ых|их|ыми|ими)", word.casefold()) is not None
    if not all(adjective(word) for word in new_words[1:]):
        return current
    first_noun = 1
    while first_noun < len(old_words) and adjective(old_words[first_noun]):
        first_noun += 1
    if (first_noun <= 1 or first_noun == len(old_words)
            or not re.fullmatch(r"[а-яё]{3,}", old_words[first_noun].casefold())):
        return current
    return current + " " + " ".join(old_words[first_noun:])


def preference_key(content, *, self_statement=False):
    """A bounded equivalence rule, never an open-ended semantic similarity score.

    Unknown relations, negation, different subjects and compound/ambiguous
    claims fail closed. Historical text is retained in storage, not rewritten.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    # Inflection rules must never collapse names, brands or abbreviations.
    capitalized = re.findall(r"\b[А-ЯЁA-Z][а-яёa-zА-ЯЁA-Z0-9-]*", content)
    grammatical = {"пользователь", "я", "мне", "раньше", "прежде", "теперь", "сейчас", "а",
                   "люблю", "любит", "любил", "любила", "предпочитаю", "предпочитает",
                   "предпочитал", "предпочитала", "нравится", "нравятся", "нравилось", "нравились",
                   "увлекаюсь", "увлекается", "увлекался", "увлекалась"}
    if any(word.casefold() not in grammatical for word in capitalized):
        return None
    text = " ".join(content.casefold().replace("ё", "е").split()).rstrip(".! ")
    base = re.sub(r"^пользователь\s+", "", text)
    if re.match(r"^(?:раньше|прежде)\s+", base):
        parts = re.split(r",\s*(?:а\s+)?(?:теперь|сейчас)\s+", base, maxsplit=1)
        if len(parts) != 2:
            return None
        previous = re.sub(r"^(?:раньше|прежде)\s+", "", parts[0])
        if _preference_clause(previous, self_statement=self_statement, past=True) is None:
            return None
        text = _resolve_preference_ellipsis(previous, parts[1], self_statement=self_statement)
    elif " вместо " in text:
        if not re.match(r"^(?:пользователь\s+)?(?:теперь|сейчас)\s+", text):
            return None
        text, old = text.split(" вместо ", 1)
        if not re.fullmatch(r"[а-яa-z0-9\s-]+", old):
            return None
    return _preference_clause(text, self_statement=self_statement)


def memory_match_key(content, kind):
    """Group exact facts and supported paraphrases without merging conflicts."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Non-empty memory required")
    if kind not in ("fact", "preference", "hypothesis"):
        raise ValueError("Invalid memory kind")
    if kind in ("fact", "preference"):
        key = preference_key(content)
        if key is not None:
            return "preference", "preference", key
    return kind, "literal", " ".join(content.casefold().split()).rstrip(". ")


def current_memory_content(content, kind):
    """Separate supported old/current preference text without inventing values.

    Historical wording must not gain a longer life when only the current
    preference is reaffirmed by another source. The lifecycle audits the
    original representation before applying this lossless separation.
    """
    if kind not in ("fact", "preference") or preference_key(content) is None:
        return content
    text = " ".join(content.split()).rstrip(".! ")
    base = re.sub(r"^пользователь\s+", "", text, flags=re.IGNORECASE)
    if re.match(r"^(?:раньше|прежде)\s+", base, flags=re.IGNORECASE):
        parts = re.split(
            r",\s*(?:а\s+)?(?:теперь|сейчас)\s+", base, maxsplit=1,
            flags=re.IGNORECASE,
        )
        previous = re.sub(r"^(?:раньше|прежде)\s+", "", parts[0], flags=re.IGNORECASE)
        current = _resolve_preference_ellipsis(previous, parts[1])
        return current[0].upper() + current[1:]
    if re.search(r"\sвместо\s", text, flags=re.IGNORECASE):
        return re.split(r"\sвместо\s", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return content


def source_supports_memory(message_text, content, kind):
    """Only a direct self-statement can refresh a paraphrased preference."""
    if kind in ("fact", "preference") and preference_key(content) is not None:
        key = preference_key(content)
        return key is not None and preference_key(message_text, self_statement=True) == key
    if kind == "preference":
        return False
    # Exact facts may be consolidated; no guessed subject/verb transformations.
    from memory.extraction import has_statement_for_extraction
    return (has_statement_for_extraction(message_text)
            and memory_match_key(message_text, kind) == memory_match_key(content, kind))


@dataclass(frozen=True)
class UpdateDecision:
    action: str
    reason: str
    old_content: str
    new_content: str
    source: str


def _required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def make_update_decision(
    old_content,
    new_content,
    source,
    *,
    explicitly_corrected=False,
    conflict_detected=False,
):
    """Prepare a decision without writing to persistent storage.

    An explicit correction may be applied only when no conflict is flagged.
    Other meaningful changes require review.
    """
    old_content = _required_text(old_content, "old_content")
    new_content = _required_text(new_content, "new_content")
    source = _required_text(source, "source")

    if not isinstance(explicitly_corrected, bool):
        raise ValueError("explicitly_corrected must be boolean")

    if not isinstance(conflict_detected, bool):
        raise ValueError("conflict_detected must be boolean")

    if old_content == new_content:
        action = "keep"
        reason = "Content is unchanged"
    elif conflict_detected:
        action = "conflict"
        reason = "Conflicting information requires review"
    elif explicitly_corrected:
        action = "update"
        reason = "Explicit correction supplied"
    else:
        action = "review"
        reason = "Change requires confirmation"

    return UpdateDecision(
        action=action,
        reason=reason,
        old_content=old_content,
        new_content=new_content,
        source=source,
    )


def decision_to_record(decision):
    """Convert an update decision to a plain dictionary."""
    if not isinstance(decision, UpdateDecision):
        raise ValueError("decision must be an UpdateDecision")

    return {
        "action": decision.action,
        "reason": decision.reason,
        "old_content": decision.old_content,
        "new_content": decision.new_content,
        "source": decision.source,
    }
