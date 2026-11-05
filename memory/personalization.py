"""User-scoped personalization decisions; no I/O or model calls."""
from dataclasses import dataclass
import re

CATEGORIES = {"communication", "boundary", "unfinished", "approach"}
STATUSES = {"active", "paused", "resolved"}
RESPONSE_LENGTHS = {"brief", "detailed"}
LENGTH_CONTENT = {"brief": "Предпочитает краткие ответы",
                  "detailed": "Предпочитает подробные ответы"}


def communication_length(message, *, persistent=False):
    """Recognize a bounded explicit first-person preference, never infer one.

    Persistent settings require a statement about one's usual preference or
    an explicit future instruction. Imperatives for just this answer are
    useful only as a current-request override. Quoted/third-person/ambiguous
    messages and arbitrary model output are not a source of settings.
    """
    if not isinstance(message, str) or type(persistent) is not bool:
        raise ValueError("Invalid communication request")
    text = " ".join(message.casefold().replace("ё", "е").split()).strip()
    if (not text or text.startswith("/") or any(c in text for c in '«»"')
            or persistent and "?" in text):
        return None
    text = re.sub(r"^пожалуйста[, ]+", "", text).rstrip(".! ")
    text = re.sub(r",?\s+пожалуйста$", "", text)
    adjectives = {"brief": r"(?:короткие|краткие)", "detailed": r"(?:подробные|развернутые)"}
    adverbs = {"brief": r"(?:кратко|коротко)", "detailed": r"(?:подробно|развернуто)"}
    found = []
    for length in ("brief", "detailed"):
        statement = (r"(?:(?:теперь\s+)?мне\s+(?:удобнее|нравятся|подходят)\s+"
                     + adjectives[length] + r"\s+ответы|(?:я\s+)?предпочитаю\s+"
                     + adjectives[length] + r"\s+ответы)")
        future = (r"(?:всегда|впредь|в дальнейшем|с этого момента)\s+"
                  r"(?:отвечай|пиши)(?:\s+мне)?\s+" + adverbs[length])
        if re.fullmatch(statement + "|" + future, text):
            found.append(length)
        elif not persistent and re.match(
            r"^(?:сейчас\s+|в этот раз\s+)?(?:ответь|отвечай|объясни|пиши|расскажи)"
            r"(?:\s+мне)?\s+" + adverbs[length] + r"(?:[,.!: ]|$)", text
        ):
            # Reject mixed length instructions instead of choosing a side.
            if not re.search(r"\b(?:" + adverbs["detailed" if length == "brief" else "brief"] + r")\b", text):
                found.append(length)
    return found[0] if len(found) == 1 else None


@dataclass(frozen=True)
class PersonalizationItem:
    user_id: int
    category: str
    content: str
    source: str
    confirmed: bool = False
    status: str = "active"
    explicit: bool = False

    def __post_init__(self):
        if type(self.user_id) is not int or self.user_id <= 0:
            raise ValueError("Invalid user_id")
        if self.category not in CATEGORIES or self.status not in STATUSES:
            raise ValueError("Invalid category or status")
        if any(not isinstance(x, str) or not x.strip() or len(x) > 1000
               for x in (self.content, self.source)):
            raise ValueError("Invalid content or source")
        if type(self.confirmed) is not bool or type(self.explicit) is not bool:
            raise ValueError("Invalid flags")


def build_personalization(user_id, items, *, current_request="", topic_invited=False,
                          communication_preference=None):
    if type(user_id) is not int or user_id <= 0:
        raise ValueError("Invalid user_id")
    if not isinstance(items, (tuple, list)) or len(items) > 30:
        raise ValueError("Invalid items")
    if not isinstance(current_request, str) or type(topic_invited) is not bool:
        raise ValueError("Invalid options")
    if communication_preference is not None and (
            not isinstance(communication_preference, str) or communication_preference not in RESPONSE_LENGTHS):
        raise ValueError("Invalid communication preference")
    for record in items:
        if not isinstance(record, PersonalizationItem):
            raise ValueError("Invalid item")
        if record.user_id != user_id:
            raise ValueError("Cross-user record")

    active = [x for x in items if x.confirmed and x.status == "active"]
    boundaries = [x for x in active if x.category == "boundary" and x.explicit]
    preferences = [x for x in active if x.category in ("communication", "approach")]
    unfinished = [x for x in active if x.category == "unfinished"] if topic_invited else []
    instructions = [
        "Respect explicit boundaries; refusal is not psychological resistance.",
        "Do not infer diagnoses, causes, or motives from preferences.",
        "Current explicit requests override earlier non-boundary preferences.",
    ]
    instructions += ["Explicit boundary: " + x.content for x in boundaries]
    instructions += ["Communication preference: " + x.content for x in preferences]
    result = {
        "user_id": user_id,
        "instructions": instructions,
        "unfinished_topics": [x.content for x in unfinished],
        "sources": [x.source for x in boundaries + preferences + unfinished],
    }
    length = communication_length(current_request) or communication_preference
    if length is not None:
        result["response_length"] = length
    return result
