"""User-scoped personalization decisions; no I/O or model calls."""
from dataclasses import dataclass

CATEGORIES = {"communication", "boundary", "unfinished", "approach"}
STATUSES = {"active", "paused", "resolved"}


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


def build_personalization(user_id, items, *, current_request="", topic_invited=False):
    if type(user_id) is not int or user_id <= 0:
        raise ValueError("Invalid user_id")
    if not isinstance(items, (tuple, list)) or len(items) > 30:
        raise ValueError("Invalid items")
    if not isinstance(current_request, str) or type(topic_invited) is not bool:
        raise ValueError("Invalid options")
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
    return {
        "user_id": user_id,
        "instructions": instructions,
        "unfinished_topics": [x.content for x in unfinished],
        "sources": [x.source for x in boundaries + preferences + unfinished],
    }
