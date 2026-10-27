
"""Relationships between memory records and people.

This module validates relationships. It does not write to a database,
infer psychological causes, or grant access to another user's data.
"""

from dataclasses import asdict, dataclass
from typing import Optional

RELATION_TYPES = frozenset({
    "same_topic",
    "same_event",
    "related_event",
    "involves_person",
    "family_relation",
    "social_relation",
})

EVIDENCE_TYPES = frozenset({
    "confirmed",
    "hypothesis",
})

PERSON_ROLES = frozenset({
    "self",
    "relative",
    "partner",
    "friend",
    "colleague",
    "client",
    "other",
    "unknown",
})


def _positive_id(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


@dataclass(frozen=True)
class MemoryRelationship:
    user_id: int
    source_memory_id: int
    target_memory_id: int
    relation_type: str
    evidence: str
    source: str
    description: str = ""

    def __post_init__(self):
        _positive_id(self.user_id, "user_id")
        _positive_id(self.source_memory_id, "source_memory_id")
        _positive_id(self.target_memory_id, "target_memory_id")

        if self.source_memory_id == self.target_memory_id:
            raise ValueError("A memory cannot link to itself")

        if self.relation_type not in RELATION_TYPES:
            raise ValueError("Invalid relation_type")

        if self.evidence not in EVIDENCE_TYPES:
            raise ValueError("Invalid evidence")

        object.__setattr__(self, "source", _text(self.source, "source"))

        if not isinstance(self.description, str):
            raise ValueError("description must be text")

        object.__setattr__(
            self, "description", self.description.strip()
        )


@dataclass(frozen=True)
class PersonReference:
    user_id: int
    person_key: str
    role: str
    source: str
    display_name: Optional[str] = None

    def __post_init__(self):
        _positive_id(self.user_id, "user_id")
        object.__setattr__(
            self, "person_key", _text(self.person_key, "person_key")
        )
        object.__setattr__(
            self, "source", _text(self.source, "source")
        )

        if self.role not in PERSON_ROLES:
            raise ValueError("Invalid person role")

        if self.display_name is not None:
            object.__setattr__(
                self,
                "display_name",
                _text(self.display_name, "display_name"),
            )


def make_relationship(
    user_id,
    source_memory_id,
    target_memory_id,
    relation_type,
    evidence,
    source,
    description="",
):
    """Create a validated relationship without saving it."""
    return MemoryRelationship(
        user_id=user_id,
        source_memory_id=source_memory_id,
        target_memory_id=target_memory_id,
        relation_type=relation_type,
        evidence=evidence,
        source=source,
        description=description,
    )


def make_person_reference(
    user_id, person_key, role, source, display_name=None
):
    """Represent a person mentioned in one user's history."""
    return PersonReference(
        user_id=user_id,
        person_key=person_key,
        role=role,
        source=source,
        display_name=display_name,
    )


def relationship_to_record(relationship):
    """Prepare a relationship for a future storage interface."""
    if not isinstance(relationship, MemoryRelationship):
        raise ValueError("Expected MemoryRelationship")
    return asdict(relationship)


def person_to_record(person):
    """Prepare a person reference for a future storage interface."""
    if not isinstance(person, PersonReference):
        raise ValueError("Expected PersonReference")
    return asdict(person)


def filter_user_relationships(user_id, relationships):
    """Return only relationships owned by the specified user."""
    _positive_id(user_id, "user_id")

    result = []
    for relationship in relationships:
        if not isinstance(relationship, MemoryRelationship):
            raise ValueError("Expected MemoryRelationship")
        if relationship.user_id == user_id:
            result.append(relationship)

    return result
