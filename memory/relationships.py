
"""Relationships between memory records and people.

Pure validators are preserved. Persistent edges require owned live endpoint
snapshots, explicit stored consent and an independently registered source.
No psychological causes or relations are inferred by this module.
"""

from dataclasses import asdict, dataclass
from typing import Optional
from memory import storage, lifecycle

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


def save_relationship(relationship, source_snapshot, target_snapshot, now, *,
                      explicitly_confirmed=False, db_path=storage.DATABASE_PATH):
    """Save one source-backed edge, atomically rechecking both endpoint versions.

    A confirmed edge requires an explicit confirmation from the upstream
    trusted caller; a model guess must remain a hypothesis. Replaying an old
    witness cannot reactivate an edge invalidated by a correction.
    """
    if not isinstance(relationship, MemoryRelationship):
        raise ValueError("Expected MemoryRelationship")
    if type(explicitly_confirmed) is not bool:
        raise ValueError("Invalid confirmation flag")
    if relationship.evidence == "confirmed" and not explicitly_confirmed:
        raise PermissionError("Explicit relationship confirmation required")
    if len(relationship.description) > 500:
        raise ValueError("Relationship description is too long")
    user = relationship.user_id
    prefix = f"telegram:private:{user}:"
    if not relationship.source.startswith(prefix) or not relationship.source[len(prefix):]:
        raise PermissionError("Invalid relationship source")
    moment = lifecycle._iso(now)
    with storage._connection(db_path) as db:
        db.execute("BEGIN IMMEDIATE")
        lifecycle._merge_consent(db, user)
        left = lifecycle._snapshot_memory(db, user, source_snapshot)
        right = lifecycle._snapshot_memory(db, user, target_snapshot)
        if (left is None or right is None or left["id"] != relationship.source_memory_id
                or right["id"] != relationship.target_memory_id):
            raise ValueError("Stale or mismatched relationship endpoints")
        if relationship.evidence == "confirmed" and "hypothesis" in (left["kind"], right["kind"]):
            raise PermissionError("Hypothetical endpoints cannot form a confirmed claim")
        for endpoint in (left, right):
            links = lifecycle._memory_links(db, user, endpoint["id"], moment)
            if not any(link["message_id"] == endpoint["source"]
                       and link["created_at"] <= moment for link in links):
                raise PermissionError("Relationship endpoint has no live current source")
        witness = db.execute("""SELECT 1 FROM private_messages
            WHERE user_id=? AND message_id=? AND created_at<=? AND expires_at>?""",
            (user, relationship.source, moment, moment)).fetchone()
        if not witness:
            raise PermissionError("Relationship source is missing or expired")
        versions = (storage._memory_relationship_version(left["content"], left["kind"]),
                    storage._memory_relationship_version(right["content"], right["kind"]))
        db.execute("""INSERT OR IGNORE INTO memory_relationships
            (user_id,source_memory_id,target_memory_id,relation_type,evidence,message_id,
             description,source_version,target_version,status,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,'active',?)""",
            (user,left["id"],right["id"],relationship.relation_type,relationship.evidence,
             relationship.source,relationship.description,*versions,moment))
        row = db.execute("""SELECT * FROM memory_relationships WHERE user_id=?
            AND source_memory_id=? AND target_memory_id=? AND relation_type=?
            AND evidence=? AND message_id=?""", (user,left["id"],right["id"],
            relationship.relation_type,relationship.evidence,relationship.source)).fetchone()
        if (row["status"] != "active" or row["source_version"] != versions[0]
                or row["target_version"] != versions[1]
                or row["description"] != relationship.description):
            raise ValueError("Relationship witness already used for another version")
        return row["id"]


def list_live_relationships(user_id, memory_ids, allowed_sources, now, *,
                            limit=100, db_path=storage.DATABASE_PATH):
    """Read edges only between already selected, owned and source-allowed facts.

    Consent, witness lifetime, both current endpoints and their registered
    primary sources are rechecked in one database snapshot. Read does not
    initialize/migrate old schemas or reactivate reviewed edges.
    """
    _positive_id(user_id, "user_id")
    if not isinstance(memory_ids, (list, tuple, set, frozenset)) or len(memory_ids) > 50:
        raise ValueError("At most fifty selected memory IDs required")
    ids = sorted({_positive_id(value, "memory_id") for value in memory_ids})
    if not isinstance(allowed_sources, (list, tuple, set, frozenset)):
        raise ValueError("Verified source set required")
    sources = {_text(value, "source") for value in allowed_sources}
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Invalid relationship limit")
    moment = lifecycle._iso(now)
    if not ids or not sources:
        return []
    with storage._connection(db_path) as db:
        # A stable read transaction also covers the persistent consent check.
        db.execute("BEGIN")
        if not storage._relationship_table_exists(db):
            return []
        consent = db.execute("SELECT enabled FROM memory_consents WHERE user_id=?", (user_id,)).fetchone()
        if not consent or consent["enabled"] != 1:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = db.execute("""SELECT r.*, a.content AS left_content,a.kind AS left_kind,
            a.source AS left_source,b.content AS right_content,b.kind AS right_kind,
            b.source AS right_source FROM memory_relationships r
            JOIN memories a ON a.user_id=r.user_id AND a.id=r.source_memory_id
            JOIN memories b ON b.user_id=r.user_id AND b.id=r.target_memory_id
            JOIN private_messages witness ON witness.user_id=r.user_id
                AND witness.message_id=r.message_id
            WHERE r.user_id=? AND r.status='active' AND r.created_at<=?
                AND witness.created_at<=? AND witness.expires_at>?
                AND r.source_memory_id IN (""" + placeholders + ")"
                " AND r.target_memory_id IN (" + placeholders + ")" + """
                AND EXISTS (SELECT 1 FROM memory_sources ms JOIN private_messages pm
                    ON pm.user_id=ms.user_id AND pm.message_id=ms.message_id
                    WHERE ms.user_id=r.user_id AND ms.memory_id=a.id AND ms.message_id=a.source
                    AND pm.created_at<=? AND pm.expires_at>?)
                AND EXISTS (SELECT 1 FROM memory_sources ms JOIN private_messages pm
                    ON pm.user_id=ms.user_id AND pm.message_id=ms.message_id
                    WHERE ms.user_id=r.user_id AND ms.memory_id=b.id AND ms.message_id=b.source
                    AND pm.created_at<=? AND pm.expires_at>?)
            ORDER BY r.id""", (user_id,moment,moment,moment,*ids,*ids,moment,moment,moment,moment))
        result = []
        for row in rows:
            if any(row[key] not in sources or not row[key].startswith(f"telegram:private:{user_id}:")
                   for key in ("message_id", "left_source", "right_source")):
                continue
            if (row["source_version"] != storage._memory_relationship_version(row["left_content"], row["left_kind"])
                    or row["target_version"] != storage._memory_relationship_version(row["right_content"], row["right_kind"])):
                continue
            result.append({"id":row["id"],"user_id":row["user_id"],
                "source_memory_id":row["source_memory_id"],"target_memory_id":row["target_memory_id"],
                "relation_type":row["relation_type"],"evidence":row["evidence"],
                "source":row["message_id"],"description":row["description"],"created_at":row["created_at"]})
            if len(result) == limit:
                break
        return result
