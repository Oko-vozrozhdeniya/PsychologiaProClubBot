"""Fail-closed coordination of existing memory modules; no bot or AI calls.

The current storage schema has no consent, entitlement, source-expiry or
cascade-deletion fields. Callers must supply an explicit verified policy;
missing/expired permissions never expose personal memories.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from memory.extraction import extract_candidates, candidate_to_storage_record
from memory.personalization import build_personalization
from memory.progress import summary as progress_summary
from memory.retrieval import rank_memories
from memory.updating import make_update_decision
from memory import storage, lifecycle


def _utc(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp required")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid timestamp") from exc
    if result.tzinfo is None:
        raise ValueError("Timezone required")
    return result.astimezone(timezone.utc)


def _user(user_id):
    if type(user_id) is not int or user_id <= 0:
        raise ValueError("Invalid user")


@dataclass(frozen=True)
class AccessPolicy:
    user_id: int
    consent: bool
    entitled: bool
    verified: bool
    expires_at: str

    def __post_init__(self):
        _user(self.user_id)
        if any(type(flag) is not bool for flag in
               (self.consent, self.entitled, self.verified)):
            raise ValueError("Invalid permission flags")
        _utc(self.expires_at)


@dataclass(frozen=True)
class SourcePermission:
    source: str
    expires_at: str
    verified: bool = True

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Invalid source")
        if type(self.verified) is not bool:
            raise ValueError("Invalid source verification")
        _utc(self.expires_at)


def _allowed(user_id, policy, now):
    _user(user_id)
    moment = _utc(now)
    if not isinstance(policy, AccessPolicy) or policy.user_id != user_id:
        raise PermissionError("Missing or cross-user policy")
    if not (policy.verified and policy.consent and policy.entitled):
        raise PermissionError("Memory access not permitted")
    if _utc(policy.expires_at) <= moment:
        raise PermissionError("Permission expired")
    return moment


def _sources(permissions, now):
    if not isinstance(permissions, (list, tuple)):
        raise ValueError("Source permissions required")
    verified = {}
    for item in permissions:
        if not isinstance(item, SourcePermission):
            raise ValueError("Invalid source permission")
        if item.source in verified:
            raise ValueError("Ambiguous duplicate source")
        verified[item.source] = item
    return {name for name, item in verified.items()
            if item.verified and _utc(item.expires_at) > now}


def read_context(user_id, query, policy, source_permissions, now, *,
                 db_path=None, limit=5, personalization_items=(),
                 checkpoint=None, topic_invited=False, enforce_lifecycle=False):
    """Return only explicitly authorized, unexpired source-backed context.

    The source list must be independently verified by the caller; a memory's
    own source string alone is never proof of validity or retention.
    """
    moment = _allowed(user_id, policy, now)
    allowed_sources = _sources(source_permissions, moment)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query required")
    if not allowed_sources:
        return {"memories": [], "personalization": None, "progress": None}
    # Filter at the database boundary, before ranking or result limits.
    # The source permissions must have been independently verified.
    records = storage.list_authorized_memories(
        user_id, allowed_sources,
        **({"db_path": db_path} if db_path is not None else {})
    )
    if enforce_lifecycle:
        kwargs = {"db_path": db_path} if db_path is not None else {}
        live = lifecycle.list_live_memories(user_id, moment, **kwargs)
        live_ids = {item["id"] for item in live}
        records = [item for item in records if item["id"] in live_ids]
    safe = rank_memories(query, records, limit=limit)
    # These modules do not persist their own provenance/expiry. Therefore
    # callers may pass only source-verified items; reject mixed inputs.
    for item in personalization_items:
        if item.source not in allowed_sources:
            raise PermissionError("Unverified personalization source")
    personalization = (build_personalization(user_id, personalization_items,
                       topic_invited=topic_invited) if personalization_items else None)
    progress = None
    if checkpoint is not None:
        if checkpoint.source not in allowed_sources:
            raise PermissionError("Unverified checkpoint source")
        if any(x.source not in allowed_sources for x in
               checkpoint.actions + checkpoint.results):
            raise PermissionError("Unverified checkpoint event")
        progress = progress_summary(checkpoint, user_id)
    return {"memories": safe, "personalization": personalization,
            "progress": progress}


def prepare_candidates(user_id, raw_candidates, policy, source_permissions,
                       now, min_confidence=0.75):
    """Validate candidates for later storage; never persist automatically."""
    moment = _allowed(user_id, policy, now)
    allowed_sources = _sources(source_permissions, moment)
    candidates = extract_candidates(raw_candidates, min_confidence)
    if any(item.source not in allowed_sources for item in candidates):
        raise PermissionError("Unverified candidate source")
    return [candidate_to_storage_record(item) for item in candidates]


def prepare_correction(user_id, old_content, new_content, source, policy,
                       source_permissions, now, *, explicitly_corrected=False,
                       conflict_detected=False):
    moment = _allowed(user_id, policy, now)
    if source not in _sources(source_permissions, moment):
        raise PermissionError("Unverified correction source")
    return make_update_decision(old_content, new_content, source,
                                explicitly_corrected=explicitly_corrected,
                                conflict_detected=conflict_detected)
