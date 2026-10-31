"""Coordinate private memory access, persistence and deletion; no AI calls.

Callers supply a verified access policy. Persistent writes also require
stored consent and independently verified, live participant sources.
"""
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from memory.extraction import (
    extract_candidates, candidate_to_storage_record, has_statement_for_extraction,
)
from memory.personalization import build_personalization
from memory.progress import summary as progress_summary
from memory.retrieval import rank_memories
from memory.updating import make_update_decision, memory_match_key, source_supports_memory
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


def _allowed(user_id, policy, now, *, require_consent=True):
    _user(user_id)
    moment = _utc(now)
    if not isinstance(policy, AccessPolicy) or policy.user_id != user_id:
        raise PermissionError("Missing or cross-user policy")
    if not (policy.verified and policy.entitled
            and (policy.consent or not require_consent)):
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


def _write_policy(user_id, policy, now, kwargs):
    moment = _allowed(user_id, policy, now)
    lifecycle.initialize_lifecycle(**kwargs)
    if not lifecycle.has_memory_consent(user_id, **kwargs):
        raise PermissionError("Explicit memory consent required")
    return moment


def _private_source(user_id, source):
    prefix = f"telegram:private:{user_id}:"
    if (not isinstance(source, str) or source != source.strip()
            or not source.startswith(prefix) or not source[len(prefix):]):
        raise PermissionError("Invalid participant source")


def _register_source(user_id, source, message_text, moment, kwargs):
    _private_source(user_id, source)
    expiry = lifecycle.register_message(user_id, source, message_text, moment, **kwargs)
    verified = [item for item in lifecycle.list_live_sources(user_id, moment, **kwargs)
                if item["source"] == source and item["expires_at"] == expiry]
    if len(verified) != 1:
        raise PermissionError("New memory source was not independently verified")
    return SourcePermission(source, expiry)


def consolidate_memories(user_id, policy, now, *, db_path=None, diagnostics=None):
    """Bounded, source-verified consolidation through the existing lifecycle.

    Only exact facts and unambiguous supported preference paraphrases group.
    The transaction independently rechecks snapshots, consent and sources.
    """
    kwargs = {"db_path": db_path} if db_path is not None else {}
    moment = _write_policy(user_id, policy, now, kwargs)
    if diagnostics is not None and not isinstance(diagnostics, list):
        raise ValueError("Diagnostics must be a list")
    groups = {}
    for record in lifecycle.list_live_memories(user_id, moment, **kwargs)[:500]:
        if not record["source"].startswith(f"telegram:private:{user_id}:"):
            continue
        key = memory_match_key(record["content"], record["kind"])
        groups.setdefault(key, []).append(record)
    merged = 0
    for records in groups.values():
        if len(records) < 2:
            continue
        merged += lifecycle.merge_memories(user_id, records[:20], moment,
                                          diagnostics=diagnostics, **kwargs)
    return merged


def remember_memory(user_id, content, kind, source, message_text, policy, now, *,
                    db_path=None):
    """Persist an explicitly supplied memory and its live source linkage."""
    kwargs = {"db_path": db_path} if db_path is not None else {}
    moment = _write_policy(user_id, policy, now, kwargs)
    _private_source(user_id, source)
    raw = [{"content": content, "kind": kind, "source": source, "confidence": 1.0}]
    extract_candidates(raw)  # Validate before registering a source.
    permission = _register_source(user_id, source, message_text, moment, kwargs)
    record = prepare_candidates(user_id, raw, policy, [permission], now)[0]
    for existing in lifecycle.list_live_memories(user_id, moment, **kwargs):
        if (existing["source"].startswith(f"telegram:private:{user_id}:")
                and memory_match_key(existing["content"], existing["kind"])
                == memory_match_key(record["content"], record["kind"])):
            lifecycle.confirm_memory(user_id, existing, source, moment, **kwargs)
            return existing["id"]
    return lifecycle.add_memory(user_id, record["content"], record["kind"],
                                source, moment, **kwargs)


def revise_memory(user_id, memory_id, expected_content, expected_source,
                  new_content, new_source, message_text, policy, now, change_kind,
                  *, explicitly_corrected=False, db_path=None,
                  allow_existing_source=False):
    """Apply a confirmed revision atomically, rejecting stale or expired sources."""
    kwargs = {"db_path": db_path} if db_path is not None else {}
    moment = _write_policy(user_id, policy, now, kwargs)
    _private_source(user_id, expected_source)
    _private_source(user_id, new_source)
    permissions = [SourcePermission(**item)
                   for item in lifecycle.list_live_sources(user_id, moment, **kwargs)
                   if item["source"] == expected_source]
    decision = prepare_correction(
        user_id, expected_content, new_content, expected_source,
        policy, permissions, now, explicitly_corrected=explicitly_corrected,
    )
    if decision.action != "update":
        return False
    return lifecycle.revise_memory(
        user_id, memory_id, expected_content, expected_source, new_content,
        new_source, message_text, moment, change_kind,
        allow_existing_source=allow_existing_source, **kwargs,
    )


def forget_memory(user_id, memory_id, policy, now, *, db_path=None):
    """Delete owned memory, links and revisions even after consent is withdrawn.

    Source receipts remain subject to their existing retention, preventing
    a replay of the same message from resurrecting the deleted fact.
    """
    _allowed(user_id, policy, now, require_consent=False)
    kwargs = {"db_path": db_path} if db_path is not None else {}
    storage.initialize_database(**kwargs)
    return storage.delete_memory(user_id, memory_id, **kwargs)


def save_dialog_candidates(user_id, source, message_text, raw_candidates,
                           visible_memories, policy, now, *, db_path=None):
    """Validate and persist a private extraction response behind one boundary.

    The caller supplies only the owned live memories shown to the extractor.
    Proposed revisions must still match the current stored version exactly.
    """
    kwargs = {"db_path": db_path} if db_path is not None else {}
    moment = _write_policy(user_id, policy, now, kwargs)
    _private_source(user_id, source)
    if not has_statement_for_extraction(message_text):
        return 0
    if not isinstance(raw_candidates, list) or len(raw_candidates) > 5:
        raise ValueError("Invalid memory extraction response")
    if not isinstance(visible_memories, (list, tuple)):
        raise ValueError("Visible memory snapshot required")
    validated = extract_candidates([
        {"content": item["content"], "kind": item["kind"],
         "confidence": item["confidence"], "source": source}
        for item in raw_candidates
    ], min_confidence=0.85)
    if not validated:
        # An extractor may correctly return [] for a repetition. A direct,
        # unambiguous self-confirmation of an already stored preference is
        # still a real source; it never creates or guesses a new fact.
        matches = [r for r in lifecycle.list_live_memories(user_id, moment, **kwargs)
                   if r["source"].startswith(f"telegram:private:{user_id}:")
                   and memory_match_key(r["content"], r["kind"])[:2] == ("preference", "preference")
                   and source_supports_memory(message_text, r["content"], r["kind"])]
        if matches and not any(r["source"] == source for r in lifecycle.list_live_sources(user_id, moment, **kwargs)):
            _register_source(user_id, source, message_text, moment, kwargs)
            target = min(matches, key=lambda r: r["id"])
            if lifecycle.confirm_memory(user_id, target, source, moment, **kwargs):
                consolidate_memories(user_id, policy, now, **kwargs)
        return 0
    existing = lifecycle.list_live_memories(user_id, moment, **kwargs)
    registered_sources = lifecycle.list_live_sources(user_id, moment, **kwargs)
    # Retries cannot extend the 365-day expiry, duplicate or resurrect a fact.
    if any(item["source"] == source for item in registered_sources):
        return 0
    permission = _register_source(user_id, source, message_text, moment, kwargs)
    approved = prepare_candidates(
        user_id, [dict(content=item.content, kind=item.kind,
                       confidence=item.confidence, source=item.source)
                  for item in validated],
        policy, [permission], now, min_confidence=0.85,
    )
    if not lifecycle.has_memory_consent(user_id, **kwargs):
        lifecycle.delete_message(user_id, source, **kwargs)
        return 0

    visible = {item["id"]: item for item in visible_memories}
    live_by_id = {item["id"]: item for item in existing}
    live_sources = {item["source"] for item in registered_sources}
    known = {memory_match_key(item["content"], item["kind"]): item for item in existing
             if item["source"].startswith(f"telegram:private:{user_id}:")}
    retired = set()
    saved = 0
    for candidate in approved:
        key = candidate["content"].casefold().strip()
        matching = next(item for item in raw_candidates
                        if " ".join(item["content"].split()).casefold() == key
                        and item["kind"] == candidate["kind"]
                        and item["confidence"] >= 0.85)
        target = matching.get("replaces_memory_id")
        if target is not None:
            kind = matching.get("revision_kind")
            old = live_by_id.get(target) if type(target) is int else None
            marker = (
                r"\b(раньше|теперь|сейчас|прежде|уже|стал|стала)\b"
                if kind == "change" else
                r"\b(ошибка|ошибся|ошиблась|неверно|неправильно|исправь|"
                r"исправляю|на самом деле|я не)\b"
            )
            if not (
                kind in ("correction", "change")
                and isinstance(matching.get("replaces_content"), str)
                and old is not None and target in visible
                and old["content"] == matching["replaces_content"]
                and visible[target]["content"] == old["content"]
                and visible[target]["source"] == old["source"]
                and (candidate["kind"] == old["kind"] or (
                    memory_match_key(candidate["content"], candidate["kind"])[:2] == ("preference", "preference")
                    and memory_match_key(old["content"], old["kind"])[:2] == ("preference", "preference")))
                and old["source"] in live_sources
                and matching["confidence"] >= 0.95
                and re.search(marker, message_text.casefold())
            ):
                continue
            if not lifecycle.has_memory_consent(user_id, **kwargs):
                break
            if revise_memory(
                user_id, target, old["content"], old["source"],
                candidate["content"], source, message_text, policy, now, kind,
                explicitly_corrected=True, allow_existing_source=True, **kwargs,
            ):
                old_key = memory_match_key(old["content"], old["kind"])
                known.pop(old_key, None)
                retired.add(old_key)
                live_by_id[target] = {**old, "content": candidate["content"],
                                      "source": source}
                known[memory_match_key(candidate["content"], candidate["kind"])] = live_by_id[target]
                saved += 1
            continue
        identity = memory_match_key(candidate["content"], candidate["kind"])
        if identity in retired:
            continue
        if identity in known:
            lifecycle.confirm_memory(user_id, known[identity], source, moment, **kwargs)
            continue
        if not lifecycle.has_memory_consent(user_id, **kwargs):
            break
        memory_id = lifecycle.add_memory(user_id, candidate["content"], candidate["kind"],
                                         source, moment, **kwargs)
        known[identity] = storage.get_memory(user_id, memory_id, **kwargs)
        saved += 1
    return saved
