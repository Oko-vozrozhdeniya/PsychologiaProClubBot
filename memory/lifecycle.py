"""Source-backed retention for private messages and derived memories.

This module is opt-in until bot.py is integrated. Legacy memories without
registered provenance are not migrated or deleted implicitly.
"""
import sqlite3
from datetime import datetime, timedelta, timezone
from memory import storage

RETENTION_DAYS = 365


def _moment(value):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("Expected an aware timestamp")
    if dt.tzinfo is None:
        raise ValueError("Timezone required")
    return dt.astimezone(timezone.utc)


def _iso(value):
    return _moment(value).isoformat()


def initialize_lifecycle(db_path=storage.DATABASE_PATH):
    """Initialize alongside the existing memories table, without altering it."""
    storage.initialize_database(db_path)
    with storage._connection(db_path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS private_messages (
            user_id INTEGER NOT NULL CHECK(user_id > 0),
            message_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            PRIMARY KEY(user_id, message_id)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS memory_sources (
            user_id INTEGER NOT NULL CHECK(user_id > 0),
            memory_id INTEGER NOT NULL,
            message_id TEXT NOT NULL,
            PRIMARY KEY(user_id, memory_id, message_id),
            FOREIGN KEY(user_id, message_id)
                REFERENCES private_messages(user_id, message_id)
                ON DELETE CASCADE
        )""")
        db.execute("""CREATE INDEX IF NOT EXISTS idx_private_messages_expiry
            ON private_messages(expires_at)""")
        db.execute("""CREATE INDEX IF NOT EXISTS idx_memory_sources_message
            ON memory_sources(user_id, message_id)""")
        db.execute("""CREATE TABLE IF NOT EXISTS memory_consents (
            user_id INTEGER PRIMARY KEY CHECK(user_id > 0),
            enabled INTEGER NOT NULL CHECK(enabled IN (0, 1)),
            updated_at TEXT NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS memory_revisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL CHECK(user_id > 0),
            memory_id INTEGER NOT NULL,
            previous_content TEXT NOT NULL,
            new_content TEXT NOT NULL,
            previous_source TEXT NOT NULL,
            new_source TEXT NOT NULL,
            change_kind TEXT NOT NULL CHECK(change_kind IN ('correction', 'change')),
            changed_at TEXT NOT NULL
        )""")
        db.execute("""CREATE INDEX IF NOT EXISTS idx_memory_revisions_owner
            ON memory_revisions(user_id, memory_id, id)""")


def register_message(user_id, message_id, content, created_at, *,
                     db_path=storage.DATABASE_PATH):
    """Store one private message, expiring exactly 365 days after its timestamp.

    Repeated registration cannot reset its expiry or change its content.
    """
    storage._validate_user_id(user_id)
    message_id = storage._validate_text(message_id, "message_id")
    content = storage._validate_text(content, "content")
    created = _moment(created_at)
    expiry = created + timedelta(days=RETENTION_DAYS)
    with storage._connection(db_path) as db:
        db.execute("""INSERT OR IGNORE INTO private_messages
            (user_id, message_id, content, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)""",
            (user_id, message_id, content, created.isoformat(), expiry.isoformat()))
        row = db.execute("""SELECT content, created_at, expires_at
            FROM private_messages WHERE user_id=? AND message_id=?""",
            (user_id, message_id)).fetchone()
        if (row["content"], row["created_at"], row["expires_at"]) != (
                content, created.isoformat(), expiry.isoformat()):
            raise ValueError("Conflicting message registration")
    return expiry.isoformat()


def link_memory_sources(user_id, memory_id, message_ids, now, *,
                        db_path=storage.DATABASE_PATH):
    """Attach verified, live message sources to an existing owned memory."""
    storage._validate_user_id(user_id)
    storage._validate_memory_id(memory_id)
    if not isinstance(message_ids, (list, tuple, set, frozenset)) or not message_ids:
        raise ValueError("At least one source required")
    sources = sorted({storage._validate_text(s, "message_id") for s in message_ids})
    moment = _iso(now)
    with storage._connection(db_path) as db:
        owned = db.execute("SELECT 1 FROM memories WHERE user_id=? AND id=?",
                           (user_id, memory_id)).fetchone()
        if not owned:
            raise PermissionError("Memory not owned by participant")
        for source in sources:
            exists = db.execute("""SELECT 1 FROM private_messages
                WHERE user_id=? AND message_id=? AND expires_at>?""",
                (user_id, source, moment)).fetchone()
            if not exists:
                raise PermissionError("Missing or expired participant source")
        for source in sources:
            db.execute("""INSERT OR IGNORE INTO memory_sources
                (user_id, memory_id, message_id) VALUES (?, ?, ?)""",
                (user_id, memory_id, source))


def list_live_memories(user_id, now, *, db_path=storage.DATABASE_PATH):
    """Only return memories with at least one nonexpired registered source."""
    storage._validate_user_id(user_id)
    moment = _iso(now)
    with storage._connection(db_path) as db:
        rows = db.execute("""SELECT m.id, m.user_id, m.content, m.kind,
                m.source, m.created_at, m.updated_at
            FROM memories m WHERE m.user_id=? AND EXISTS (
                SELECT 1 FROM memory_sources ms
                JOIN private_messages pm
                  ON pm.user_id=ms.user_id AND pm.message_id=ms.message_id
                WHERE ms.user_id=m.user_id AND ms.memory_id=m.id
                  AND pm.expires_at>?
            ) ORDER BY m.id DESC""", (user_id, moment)).fetchall()
        return [dict(row) for row in rows]


def purge_expired(now, *, db_path=storage.DATABASE_PATH):
    """Atomically expire messages and remove memories without live sources.

    Unlinked legacy memories are deliberately untouched, pending migration.
    """
    moment = _iso(now)
    with storage._connection(db_path) as db:
        # Snapshot memories whose source linkage is managed by this module.
        managed = db.execute("SELECT DISTINCT user_id, memory_id FROM memory_sources").fetchall()
        expired = db.execute("DELETE FROM private_messages WHERE expires_at<=?",
                             (moment,)).rowcount
        # Foreign keys must be enabled on this connection before any deletion.
        # Explicitly remove dangling links so behavior is independent of FK settings.
        db.execute("""DELETE FROM memory_sources WHERE NOT EXISTS (
            SELECT 1 FROM private_messages pm
            WHERE pm.user_id=memory_sources.user_id
              AND pm.message_id=memory_sources.message_id
        )""")
        # History is source-backed as well. Never retain revisions whose
        # previous or new source has been removed or expired.
        db.execute("""DELETE FROM memory_revisions
            WHERE NOT EXISTS (
                SELECT 1 FROM private_messages pm
                WHERE pm.user_id=memory_revisions.user_id
                  AND pm.message_id=memory_revisions.previous_source
            ) OR NOT EXISTS (
                SELECT 1 FROM private_messages pm
                WHERE pm.user_id=memory_revisions.user_id
                  AND pm.message_id=memory_revisions.new_source
            )""")
        removed = 0
        for row in managed:
            remaining = db.execute("""SELECT 1 FROM memory_sources
                WHERE user_id=? AND memory_id=? LIMIT 1""",
                (row["user_id"], row["memory_id"])).fetchone()
            if not remaining:
                db.execute("DELETE FROM memory_revisions WHERE user_id=? AND memory_id=?",
                           (row["user_id"], row["memory_id"]))
                removed += db.execute(
                    "DELETE FROM memories WHERE user_id=? AND id=?",
                    (row["user_id"], row["memory_id"])).rowcount
    return {"expired_messages": expired, "deleted_memories": removed}


def delete_message(user_id, message_id, *, db_path=storage.DATABASE_PATH):
    """Remove a participant's message and dependent unsupported memories."""
    storage._validate_user_id(user_id)
    message_id = storage._validate_text(message_id, "message_id")
    with storage._connection(db_path) as db:
        affected = db.execute("""SELECT DISTINCT memory_id FROM memory_sources
            WHERE user_id=? AND message_id=?""", (user_id, message_id)).fetchall()
        db.execute("""DELETE FROM memory_revisions
            WHERE user_id=? AND (previous_source=? OR new_source=?)""",
            (user_id, message_id, message_id))
        deleted = db.execute("DELETE FROM private_messages WHERE user_id=? AND message_id=?",
                             (user_id, message_id)).rowcount
        db.execute("DELETE FROM memory_sources WHERE user_id=? AND message_id=?",
                   (user_id, message_id))
        removed = 0
        for row in affected:
            if not db.execute("""SELECT 1 FROM memory_sources
                WHERE user_id=? AND memory_id=?""",
                (user_id, row["memory_id"])).fetchone():
                db.execute("DELETE FROM memory_revisions WHERE user_id=? AND memory_id=?",
                           (user_id, row["memory_id"]))
                removed += db.execute("DELETE FROM memories WHERE user_id=? AND id=?",
                                      (user_id, row["memory_id"])).rowcount
    return {"deleted_messages": deleted, "deleted_memories": removed}

    

def set_memory_consent(user_id, enabled, now, *, db_path=storage.DATABASE_PATH):
    """Record an explicit opt-in or opt-out. Never infer consent from identity."""
    storage._validate_user_id(user_id)
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    moment = _iso(now)
    initialize_lifecycle(db_path)
    with storage._connection(db_path) as db:
        db.execute("""INSERT INTO memory_consents (user_id, enabled, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE
                SET enabled=excluded.enabled, updated_at=excluded.updated_at
        """, (user_id, int(enabled), moment))
    return enabled


def has_memory_consent(user_id, *, db_path=storage.DATABASE_PATH):
    """Return True only for an explicit, persistently stored opt-in."""
    storage._validate_user_id(user_id)
    initialize_lifecycle(db_path)
    with storage._connection(db_path) as db:
        row = db.execute(
            "SELECT enabled FROM memory_consents WHERE user_id=?", (user_id,)
        ).fetchone()
    return bool(row and row["enabled"] == 1)


def list_live_sources(user_id, now, *, db_path=storage.DATABASE_PATH):
    """Verify source ownership and expiry from independent source records.

    Derived memory rows alone are never proof of source authorization.
    """
    storage._validate_user_id(user_id)
    moment = _iso(now)
    prefix = f"telegram:private:{user_id}:"
    with storage._connection(db_path) as db:
        rows = db.execute(
            """SELECT message_id, expires_at FROM private_messages
               WHERE user_id=? AND expires_at>? ORDER BY message_id""",
            (user_id, moment),
        ).fetchall()
    return [
        {"source": row["message_id"], "expires_at": row["expires_at"]}
        for row in rows if row["message_id"].startswith(prefix)
    ]


def revise_memory(user_id, memory_id, expected_content, expected_source,
                  new_content, new_source, message_text, now, change_kind,
                  *, db_path=storage.DATABASE_PATH):
    """Atomically revise an owned live memory while preserving its provenance.

    A new authenticated Telegram message becomes the only *current* source.
    The previous content remains in a source-bound revision history, not in
    active retrieval. Missing consent, stale content, replay, or expired
    sources fail without changing the database.
    """
    storage._validate_user_id(user_id)
    storage._validate_memory_id(memory_id)
    previous = storage._validate_text(expected_content, "expected_content")
    previous_source = storage._validate_text(expected_source, "expected_source")
    new_content = storage._validate_text(new_content, "new_content")
    new_source = storage._validate_text(new_source, "new_source")
    message_text = storage._validate_text(message_text, "message_text")
    if change_kind not in ("correction", "change"):
        raise ValueError("Invalid change kind")
    if len(new_content) > 2000:
        raise ValueError("Memory content exceeds limit")
    if not previous_source.startswith(f"telegram:private:{user_id}:"):
        raise PermissionError("Invalid previous source")
    if not new_source.startswith(f"telegram:private:{user_id}:"):
        raise PermissionError("Invalid new source")
    if new_source == previous_source:
        raise ValueError("A revision requires a new source")
    moment = _moment(now)
    expiry = (moment + timedelta(days=RETENTION_DAYS)).isoformat()
    with storage._connection(db_path) as db:
        consent = db.execute(
            "SELECT enabled FROM memory_consents WHERE user_id=?", (user_id,)
        ).fetchone()
        if not consent or consent["enabled"] != 1:
            raise PermissionError("Explicit memory consent required")
        current = db.execute(
            "SELECT content, source FROM memories WHERE user_id=? AND id=?",
            (user_id, memory_id),
        ).fetchone()
        if current is None:
            raise LookupError("Memory not owned or not found")
        if (current["content"], current["source"]) != (previous, previous_source):
            raise ValueError("Stale memory version")
        active = db.execute(
            """SELECT 1 FROM memory_sources ms
               JOIN private_messages pm ON pm.user_id=ms.user_id
                  AND pm.message_id=ms.message_id
               WHERE ms.user_id=? AND ms.memory_id=? AND pm.expires_at>?
               LIMIT 1""", (user_id, memory_id, moment.isoformat()),
        ).fetchone()
        if not active:
            raise PermissionError("Memory has no live source")
        old_message = db.execute(
            """SELECT 1 FROM private_messages WHERE user_id=?
               AND message_id=? AND expires_at>?""",
            (user_id, previous_source, moment.isoformat()),
        ).fetchone()
        if not old_message:
            raise PermissionError("Previous primary source expired")
        if db.execute(
            "SELECT 1 FROM private_messages WHERE user_id=? AND message_id=?",
            (user_id, new_source),
        ).fetchone():
            raise ValueError("Duplicate revision message")
        if previous == new_content:
            return False
        db.execute(
            """INSERT INTO private_messages(user_id, message_id, content,
               created_at, expires_at) VALUES (?, ?, ?, ?, ?)""",
            (user_id, new_source, message_text, moment.isoformat(), expiry),
        )
        db.execute(
            """INSERT INTO memory_revisions(user_id, memory_id,
               previous_content, new_content, previous_source, new_source,
               change_kind, changed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, memory_id, previous, new_content, previous_source,
             new_source, change_kind, moment.isoformat()),
        )
        db.execute(
            """UPDATE memories SET content=?, source=?, updated_at=?
               WHERE user_id=? AND id=?""",
            (new_content, new_source, moment.isoformat(), user_id, memory_id),
        )
        # Previous versions are audit history only; do not expose them as
        # competing current facts through the active coordinator.
        db.execute("DELETE FROM memory_sources WHERE user_id=? AND memory_id=?",
                   (user_id, memory_id))
        db.execute(
            """INSERT INTO memory_sources(user_id, memory_id, message_id)
               VALUES (?, ?, ?)""", (user_id, memory_id, new_source),
        )
    return True


def list_memory_revisions(user_id, memory_id, now, *,
                          db_path=storage.DATABASE_PATH):
    """Read user-owned revision history only while its current source is live."""
    storage._validate_user_id(user_id)
    storage._validate_memory_id(memory_id)
    moment = _iso(now)
    with storage._connection(db_path) as db:
        live = db.execute(
            """SELECT 1 FROM memories m
               JOIN memory_sources ms ON ms.user_id=m.user_id
                  AND ms.memory_id=m.id
               JOIN private_messages pm ON pm.user_id=ms.user_id
                  AND pm.message_id=ms.message_id
               WHERE m.user_id=? AND m.id=? AND pm.expires_at>? LIMIT 1""",
            (user_id, memory_id, moment),
        ).fetchone()
        if not live:
            return []
        rows = db.execute(
            """SELECT previous_content, new_content, change_kind, changed_at
               FROM memory_revisions WHERE user_id=? AND memory_id=?
               AND EXISTS (
                  SELECT 1 FROM private_messages old
                  WHERE old.user_id=memory_revisions.user_id
                  AND old.message_id=memory_revisions.previous_source
                  AND old.expires_at>?
               ) AND EXISTS (
                  SELECT 1 FROM private_messages new
                  WHERE new.user_id=memory_revisions.user_id
                  AND new.message_id=memory_revisions.new_source
                  AND new.expires_at>?
               ) ORDER BY id ASC""",
            (user_id, memory_id, moment, moment),
        ).fetchall()
    return [dict(row) for row in rows]
