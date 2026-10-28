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
        removed = 0
        for row in managed:
            remaining = db.execute("""SELECT 1 FROM memory_sources
                WHERE user_id=? AND memory_id=? LIMIT 1""",
                (row["user_id"], row["memory_id"])).fetchone()
            if not remaining:
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
        deleted = db.execute("DELETE FROM private_messages WHERE user_id=? AND message_id=?",
                             (user_id, message_id)).rowcount
        db.execute("DELETE FROM memory_sources WHERE user_id=? AND message_id=?",
                   (user_id, message_id))
        removed = 0
        for row in affected:
            if not db.execute("""SELECT 1 FROM memory_sources
                WHERE user_id=? AND memory_id=?""",
                (user_id, row["memory_id"])).fetchone():
                removed += db.execute("DELETE FROM memories WHERE user_id=? AND id=?",
                                      (user_id, row["memory_id"])).rowcount
    return {"deleted_messages": deleted, "deleted_memories": removed}
