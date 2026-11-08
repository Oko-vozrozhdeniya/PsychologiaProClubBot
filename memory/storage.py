
"""SQLite storage for user-scoped long-term memory.

No database is created on import. Call initialize_database() explicitly.
"""

import os
import sqlite3
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DATABASE_PATH = Path(os.environ.get("MEMORY_DB_PATH", "/app/data/memory.db"))
VALID_KINDS = frozenset({"fact", "preference", "hypothesis"})


def _prune_communication_preferences(db, now=None):
    # Explicit cleanup also works on connections with foreign keys disabled.
    db.execute("""DELETE FROM communication_preferences WHERE NOT EXISTS (
        SELECT 1 FROM private_messages pm WHERE pm.user_id=communication_preferences.user_id
        AND pm.message_id=communication_preferences.message_id"""
        + (" AND pm.expires_at>?" if now is not None else "") + ")",
        (now,) if now is not None else ())


def save_communication_preference(user_id, source, response_length, now, *, db_path=DATABASE_PATH):
    """Store only a source-verified explicit setting; superseded history stays inert."""
    from memory import lifecycle
    from memory.personalization import communication_length, RESPONSE_LENGTHS
    _validate_user_id(user_id)
    if not isinstance(source, str) or not source.startswith(f"telegram:private:{user_id}:"):
        raise PermissionError("Invalid participant source")
    if not isinstance(response_length, str) or response_length not in RESPONSE_LENGTHS:
        raise ValueError("Invalid communication preference")
    moment = lifecycle._iso(now)
    with _connection(db_path) as db:
        db.execute("BEGIN IMMEDIATE")
        lifecycle._merge_consent(db, user_id)
        witness = db.execute("""SELECT content,created_at FROM private_messages WHERE user_id=?
            AND message_id=? AND created_at<=? AND expires_at>?""", (user_id,source,moment,moment)).fetchone()
        if not witness or communication_length(witness["content"], persistent=True) != response_length:
            raise PermissionError("Explicit live communication preference required")
        if db.execute("SELECT 1 FROM communication_preferences WHERE user_id=? AND message_id=?",
                      (user_id,source)).fetchone():
            return False
        current = db.execute("""SELECT p.message_id,m.created_at FROM communication_preferences p
            JOIN private_messages m ON m.user_id=p.user_id AND m.message_id=p.message_id
            WHERE p.user_id=? AND p.status='active'""", (user_id,)).fetchone()
        def event_key(created, message_id):
            suffix = message_id.rsplit(":", 1)[-1]
            return (created, int(suffix) if suffix.isdecimal() else 0, message_id)
        if current and event_key(witness["created_at"],source) <= event_key(current["created_at"],current["message_id"]):
            return False
        db.execute("UPDATE communication_preferences SET status='superseded' WHERE user_id=? AND status='active'",
                   (user_id,))
        db.execute("INSERT INTO communication_preferences VALUES (?,?,?,'active')", (user_id,source,response_length))
        return True


def get_communication_preference(user_id, allowed_sources, now, *, db_path=DATABASE_PATH):
    """Read the active setting without falling back to superseded preferences."""
    from memory import lifecycle
    from memory.personalization import communication_length
    _validate_user_id(user_id)
    if not isinstance(allowed_sources, (set, frozenset, list, tuple)):
        raise ValueError("Verified sources required")
    moment = lifecycle._iso(now)
    with _connection(db_path) as db:
        db.execute("BEGIN")
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='communication_preferences'").fetchone():
            return None
        consent = db.execute("SELECT enabled FROM memory_consents WHERE user_id=?", (user_id,)).fetchone()
        if not consent or consent["enabled"] != 1:
            return None
        row = db.execute("""SELECT p.message_id,p.response_length,m.content FROM communication_preferences p
            JOIN private_messages m ON m.user_id=p.user_id AND m.message_id=p.message_id
            WHERE p.user_id=? AND p.status='active' AND m.created_at<=? AND m.expires_at>?""",
            (user_id,moment,moment)).fetchone()
        if (not row or row["message_id"] not in allowed_sources
                or not row["message_id"].startswith(f"telegram:private:{user_id}:")
                or communication_length(row["content"], persistent=True) != row["response_length"]):
            return None
        return {"source":row["message_id"], "response_length":row["response_length"]}


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _validate_user_id(user_id):
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("user_id must be a positive integer")


def _validate_memory_id(memory_id):
    if isinstance(memory_id, bool) or not isinstance(memory_id, int) or memory_id <= 0:
        raise ValueError("memory_id must be a positive integer")


def _validate_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


@contextmanager
def _connection(db_path=DATABASE_PATH):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 10000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(db_path=DATABASE_PATH):
    """Create schema; safe to call more than once."""
    with _connection(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL CHECK(user_id > 0),
                content TEXT NOT NULL CHECK(length(trim(content)) > 0),
                kind TEXT NOT NULL CHECK(
                    kind IN ('fact', 'preference', 'hypothesis')
                ),
                source TEXT NOT NULL CHECK(length(trim(source)) > 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_user_id
            ON memories(user_id, id)
        """)
        connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_owner_id
            ON memories(user_id, id)""")


def _relationship_table_exists(connection):
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                              "AND name='memory_relationships'").fetchone() is not None


def _memory_relationship_version(content, kind):
    """Bind an edge to the actual claim, without copying its body into the edge."""
    value = json.dumps([content, kind], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _review_memory_relationships(connection, user_id, memory_id):
    from memory.updating import review_dependents
    review_dependents(connection, user_id, memory_id)
    if _relationship_table_exists(connection):
        connection.execute("""UPDATE memory_relationships SET status='review_required'
            WHERE user_id=? AND (source_memory_id=? OR target_memory_id=?)""",
            (user_id, memory_id, memory_id))


def _delete_memory_relationships(connection, user_id, memory_id):
    from memory.updating import review_dependents
    review_dependents(connection, user_id, memory_id)
    if _relationship_table_exists(connection):
        connection.execute("""DELETE FROM memory_relationships
            WHERE user_id=? AND (source_memory_id=? OR target_memory_id=?)""",
            (user_id, memory_id, memory_id))


def _prune_memory_relationships(connection, now=None):
    """Expire edge provenance and missing endpoints, even with SQLite FKs off."""
    if not _relationship_table_exists(connection):
        return
    expiry = " AND pm.expires_at>?" if now is not None else ""
    parameters = (now, now, now) if now is not None else ()
    connection.execute("""DELETE FROM memory_relationships AS r WHERE
        NOT EXISTS (SELECT 1 FROM private_messages pm
            WHERE pm.user_id=r.user_id AND pm.message_id=r.message_id""" + expiry + """)
        OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.user_id=r.user_id
            AND m.id=r.source_memory_id AND EXISTS (
                SELECT 1 FROM memory_sources ms JOIN private_messages pm
                ON pm.user_id=ms.user_id AND pm.message_id=ms.message_id
                WHERE ms.user_id=m.user_id AND ms.memory_id=m.id""" + expiry + """))
        OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.user_id=r.user_id
            AND m.id=r.target_memory_id AND EXISTS (
                SELECT 1 FROM memory_sources ms JOIN private_messages pm
                ON pm.user_id=ms.user_id AND pm.message_id=ms.message_id
                WHERE ms.user_id=m.user_id AND ms.memory_id=m.id""" + expiry + "))", parameters)


def add_memory(user_id, content, kind, source, db_path=DATABASE_PATH):
    """Add a memory and return its ID. Initialize schema first."""
    _validate_user_id(user_id)
    content = _validate_text(content, "content")
    source = _validate_text(source, "source")
    if kind not in VALID_KINDS:
        raise ValueError("kind must be fact, preference, or hypothesis")

    now = _timestamp()
    with _connection(db_path) as connection:
        cursor = connection.execute("""
            INSERT INTO memories
                (user_id, content, kind, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, content, kind, source, now, now))
        return cursor.lastrowid


def get_memory(user_id, memory_id, db_path=DATABASE_PATH):
    """Return one memory belonging to user_id, or None."""
    _validate_user_id(user_id)
    _validate_memory_id(memory_id)
    with _connection(db_path) as connection:
        row = connection.execute("""
            SELECT id, user_id, content, kind, source,
                   created_at, updated_at
            FROM memories
            WHERE user_id = ? AND id = ?
        """, (user_id, memory_id)).fetchone()
        return dict(row) if row else None


def list_memories(user_id, limit=100, db_path=DATABASE_PATH):
    """List recent memories for one user only."""
    _validate_user_id(user_id)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    with _connection(db_path) as connection:
        rows = connection.execute("""
            SELECT id, user_id, content, kind, source,
                   created_at, updated_at
            FROM memories
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
        return [dict(row) for row in rows]


def update_memory(user_id, memory_id, content, db_path=DATABASE_PATH):
    """Update content of user's memory; return True if found."""
    _validate_user_id(user_id)
    _validate_memory_id(memory_id)
    content = _validate_text(content, "content")

    with _connection(db_path) as connection:
        previous = connection.execute("SELECT content FROM memories WHERE user_id=? AND id=?",
                                      (user_id, memory_id)).fetchone()
        cursor = connection.execute("""
            UPDATE memories
            SET content = ?, updated_at = ?
            WHERE user_id = ? AND id = ?
        """, (content, _timestamp(), user_id, memory_id))
        if previous is not None and previous["content"] != content:
            _review_memory_relationships(connection, user_id, memory_id)
        return cursor.rowcount == 1


def delete_memory(user_id, memory_id, db_path=DATABASE_PATH):
    """Delete one user's memory; return True if found."""
    _validate_user_id(user_id)
    _validate_memory_id(memory_id)

    with _connection(db_path) as connection:
        # A forgotten fact must not survive in revision history or source links.
        tables = {row["name"] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if {"private_archive_exclusions", "private_messages", "memory_sources"} <= tables:
            # An explicitly forgotten fact must not be recovered from its raw
            # source pair. Preserve only a digest until the original expiry.
            from memory.retrieval import archive_source_digest
            originals = connection.execute("""SELECT DISTINCT p.content,p.created_at,p.expires_at
                FROM private_messages p JOIN memory_sources ms
                ON p.user_id=ms.user_id AND p.message_id=ms.message_id
                WHERE ms.user_id=? AND ms.memory_id=?""", (user_id,memory_id)).fetchall()
            for original in originals:
                connection.execute("INSERT OR REPLACE INTO private_archive_exclusions VALUES (?,?,?)",
                    (user_id,archive_source_digest(original["content"],original["created_at"]),original["expires_at"]))
        if "communication_preferences" in tables:
            # Forgetting a fact extracted from an explicit style statement
            # must also remove the setting derived from that same source.
            connection.execute("""DELETE FROM communication_preferences WHERE user_id=?
                AND message_id=(SELECT source FROM memories WHERE user_id=? AND id=?)""",
                (user_id,user_id,memory_id))
        _delete_memory_relationships(connection, user_id, memory_id)
        if "memory_revisions" in tables:
            connection.execute(
                "DELETE FROM memory_revisions WHERE user_id=? AND memory_id=?",
                (user_id, memory_id),
            )
        if "memory_sources" in tables:
            connection.execute(
                "DELETE FROM memory_sources WHERE user_id=? AND memory_id=?",
                (user_id, memory_id),
            )
        if "memory_merges" in tables:
            connection.execute(
                "DELETE FROM memory_merges WHERE user_id=? AND kept_memory_id=?",
                (user_id, memory_id),
            )
        cursor = connection.execute("""
            DELETE FROM memories
            WHERE user_id = ? AND id = ?
        """, (user_id, memory_id))
        from memory.updating import prune_dependencies
        prune_dependencies(connection)
        return cursor.rowcount == 1


def list_authorized_memories(user_id, allowed_sources, db_path=DATABASE_PATH):
    """Read only one user's memories whose sources were verified upstream.

    An empty source set denies access. No candidate-count truncation occurs
    before relevance ranking. This is a scoped read, not an authorization
    decision: callers must independently verify source ownership and expiry.
    """
    _validate_user_id(user_id)
    if not isinstance(allowed_sources, (set, frozenset, list, tuple)):
        raise ValueError("allowed_sources must be a collection")
    if any(not isinstance(s, str) or not s.strip() for s in allowed_sources):
        raise ValueError("allowed_sources contains an invalid source")
    sources = sorted(set(allowed_sources))
    if not sources:
        return []
    placeholders = ",".join("?" for _ in sources)
    with _connection(db_path) as connection:
        rows = connection.execute(
            "SELECT id, user_id, content, kind, source, created_at, updated_at "
            "FROM memories WHERE user_id = ? AND source IN (" + placeholders + ") "
            "ORDER BY id DESC",
            (user_id, *sources),
        ).fetchall()
        return [dict(row) for row in rows]
