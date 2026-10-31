"""Upgrade rehearsal against formats written by pinned main, not live user data.

The frozen functions come from main 21498987; no Telegram/API credentials,
network access or Bothost volume are used. This does not accept a live release.
"""
import ast
import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
import zipfile

from memory import cleanup_worker as worker, lifecycle, storage
import test_bot_memory_round_trip as bot_helpers
import test_retention_operation as retention_fixtures

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
FIXTURE = Path(__file__).parent / "fixtures" / "main_21498987_storage.txt"
LEGACY_FILES = ("diagnostics.jsonl", "messages.jsonl", "messages.txt",
                "private_dialog_state.json", "private_history.jsonl", "topics.json")


class Clock(datetime):
    instant = NOW

    @classmethod
    def now(cls, tz=None):
        return cls.instant.astimezone(tz) if tz else cls.instant.replace(tzinfo=None)


def load_legacy(namespace):
    tree = ast.parse(FIXTURE.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned_main_storage", "exec"), namespace)


class MainDataCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = retention_fixtures.ActualPrivateRetentionTests(
            "test_real_handler_excludes_expired_pairs_before_provider_and_keeps_own_scope")
        self.fixture.setUp()
        # The outer helper is an unrun IsolatedAsyncioTestCase; transfer its
        # actual storage fixture cleanup to this test's active lifecycle.
        self.addCleanup(self.fixture.fixture.doCleanups)
        self.root, self.ns = self.fixture.root, self.fixture.ns
        self.db = self.fixture.db
        self.ns.update(DATA_DIR=self.root, DIALOG_STATE_FILE=self.root / "private_dialog_state.json",
                       TEXT_FILE=self.root / "messages.txt", JSON_FILE=self.root / "messages.jsonl",
                       TOPICS_FILE=self.root / "topics.json", DIAGNOSTICS_FILE=self.root / "diagnostics.jsonl",
                       DEFAULT_TOPIC_NAMES={"0": "General", "1": "General"},
                       MAX_ARCHIVE_CHARS=45000, CLUB_CHAT_ID=-9001, datetime=Clock)
        self.legacy = dict(self.ns, json=json, os=os, logger=logging.getLogger(__name__))
        load_legacy(self.legacy)
        for name in ("is_owner", "save_dialog_state", "reset_dialog", "memory_on_command",
                     "read_topic_names", "save_topic_name", "read_archive", "build_archive_context",
                     "save_message", "track_topic_changes", "answer_archive_question", "analyze"):
            bot_helpers.load_function(name, self.ns)
        Clock.instant = NOW
        self.legacy["save_dialog_state"]({"101": 2, "202": 1})
        for user, session, text, age in (
                (101, 1, "CONTROL_PRE_RESET_PRIVATE", 3),
                (101, 2, "CONTROL_CURRENT_PRIVATE", 2),
                (202, 1, "CONTROL_OTHER_USER_PRIVATE", 1),
                (101, 2, "CONTROL_EXPIRED_PRIVATE", 366)):
            Clock.instant = NOW - timedelta(days=age)
            self.legacy["append_dialog_record"](user, session, text, "CONTROL_REPLY")
        Clock.instant = NOW
        self.legacy["save_topic_name"](7, "CONTROL_CLUB_TOPIC")
        self.group_message = SimpleNamespace(id=42, text="CONTROL_CLUB_MESSAGE", message_thread_id=7,
                                             from_user=SimpleNamespace(id=202))
        self.group_update = SimpleNamespace(effective_message=self.group_message,
                                             effective_chat=SimpleNamespace(id=-9001))
        await self.legacy["save_message"](self.group_update, SimpleNamespace())
        (self.root / "diagnostics.jsonl").write_text(json.dumps({"timestamp": NOW.isoformat(),
            "service": "CONTROL_SERVICE", "operation": "CONTROL_OPERATION"}) + "\n")
        # A legacy volume has no explicit opt-in simply because it belongs to
        # the owner. Remove only fixture consent, never an actual database.
        with storage._connection(self.db) as db:
            db.execute("DELETE FROM memory_consents")
        self.update, self.context = self.fixture.update, self.fixture.context
        self.update.effective_message.date = NOW
        self.before = self.file_bytes()

    def file_bytes(self):
        return {name: (self.root / name).read_bytes() for name in LEGACY_FILES}

    async def enable_consent(self):
        await self.ns["memory_on_command"](self.update, self.context)

    def texts(self, history):
        return "\n".join(p["text"] for record in history for p in record["parts"])

    def test_reader_keeps_legacy_user_session_and_original_date_boundaries(self):
        history = self.ns["read_recent_dialog"](101, 2)
        self.assertIn("CONTROL_CURRENT_PRIVATE", self.texts(history))
        for excluded in ("CONTROL_PRE_RESET_PRIVATE", "CONTROL_OTHER_USER_PRIVATE", "CONTROL_EXPIRED_PRIVATE"):
            self.assertNotIn(excluded, self.texts(history))
        self.assertEqual(self.file_bytes(), self.before)

    def test_new_append_keeps_old_rows_and_adds_original_message_dates(self):
        history = self.root / "private_history.jsonl"
        self.ns["append_dialog_record"](101, 2, "CONTROL_NEW_QUESTION", "CONTROL_NEW_ANSWER",
            question_created_at=NOW - timedelta(hours=1), answer_created_at=NOW)
        self.assertTrue(history.read_bytes().startswith(self.before[history.name]))
        last = json.loads(history.read_text().splitlines()[-1])
        self.assertEqual(last["question_timestamp"], (NOW - timedelta(hours=1)).isoformat())
        self.assertEqual(last["answer_timestamp"], NOW.isoformat())
        self.assertIn("CONTROL_NEW_QUESTION", self.texts(self.ns["read_recent_dialog"](101, 2)))
        for name in LEGACY_FILES:
            if name != history.name:
                self.assertEqual((self.root / name).read_bytes(), self.before[name])

    async def test_reset_preserves_legacy_archive_and_other_users_session(self):
        await self.ns["reset_dialog"](self.update, self.context)
        self.assertEqual(json.loads((self.root / "private_dialog_state.json").read_text()), {"101": 3, "202": 1})
        self.assertEqual(self.ns["read_recent_dialog"](101, 3), [])
        for name in LEGACY_FILES:
            if name != "private_dialog_state.json":
                self.assertEqual((self.root / name).read_bytes(), self.before[name])

    async def test_legacy_volume_without_consent_never_reaches_provider_or_extraction(self):
        await self.ns["private_message"](self.update, self.context)
        self.assertEqual(self.fixture.requests, [])
        self.assertEqual(self.fixture.fixture.calls, 0)
        self.assertEqual(self.file_bytes(), self.before)
        self.assertIn("/memory_on", self.update.effective_message.reply_text.await_args.args[0])

    async def test_explicit_opt_in_allows_only_current_own_legacy_history(self):
        await self.enable_consent()
        await self.ns["private_message"](self.update, self.context)
        self.ns["send_error_response"].assert_not_awaited()
        self.assertEqual(len(self.fixture.requests), 1)
        _, history = self.fixture.requests[0]
        self.assertIn("CONTROL_CURRENT_PRIVATE", self.texts(history))
        for excluded in ("CONTROL_PRE_RESET_PRIVATE", "CONTROL_OTHER_USER_PRIVATE", "CONTROL_EXPIRED_PRIVATE", "CONTROL_CLUB_MESSAGE"):
            self.assertNotIn(excluded, self.texts(history))
        # Only the new explicit statement is a source. The existing archive is
        # not silently re-extracted into durable facts during an upgrade.
        with storage._connection(self.db) as db:
            rows = db.execute("SELECT content,created_at FROM private_messages").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], self.update.effective_message.text)
        self.assertEqual(rows[0]["created_at"], NOW.isoformat())

    async def test_first_maintenance_initializes_missing_db_and_preserves_club_data(self):
        fresh = self.root / "legacy-only-volume"
        fresh.mkdir()
        for name, body in self.before.items():
            (fresh / name).write_bytes(body)
        self.ns.update(DATA_DIR=fresh, DIALOG_HISTORY_FILE=fresh / "private_history.jsonl")
        self.ns["personal_storage"].DATABASE_PATH = fresh / "memory.db"
        await self.ns["start_retention_cleanup"](self.fixture.app)
        await self.ns["stop_retention_cleanup"](self.fixture.app)
        self.assertTrue((fresh / "memory.db").is_file())
        self.assertFalse(lifecycle.has_memory_consent(101, db_path=fresh / "memory.db"))
        self.assertEqual(storage.list_memories(101, db_path=fresh / "memory.db"), [])
        records = [json.loads(line) for line in (fresh / "private_history.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertNotIn("CONTROL_EXPIRED_PRIVATE", (fresh / "private_history.jsonl").read_text())
        self.assertTrue(all("question_timestamp" not in row for row in records))
        for name in LEGACY_FILES:
            if name != "private_history.jsonl":
                self.assertEqual((fresh / name).read_bytes(), self.before[name])
        self.assertEqual(self.file_bytes(), self.before)

    def test_club_context_from_main_format_is_compatible_and_excludes_private_files(self):
        expected = self.legacy["build_archive_context"](self.legacy["read_archive"]())
        context = self.ns["build_archive_context"](self.ns["read_archive"]())
        self.assertEqual(context, expected)
        self.assertIn("CONTROL_CLUB_MESSAGE", context)
        self.assertIn("CONTROL_CLUB_TOPIC", context)
        self.assertNotIn("CONTROL_CURRENT_PRIVATE", context)
        self.assertEqual(self.file_bytes(), self.before)

    async def test_group_collection_appends_to_both_existing_archives(self):
        self.group_message.id, self.group_message.text = 43, "CONTROL_NEW_CLUB_MESSAGE"
        await self.ns["save_message"](self.group_update, SimpleNamespace())
        for name in ("messages.txt", "messages.jsonl"):
            self.assertTrue((self.root / name).read_bytes().startswith(self.before[name]))
            self.assertIn("CONTROL_NEW_CLUB_MESSAGE", (self.root / name).read_text())
        self.assertEqual([r["message_id"] for r in self.ns["read_archive"]()], [42, 43])
        for name in LEGACY_FILES:
            if name not in ("messages.txt", "messages.jsonl"):
                self.assertEqual((self.root / name).read_bytes(), self.before[name])

    async def test_topic_edit_keeps_existing_names_and_other_data(self):
        self.group_message.forum_topic_created = None
        self.group_message.forum_topic_edited = SimpleNamespace(name="CONTROL_RENAMED_TOPIC")
        await self.ns["track_topic_changes"](self.group_update, SimpleNamespace())
        names = self.ns["read_topic_names"]()
        self.assertEqual(names["7"], "CONTROL_RENAMED_TOPIC")
        self.assertEqual(names["0"], "General")
        for name in LEGACY_FILES:
            if name != "topics.json":
                self.assertEqual((self.root / name).read_bytes(), self.before[name])

    async def test_analyze_uses_old_club_archive_without_private_context(self):
        self.ns["request_gemini"] = Mock(return_value="CONTROL_ARCHIVE_ANSWER")
        self.context.args = ["CONTROL_ARCHIVE_QUESTION"]
        await self.ns["analyze"](self.update, self.context)
        question, context = self.ns["request_gemini"].call_args.args
        self.assertEqual(question, "CONTROL_ARCHIVE_QUESTION")
        self.assertIn("CONTROL_CLUB_MESSAGE", context)
        self.assertNotIn("CONTROL_CURRENT_PRIVATE", context)
        self.ns["send_long_message"].assert_awaited_once_with(self.update.effective_message, "CONTROL_ARCHIVE_ANSWER")
        self.assertEqual(self.file_bytes(), self.before)

    async def test_participant_cannot_send_old_club_archive_to_provider(self):
        self.update.effective_user.id = 202
        self.ns["request_gemini"] = Mock()
        self.context.args = ["CONTROL_ARCHIVE_QUESTION"]
        await self.ns["analyze"](self.update, self.context)
        self.ns["request_gemini"].assert_not_called()
        self.assertEqual(self.file_bytes(), self.before)

    async def test_broken_legacy_state_is_not_silently_reset_during_upgrade(self):
        await self.enable_consent()
        state = self.root / "private_dialog_state.json"
        state.write_text("CONTROL_BROKEN_STATE")
        before = self.file_bytes()
        await self.ns["private_message"](self.update, self.context)
        self.assertEqual(self.fixture.requests, [])
        self.ns["send_error_response"].assert_awaited_once()
        self.assertEqual(self.file_bytes(), before)

    def test_old_reader_demonstrates_why_rollback_cannot_resume_processing_blindly(self):
        self.assertIn("CONTROL_EXPIRED_PRIVATE", self.texts(self.legacy["read_recent_dialog"](101, 2)))
        self.assertNotIn("CONTROL_EXPIRED_PRIVATE", self.texts(self.ns["read_recent_dialog"](101, 2)))

    def test_private_snapshot_is_not_a_full_backup_of_the_six_legacy_files(self):
        snapshot = self.root / "private.zip"
        worker.create_snapshot(self.root, snapshot, NOW, db_path=self.db)
        with zipfile.ZipFile(snapshot) as archive:
            names = set(archive.namelist())
        self.assertIn("private_history.jsonl", names)
        self.assertTrue({"messages.txt", "messages.jsonl", "topics.json", "diagnostics.jsonl"}.isdisjoint(names))
        self.assertEqual(self.file_bytes(), self.before)


if __name__ == "__main__":
    unittest.main()
