"""Consistent private snapshots and isolated restore, using only temporary data."""
import ast
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import zipfile

from memory import cleanup_worker as worker, lifecycle, storage
import test_bot_memory_round_trip as bot_fixtures
from test_retention_operation import pair, source

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[2]


def dump(path):
    db = sqlite3.connect(path)
    try:
        return "\n".join(db.iterdump())
    finally:
        db.close()


def main_namespace(data_dir):
    """Execute the real main with Telegram construction/polling replaced."""
    tree = ast.parse((REPO / "bot.py").read_text())
    ns = {n.name: Mock() for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    app = MagicMock()
    builder = MagicMock()
    builder.token.return_value = builder
    builder.post_init.return_value = builder
    builder.post_stop.return_value = builder
    builder.post_shutdown.return_value = builder
    builder.build.return_value = app
    ns.update(Application=SimpleNamespace(builder=lambda: builder),
              CommandHandler=Mock(), MessageHandler=Mock(), filters=MagicMock(),
              TOKEN="CONTROL_NOT_A_TOKEN", OWNER_ID=101, DATA_DIR=data_dir,
              retention_worker=worker, print=Mock())
    bot_fixtures.load_function("main", ns)
    return ns, app


class BackupRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "live"
        self.data.mkdir()
        self.db = self.data / "memory.db"
        self.sid1, self.mid1 = source(self.db, 101, "one", NOW - timedelta(days=5))
        self.sid2, self.mid2 = source(self.db, 202, "two", NOW - timedelta(days=4))
        lifecycle.set_memory_consent(202, False, NOW - timedelta(days=3), db_path=self.db)
        self.state = self.data / "private_dialog_state.json"
        self.state.write_text('{"101": 3, "202": 1}\n')
        self.history = self.data / "private_history.jsonl"
        a = dict(pair(NOW - timedelta(days=5)), session_id=3)
        b = dict(pair(NOW - timedelta(days=4), user=202), session_id=1)
        self.history.write_text("".join(json.dumps(x) + "\n" for x in [a, b]))
        self.snapshot = self.root / "snapshot.zip"

    def capture(self):
        return worker.create_snapshot(self.data, self.snapshot, NOW)

    def assert_live_unchanged(self, before):
        self.assertEqual((dump(self.db), self.history.read_bytes(), self.state.read_bytes()), before)

    def baseline(self):
        return dump(self.db), self.history.read_bytes(), self.state.read_bytes()

    def repack(self, changes, *, update_hash=False, extra=None):
        with zipfile.ZipFile(self.snapshot) as archive:
            bodies = {n: archive.read(n) for n in archive.namelist()}
        bodies.update(changes)
        if update_hash:
            manifest = json.loads(bodies["manifest.json"])
            for name, data in changes.items():
                manifest["files"][name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            bodies["manifest.json"] = json.dumps(manifest).encode()
        changed = self.root / "changed.zip"
        with zipfile.ZipFile(changed, "w") as archive:
            for name, body in bodies.items():
                archive.writestr(name, body)
            if extra:
                archive.writestr(*extra)
        return changed

    def test_round_trip_preserves_rows_dates_consent_sessions_and_live_bytes(self):
        before = self.baseline()
        self.capture()
        target = self.root / "restored"
        result = worker.restore_snapshot(self.snapshot, target, NOW + timedelta(hours=1))
        self.assertEqual(dump(target / "memory.db"), before[0])
        self.assertEqual((target / self.history.name).read_bytes(), before[1])
        self.assertEqual((target / self.state.name).read_bytes(), before[2])
        self.assertTrue(lifecycle.has_memory_consent(101, db_path=target / "memory.db"))
        self.assertFalse(lifecycle.has_memory_consent(202, db_path=target / "memory.db"))
        self.assertIsNone(storage.get_memory(101, self.mid2, db_path=target / "memory.db"))
        self.assertEqual(result["counts"], {"memories": 2, "sources": 2, "dialogs": 2, "sessions": 2})
        self.assertTrue(result["review_required"])
        self.assert_live_unchanged(before)

    def test_real_recent_dialog_reader_keeps_reset_and_user_boundaries(self):
        self.capture()
        target = self.root / "restored"
        worker.restore_snapshot(self.snapshot, target, NOW)
        clock = Mock()
        clock.now.return_value = NOW
        ns = dict(DIALOG_HISTORY_FILE=target / self.history.name, retention_worker=worker,
                  datetime=clock, timezone=timezone, MAX_DIALOG_PAIRS=30, MAX_DIALOG_CHARS=30000,
                  logger=logging.getLogger(__name__))
        bot_fixtures.load_function("read_recent_dialog", ns)
        self.assertEqual(ns["read_recent_dialog"](101, 0), [])
        self.assertEqual(len(ns["read_recent_dialog"](101, 3)), 2)
        self.assertEqual(ns["read_recent_dialog"](101, 1), [])
        self.assertEqual(len(ns["read_recent_dialog"](202, 1)), 2)

    def test_expiry_on_restore_removes_old_sources_and_archive_keeps_fresh_dates(self):
        sid3, mid3 = source(self.db, 303, "fresh", NOW - timedelta(days=1))
        worker.append_archive_record(self.history, pair(NOW - timedelta(days=1), user=303))
        before = self.baseline()
        with sqlite3.connect(self.db) as db:
            dates = db.execute("SELECT created_at,expires_at FROM private_messages WHERE message_id=?", (sid3,)).fetchone()
        self.capture()
        target = self.root / "later"
        result = worker.restore_snapshot(self.snapshot, target, NOW + timedelta(days=360))
        self.assertEqual(result["cleanup"]["expired_messages"], 1)
        self.assertEqual(result["cleanup"]["expired_dialog_messages"], 2)
        self.assertIsNone(storage.get_memory(101, self.mid1, db_path=target / "memory.db"))
        self.assertIsNotNone(storage.get_memory(303, mid3, db_path=target / "memory.db"))
        with sqlite3.connect(target / "memory.db") as db:
            self.assertEqual(db.execute("SELECT created_at,expires_at FROM private_messages WHERE message_id=?", (sid3,)).fetchone(), dates)
        self.assertEqual([r["user_id"] for r in worker.live_dialog_records(target / self.history.name,
                          NOW + timedelta(days=360))], [202, 303])
        self.assert_live_unchanged(before)

    def test_creation_filters_already_expired_only_in_the_copy(self):
        source(self.db, 303, "expired", NOW - timedelta(days=366))
        worker.append_archive_record(self.history, pair(NOW - timedelta(days=366), user=303))
        before = self.baseline()
        result = self.capture()
        self.assertEqual(result["cleanup"]["expired_messages"], 1)
        self.assertEqual(result["cleanup"]["expired_dialog_messages"], 2)
        self.assertEqual(result["counts"]["sources"], 2)
        self.assert_live_unchanged(before)

    def test_sqlite_backup_includes_committed_wal_rows(self):
        holder = sqlite3.connect(self.db)
        self.addCleanup(holder.close)
        self.assertEqual(holder.execute("PRAGMA journal_mode=WAL").fetchone()[0], "wal")
        holder.execute("BEGIN")
        holder.execute("SELECT count(*) FROM memories").fetchone()
        sid, mid = source(self.db, 303, "wal", NOW - timedelta(days=1))
        self.assertTrue(Path(str(self.db) + "-wal").exists())
        self.capture()
        target = self.root / "wal-restore"
        worker.restore_snapshot(self.snapshot, target, NOW)
        self.assertEqual(storage.get_memory(303, mid, db_path=target / "memory.db")["source"], sid)

    def test_lease_refuses_another_process_and_succeeds_after_release(self):
        before = self.baseline()
        with worker.data_directory_lock(self.data):
            run = subprocess.run([sys.executable, "-m", "memory.cleanup_worker", "--db", str(self.db),
                                  "--data-dir", str(self.data), "--backup", str(self.snapshot)],
                                 cwd=REPO, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("BlockingIOError", run.stderr)
            self.assertFalse(self.snapshot.exists())
        self.capture()
        self.assertTrue(self.snapshot.is_file())
        self.assert_live_unchanged(before)

    def test_existing_snapshot_is_never_overwritten(self):
        self.snapshot.write_bytes(b"KEEP_EXISTING_COPY")
        with self.assertRaises(FileExistsError):
            self.capture()
        self.assertEqual(self.snapshot.read_bytes(), b"KEEP_EXISTING_COPY")

    def test_logically_cross_user_source_is_refused_even_with_valid_sqlite(self):
        with sqlite3.connect(self.db) as db:
            db.execute("UPDATE memory_sources SET user_id=202,message_id=? WHERE memory_id=?",
                       (self.sid2, self.mid1))
        with self.assertRaises(ValueError):
            self.capture()
        self.assertFalse(self.snapshot.exists())

    def test_pre_snapshot_forget_is_not_recreated_from_a_source_receipt(self):
        storage.delete_memory(101, self.mid1, db_path=self.db)
        self.capture()
        target = self.root / "restored"
        worker.restore_snapshot(self.snapshot, target, NOW)
        self.assertIsNone(storage.get_memory(101, self.mid1, db_path=target / "memory.db"))
        with sqlite3.connect(target / "memory.db") as db:
            self.assertIsNotNone(db.execute("SELECT message_id FROM private_messages WHERE message_id=?",
                                           (self.sid1,)).fetchone())

    def test_real_main_holds_lease_through_polling(self):
        ns, app = main_namespace(self.data)
        def polling():
            with self.assertRaises(BlockingIOError):
                worker.create_snapshot(self.data, self.snapshot, NOW)
        app.run_polling.side_effect = polling
        ns["main"]()
        app.run_polling.assert_called_once()
        self.capture()

    def test_existing_restore_directory_and_other_files_are_untouched(self):
        self.capture()
        target = self.root / "existing"
        target.mkdir()
        (target / "keep").write_bytes(b"KEEP")
        with self.assertRaises(FileExistsError):
            worker.restore_snapshot(self.snapshot, target, NOW)
        self.assertEqual((target / "keep").read_bytes(), b"KEEP")
        self.assertEqual(len(list(target.iterdir())), 1)

    def test_publish_failure_leaves_no_partial_snapshot_or_plaintext(self):
        before = self.baseline()
        with patch.object(worker.os, "link", side_effect=OSError("CONTROL_PRIVATE_ERROR")):
            with self.assertRaises(OSError):
                self.capture()
        self.assertFalse(self.snapshot.exists())
        self.assertEqual(list(self.root.glob(".memory-snapshot-*")), [])
        self.assert_live_unchanged(before)
        self.capture()

    def test_restore_publish_failure_cleans_stage_and_can_retry(self):
        self.capture()
        target = self.root / "restored"
        with patch.object(worker.os, "rename", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                worker.restore_snapshot(self.snapshot, target, NOW)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob(".memory-restore-*")), [])
        worker.restore_snapshot(self.snapshot, target, NOW)

    def test_corruption_is_refused_before_destination_publication(self):
        self.capture()
        bad = self.repack({"private_history.jsonl": b"{}\n"})
        target = self.root / "bad"
        with self.assertRaises(ValueError):
            worker.restore_snapshot(bad, target, NOW)
        self.assertFalse(target.exists())

    def test_matching_hash_does_not_accept_broken_database_or_json(self):
        self.capture()
        for changes in ({"memory.db": b"NOT_A_DATABASE"}, {"private_history.jsonl": b"PRIVATE_BAD_JSON"},
                        {"private_dialog_state.json": b'{"101":true}'}):
            with self.subTest(component=next(iter(changes))):
                bad = self.repack(changes, update_hash=True)
                with self.assertRaises((ValueError, sqlite3.DatabaseError)):
                    worker.restore_snapshot(bad, self.root / "bad", NOW)
                self.assertFalse((self.root / "bad").exists())

    def test_path_traversal_and_duplicate_members_are_refused(self):
        self.capture()
        for extra in [("../escape", b"BAD"), ("memory.db", b"DUPLICATE")]:
            with self.subTest(name=extra[0]):
                bad = self.repack({}, extra=extra)
                with self.assertRaises(ValueError):
                    worker.verify_snapshot(bad, NOW)
        self.assertFalse((self.root.parent / "escape").exists())

    def test_snapshot_contains_only_private_components_and_restricts_permissions(self):
        (self.data / ".env").write_text("CONTROL_SECRET_NOT_FOR_BACKUP")
        (self.data / "diagnostics.jsonl").write_text("CONTROL_DIAGNOSTIC_NOT_FOR_BACKUP")
        self.capture()
        with zipfile.ZipFile(self.snapshot) as archive:
            self.assertEqual(set(archive.namelist()), {"manifest.json", *worker.SNAPSHOT_FILES})
            self.assertNotIn(b"CONTROL_SECRET", b"".join(archive.read(n) for n in archive.namelist()))
        self.assertEqual(stat.S_IMODE(self.snapshot.stat().st_mode), 0o600)
        target = self.root / "restored"
        worker.restore_snapshot(self.snapshot, target, NOW)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o700)
        for name in worker.SNAPSHOT_FILES:
            self.assertEqual(stat.S_IMODE((target / name).stat().st_mode), 0o600)

    def test_missing_optional_files_represent_empty_initial_state(self):
        self.history.unlink()
        self.state.unlink()
        self.capture()
        target = self.root / "restored"
        worker.restore_snapshot(self.snapshot, target, NOW)
        self.assertEqual((target / self.history.name).read_bytes(), b"")
        self.assertEqual(json.loads((target / self.state.name).read_text()), {})

    def test_inconsistent_session_state_and_future_dates_refuse_capture(self):
        before = self.history.read_bytes()
        for record in [dict(pair(NOW), session_id=4), pair(NOW + timedelta(seconds=1))]:
            self.history.write_text(json.dumps(record) + "\n")
            with self.assertRaises(ValueError):
                self.capture()
            self.assertFalse(self.snapshot.exists())
        self.history.write_bytes(before)

    def test_old_snapshot_cannot_start_real_bot_after_later_forget_or_revocation(self):
        self.capture()
        storage.delete_memory(101, self.mid1, db_path=self.db)
        lifecycle.set_memory_consent(101, False, NOW + timedelta(hours=1), db_path=self.db)
        target = self.root / "restored"
        worker.restore_snapshot(self.snapshot, target, NOW + timedelta(hours=2))
        ns, app = main_namespace(target)
        with self.assertRaises(PermissionError):
            ns["main"]()
        app.run_polling.assert_not_called()
        self.assertIsNone(storage.get_memory(101, self.mid1, db_path=self.db))
        self.assertFalse(lifecycle.has_memory_consent(101, db_path=self.db))

    def test_repeat_restore_into_different_directories_does_not_duplicate_rows(self):
        self.capture()
        for name in ("first", "second"):
            worker.restore_snapshot(self.snapshot, self.root / name, NOW)
            self.assertEqual(dump(self.root / name / "memory.db"), dump(self.db))
        with self.assertRaises(FileExistsError):
            worker.restore_snapshot(self.snapshot, self.root / "first", NOW)

    def test_host_rehearsal_removes_its_copies_and_preserves_live_data(self):
        before = self.baseline()
        with worker.data_directory_lock(self.data):
            result = worker.check_recovery_quiescent(self.data, NOW, db_path=self.db)
        self.assertEqual(result["verified_files"], 3)
        self.assertTrue(result["temporary_copies_removed"])
        self.assertFalse(result["live_data_replaced"])
        self.assertEqual(list(self.data.glob(".recovery-check-*")), [])
        self.assert_live_unchanged(before)

    def test_cli_verify_and_restore_work_in_a_new_process(self):
        # Exercise actual argparse modes in new processes with controlled UTC.
        target = self.root / "cli"
        code = ("from datetime import datetime,timezone\n"
                "from memory import cleanup_worker as worker\n"
                "class Clock(datetime):\n"
                " @classmethod\n"
                " def now(cls,tz=None):\n"
                "  t=datetime(2026,11,1,12,tzinfo=timezone.utc)\n"
                "  return t.astimezone(tz) if tz else t.replace(tzinfo=None)\n"
                "worker.datetime=Clock\nworker.main()\n")
        commands = [["--db", str(self.db), "--backup", str(self.snapshot)],
                    ["--verify-snapshot", str(self.snapshot)],
                    ["--restore", str(self.snapshot), "--destination", str(target)]]
        for args in commands:
            run = subprocess.run([sys.executable, "-c", code, *args],
                                 cwd=REPO, capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(dump(target / "memory.db"), dump(self.db))


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


class RecoveryCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = BackupRecoveryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.records = []
        self.ns = dict(Update=object, ContextTypes=SimpleNamespace(DEFAULT_TYPE=object),
                       asyncio=asyncio, dialog_lock=asyncio.Lock(), datetime=Clock, timezone=timezone,
                       retention_worker=worker, DATA_DIR=self.fixture.data,
                       personal_storage=SimpleNamespace(DATABASE_PATH=self.fixture.db),
                       OWNER_ID=101, logger=logging.getLogger(__name__),
                       save_diagnostic_record=lambda record: self.records.append(record))
        for name in ("is_owner", "has_memory_test_access", "format_diagnostic", "recovery_check_command"):
            bot_fixtures.load_function(name, self.ns)
        self.update = SimpleNamespace(effective_user=SimpleNamespace(id=101),
            effective_chat=SimpleNamespace(type="private"),
            effective_message=SimpleNamespace(reply_text=AsyncMock()))

    async def test_real_command_and_diagnostic_expose_status_without_private_bodies(self):
        before = self.fixture.baseline()
        with worker.data_directory_lock(self.fixture.data):
            await self.ns["recovery_check_command"](self.update, SimpleNamespace())
        self.assertEqual(self.records[0]["status"], "выполнено")
        self.assertTrue(self.records[0]["temporary_copies_removed"])
        formatted = self.ns["format_diagnostic"](self.records[0])
        self.assertIn("3", formatted)
        self.assertNotIn("CONTROL_QUESTION", formatted)
        self.assertNotIn("101", json.dumps(self.records))
        self.fixture.assert_live_unchanged(before)

    async def test_participant_and_group_cannot_run_or_export(self):
        for user, chat in [(202, "private"), (101, "group")]:
            self.update.effective_user.id, self.update.effective_chat.type = user, chat
            with patch.object(worker, "check_recovery_quiescent") as check:
                await self.ns["recovery_check_command"](self.update, SimpleNamespace())
                check.assert_not_called()
        self.assertEqual(self.records, [])
        self.update.effective_message.reply_text.assert_not_awaited()

    async def test_failure_is_safe_and_cleans_temporary_files(self):
        self.fixture.history.write_text("CONTROL_PRIVATE_CORRUPTION")
        await self.ns["recovery_check_command"](self.update, SimpleNamespace())
        self.assertEqual(self.records[0]["status"], "ошибка")
        self.assertEqual(list(self.fixture.data.glob(".recovery-check-*")), [])
        self.assertNotIn("CONTROL_PRIVATE", json.dumps(self.records))
        self.assertNotIn("CONTROL_PRIVATE", str(self.update.effective_message.reply_text.await_args))

    async def test_stop_drains_filesystem_work_before_releasing_dialog_lock(self):
        entered, release, removed = threading.Event(), threading.Event(), threading.Event()
        def slow(*args, **kwargs):
            try:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test synchronization timeout")
                return {"verified_files": 3}
            finally:
                removed.set()
        with patch.object(worker, "check_recovery_quiescent", side_effect=slow):
            task = asyncio.create_task(self.ns["recovery_check_command"](self.update, SimpleNamespace()))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                task.cancel()
                await asyncio.sleep(0)
                self.assertTrue(self.ns["dialog_lock"].locked())
                self.assertFalse(task.done())
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(removed.is_set())
        self.assertFalse(self.ns["dialog_lock"].locked())


if __name__ == "__main__":
    unittest.main()
