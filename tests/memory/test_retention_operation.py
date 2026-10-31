"""Original-date retention across the real handler, archive, DB and runtime.

Only temporary files, controlled clocks and stubbed AI/Telegram delivery.
No live credentials or production storage are used.
"""
import asyncio
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta, timezone

from memory import cleanup_worker as worker, coordinator, lifecycle, storage
import test_bot_memory_round_trip as fixtures

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[2]


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


def pair(when, *, user=101, question="CONTROL_QUESTION", answer="CONTROL_ANSWER", answer_at=None):
    record = dict(timestamp=when.isoformat(), user_id=user, session_id=0,
                  question=question, answer=answer)
    if answer_at is not None:
        record.update(question_timestamp=when.isoformat(), answer_timestamp=answer_at.isoformat())
    return record


def source(db, user, name, when):
    lifecycle.initialize_lifecycle(db)
    lifecycle.set_memory_consent(user, True, when, db_path=db)
    sid = f"telegram:private:{user}:{name}"
    lifecycle.register_message(user, sid, "Мне нравится выращивать жёлтые кактусы", when, db_path=db)
    mid = lifecycle.add_memory(user, "Предпочитает выращивать жёлтые кактусы",
                               "preference", sid, when, db_path=db)
    return sid, mid


class FullArchiveRetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "private_history.jsonl"

    def write(self, records):
        self.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

    def test_exact_boundary_hides_and_purges_legacy_pairs_without_refreshing_dates(self):
        old = pair(NOW - timedelta(days=365))
        fresh = pair(NOW - timedelta(days=365) + timedelta(microseconds=1), user=202)
        self.write([old, fresh])
        fresh_bytes = self.path.read_bytes().splitlines(keepends=True)[1]
        self.assertEqual(list(worker.live_dialog_records(self.path, NOW)), [fresh])
        counts = worker.prune_dialog_archive(self.path, NOW)
        self.assertEqual(counts, {"expired_dialog_messages": 2, "deleted_dialog_records": 1})
        self.assertEqual(self.path.read_bytes(), fresh_bytes)
        self.assertNotIn("question_timestamp", json.loads(self.path.read_text()))

    def test_each_message_expires_separately_but_orphan_reply_is_never_ai_context(self):
        record = pair(NOW - timedelta(days=365), answer_at=NOW - timedelta(days=364))
        self.write([record])
        self.assertEqual(list(worker.live_dialog_records(self.path, NOW)), [])
        self.assertEqual(worker.prune_dialog_archive(self.path, NOW)["expired_dialog_messages"], 1)
        kept = json.loads(self.path.read_text())
        self.assertIsNone(kept["question"])
        self.assertEqual(kept["answer"], record["answer"])
        self.assertEqual(kept["answer_timestamp"], record["answer_timestamp"])
        counts = worker.prune_dialog_archive(self.path, NOW + timedelta(days=1))
        self.assertEqual(counts, {"expired_dialog_messages": 1, "deleted_dialog_records": 1})
        self.assertEqual(self.path.read_text(), "")

    def test_noop_preserves_all_bytes_and_repeat_is_empty(self):
        self.write([pair(NOW - timedelta(days=10))])
        before, inode = self.path.read_bytes(), self.path.stat().st_ino
        for _ in range(2):
            self.assertEqual(worker.prune_dialog_archive(self.path, NOW)["expired_dialog_messages"], 0)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.path.stat().st_ino, inode)

    def test_invalid_or_undated_rows_abort_replacement_and_fail_closed_on_read(self):
        bad_records = [dict(pair(NOW), timestamp="PRIVATE_BAD_DATE"),
                       {k:v for k,v in pair(NOW).items() if k != "timestamp"},
                       dict(pair(NOW), question_timestamp=NOW.isoformat()),
                       dict(pair(NOW), timestamp="2026-11-01T12:00:00")]
        for bad in bad_records:
            with self.subTest(record_keys=tuple(bad)):
                self.write([pair(NOW - timedelta(days=366)), bad])
                before = self.path.read_bytes()
                with self.assertRaises((ValueError, KeyError)):
                    worker.prune_dialog_archive(self.path, NOW)
                self.assertEqual(self.path.read_bytes(), before)
                with self.assertRaises((ValueError, KeyError)):
                    list(worker.live_dialog_records(self.path, NOW))
                self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_failed_replace_preserves_original_then_retry_cleans(self):
        self.write([pair(NOW - timedelta(days=366)), pair(NOW)])
        before = self.path.read_bytes()
        with patch.object(worker.os, "replace", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                worker.prune_dialog_archive(self.path, NOW)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        self.assertEqual(worker.prune_dialog_archive(self.path, NOW)["deleted_dialog_records"], 1)
        self.assertEqual(len(list(worker.live_dialog_records(self.path, NOW))), 1)

    def test_recovery_removes_only_abandoned_retention_staging_files(self):
        self.write([pair(NOW)])
        abandoned = self.root / (self.path.name + ".retention-crashed.tmp")
        unrelated = self.root / "unrelated.tmp"
        abandoned.write_text("OLD_TEMP_CONTROL")
        unrelated.write_text("KEEP")
        worker.prune_dialog_archive(self.path, NOW)
        self.assertFalse(abandoned.exists())
        self.assertEqual(unrelated.read_text(), "KEEP")

    def test_append_handles_last_legacy_line_without_newline(self):
        self.path.write_text(json.dumps(pair(NOW)))
        worker.append_archive_record(self.path, pair(NOW, user=202))
        self.assertEqual(len(list(worker.live_dialog_records(self.path, NOW))), 2)

    def test_other_process_append_waits_for_replace_and_is_not_lost(self):
        self.write([pair(NOW - timedelta(days=366)), pair(NOW, question="FRESH")])
        entered, release = threading.Event(), threading.Event()
        failures = []
        real_replace = os.replace

        def blocked_replace(src, dst):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test synchronization timeout")
            real_replace(src, dst)

        def prune():
            try:
                worker.prune_dialog_archive(self.path, NOW)
            except BaseException as error:
                failures.append(error)

        with patch.object(worker.os, "replace", side_effect=blocked_replace):
            thread = threading.Thread(target=prune)
            thread.start()
            self.assertTrue(entered.wait(3))
            ready = self.root / "child-ready"
            code = ("import json,sys;from pathlib import Path;"
                    "from memory.cleanup_worker import append_archive_record;"
                    "Path(sys.argv[2]).write_text('ready');"
                    "append_archive_record(sys.argv[1],json.loads(sys.argv[3]))")
            child = subprocess.Popen([sys.executable, "-c", code, str(self.path), str(ready),
                json.dumps(pair(NOW, user=202, question="APPENDED"))], cwd=REPO,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 3
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                time.sleep(0.05)
                self.assertIsNone(child.poll(), "Appender must wait on the stable archive lock")
            finally:
                release.set()
                thread.join(5)
                stdout, stderr = child.communicate(timeout=5)
        self.assertFalse(failures)
        self.assertEqual(child.returncode, 0, stderr)
        live = list(worker.live_dialog_records(self.path, NOW))
        self.assertEqual([r["question"] for r in live], ["FRESH", "APPENDED"])

    def test_cli_restarts_with_both_stores_and_original_expiry(self):
        db = self.root / "memory.db"
        source(db, 101, "old", NOW - timedelta(days=367))
        _, fresh_id = source(db, 202, "fresh", NOW - timedelta(days=2))
        self.write([pair(NOW - timedelta(days=367)), pair(NOW - timedelta(days=2), user=202)])
        command = [sys.executable, "-m", "memory.cleanup_worker", "--db", str(db), "--once"]
        for index in range(2):
            result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("'expired_messages': " + str(1 if index == 0 else 0), result.stderr)
            self.assertEqual(storage.list_memories(101, db_path=db), [])
            self.assertIsNotNone(storage.get_memory(202, fresh_id, db_path=db))
            self.assertEqual([r["user_id"] for r in worker.live_dialog_records(self.path, NOW)], [202])

    def test_standalone_worker_retries_after_io_failure_without_logging_private_body(self):
        class EndTest(Exception):
            pass
        intervals = []
        def sleep(seconds):
            intervals.append(seconds)
            if len(intervals) == 2:
                raise EndTest()
        with patch.object(worker, "cleanup_once", side_effect=[OSError("PRIVATE_ERROR_BODY"),
                {"expired_messages":0,"deleted_memories":0}]) as cleanup:
            with self.assertLogs(worker.LOGGER, level="INFO") as logs:
                with self.assertRaises(EndTest):
                    worker.run_forever(self.root / "memory.db", 10, sleep=sleep,
                        clock=lambda:NOW, archive_path=self.path)
        self.assertEqual(cleanup.call_count, 2)
        self.assertEqual(intervals, [10,10])
        self.assertIn("OSError", str(logs.output))
        self.assertNotIn("PRIVATE_ERROR_BODY", str(logs.output))


class OriginalSourceDateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ActualMemoryRoundTripTests(
            "test_actual_commands_save_revise_revoke_and_delete_through_coordinator")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.ns["datetime"] = Clock
        self.db, self.ns = self.fixture.db, self.fixture.ns

    def test_delayed_extraction_and_revision_keep_original_dates(self):
        f = self.fixture
        f.model_result = [{"content":"Предпочитает выращивать жёлтые кактусы",
                           "kind":"preference", "confidence":0.99}]
        first_date = NOW - timedelta(days=20)
        self.assertEqual(f.save(101, 1, "Мне нравится выращивать жёлтые кактусы",
                               source_created_at=first_date), 1)
        old = storage.list_memories(101, db_path=self.db)[0]
        second_date = NOW - timedelta(days=5)
        f.model_result = [{"content":"Предпочитает выращивать красные кактусы",
            "kind":"preference", "confidence":0.99,"replaces_memory_id":old["id"],
            "replaces_content":old["content"],"revision_kind":"change"}]
        self.assertEqual(f.save(101, 2, "Теперь мне нравится выращивать красные кактусы",
                               source_created_at=second_date), 1)
        with storage._connection(self.db) as db:
            dates = [dict(r) for r in db.execute("SELECT created_at,expires_at FROM private_messages ORDER BY message_id")]
        self.assertEqual([r["created_at"] for r in dates], [first_date.isoformat(), second_date.isoformat()])
        self.assertEqual(datetime.fromisoformat(dates[1]["expires_at"]), second_date + timedelta(days=365))
        worker.cleanup_once(self.db, second_date + timedelta(days=365))
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])
        self.assertEqual(lifecycle.list_memory_revisions(101, old["id"], NOW, db_path=self.db), [])

    def test_owner_technical_save_and_revision_also_use_original_telegram_dates(self):
        first = NOW - timedelta(days=2)
        self.fixture.command("remember_command", ["Контрольный факт"], created_at=first)
        memory = storage.list_memories(101, db_path=self.db)[0]
        second = NOW - timedelta(days=1)
        reply = self.fixture.command("revise_command", [str(memory["id"]), "Другой контрольный факт"],
                                     message_id=701, created_at=second)
        self.assertIn("Обновил", reply[0])
        with storage._connection(self.db) as db:
            dates = [r[0] for r in db.execute("SELECT created_at FROM private_messages ORDER BY message_id")]
        self.assertEqual(dates, [first.isoformat(), second.isoformat()])

    def test_expired_future_or_naive_source_never_reaches_extractor(self):
        for date in (NOW - timedelta(days=365), NOW + timedelta(seconds=1), NOW.replace(tzinfo=None)):
            with self.subTest(date=date):
                before = self.fixture.calls
                with self.assertRaises((ValueError, PermissionError)):
                    self.fixture.save(101, 50, "Я изучаю биологию", source_created_at=date)
                self.assertEqual(self.fixture.calls, before)
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])

    def test_purged_receipt_cannot_be_replayed_with_a_new_retention_window(self):
        date = NOW - timedelta(days=364)
        self.fixture.save(101, 51, "Я изучаю биологию", source_created_at=date)
        worker.cleanup_once(self.db, NOW + timedelta(days=2))
        policy = coordinator.AccessPolicy(101, True, True, True, (NOW + timedelta(days=3)).isoformat())
        with self.assertRaises(PermissionError):
            coordinator.save_dialog_candidates(101, "telegram:private:101:51", "Я изучаю биологию",
                [{"content":"Пользователь изучает биологию","kind":"fact","confidence":0.99}], [],
                policy, (NOW + timedelta(days=2)).isoformat(), db_path=self.db, source_created_at=date)
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])

    def test_delayed_revision_cannot_overwrite_a_more_recent_source(self):
        sid, mid = source(self.db, 101, "recent", NOW - timedelta(days=1))
        with self.assertRaises(PermissionError):
            lifecycle.revise_memory(101, mid, "Предпочитает выращивать жёлтые кактусы", sid,
                "Предпочитает выращивать красные кактусы", "telegram:private:101:older",
                "Теперь предпочитаю красные кактусы", NOW, "change", db_path=self.db,
                source_created_at=NOW - timedelta(days=2))
        self.assertEqual(storage.get_memory(101, mid, db_path=self.db)["source"], sid)


class ActualPrivateRetentionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = fixtures.ActualMemoryRoundTripTests(
            "test_actual_commands_save_revise_revoke_and_delete_through_coordinator")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = Path(self.fixture.tmp.name)
        self.db, self.ns = self.fixture.db, self.fixture.ns
        self.records, self.requests = [], []
        self.ns.update(datetime=Clock, os=os, asyncio=asyncio, Update=object,
            ContextTypes=SimpleNamespace(DEFAULT_TYPE=object), DATA_DIR=self.root,
            DIALOG_STATE_FILE=self.root / "state.json", DIALOG_HISTORY_FILE=self.root / "private_history.jsonl",
            MAX_DIALOG_PAIRS=30, MAX_DIALOG_CHARS=30000, RETENTION_INTERVAL_SECONDS=3600,
            dialog_lock=asyncio.Lock(), logger=logging.getLogger(__name__),
            send_long_message=AsyncMock(), send_error_response=AsyncMock(),
            save_diagnostic_record=lambda r:self.records.append(dict(r)), request_gemini=Mock())
        self.ns["personal_storage"].DATABASE_PATH = self.db
        def request(question, history, *args):
            self.requests.append((question, history))
            return "CONTROL_PROVIDER_REPLY", "Gemini"
        self.ns["request_personal_ai"] = request
        for name in ("has_personal_ai_access", "read_dialog_state", "get_session_id", "append_dialog_record",
                     "read_recent_dialog", "private_message", "format_diagnostic", "retention_cleanup_pass",
                     "retention_cleanup_loop", "start_retention_cleanup", "stop_retention_cleanup"):
            fixtures.load_function(name, self.ns)
        self.message = SimpleNamespace(message_id=701, text="Я изучаю биологию",
            date=NOW - timedelta(days=2), reply_text=AsyncMock())
        self.update = SimpleNamespace(effective_user=SimpleNamespace(id=101),
            effective_chat=SimpleNamespace(type="private"), effective_message=self.message, message=self.message)
        self.app = SimpleNamespace(bot_data={})
        self.context = SimpleNamespace(application=self.app)

    async def invoke(self):
        await self.ns["private_message"](self.update, self.context)

    async def test_real_handler_excludes_expired_pairs_before_provider_and_keeps_own_scope(self):
        history = self.ns["DIALOG_HISTORY_FILE"]
        records = [pair(NOW - timedelta(days=366), question="EXPIRED_CONTROL"),
                   pair(NOW - timedelta(days=1), question="FRESH_CONTROL"),
                   pair(NOW - timedelta(days=1), user=202, question="OTHER_USER_CONTROL")]
        history.write_text("".join(json.dumps(r) + "\n" for r in records))
        await self.invoke()
        self.ns["send_error_response"].assert_not_awaited()
        payload = str(self.requests[0][1])
        self.assertNotIn("EXPIRED_CONTROL", payload)
        self.assertNotIn("OTHER_USER_CONTROL", payload)
        self.assertIn("FRESH_CONTROL", payload)

    async def test_real_handler_passes_original_date_to_archive_and_sqlite(self):
        await self.invoke()
        self.ns["send_error_response"].assert_not_awaited()
        record = json.loads(self.ns["DIALOG_HISTORY_FILE"].read_text())
        self.assertEqual(record["question_timestamp"], self.message.date.isoformat())
        self.assertEqual(record["answer_timestamp"], NOW.isoformat())
        with storage._connection(self.db) as db:
            row = db.execute("SELECT created_at,expires_at FROM private_messages WHERE message_id=?",
                ("telegram:private:101:701",)).fetchone()
        self.assertEqual(row["created_at"], self.message.date.isoformat())
        self.assertEqual(datetime.fromisoformat(row["expires_at"]), self.message.date + timedelta(days=365))

    async def test_expired_inbound_message_does_not_read_history_call_ai_or_write(self):
        self.message.date = NOW - timedelta(days=365)
        self.ns["read_recent_dialog"] = Mock(side_effect=AssertionError("must not read"))
        await self.invoke()
        self.assertEqual(self.requests, [])
        self.ns["read_recent_dialog"].assert_not_called()
        self.assertFalse(self.ns["DIALOG_HISTORY_FILE"].exists())
        self.assertIn("истёк", self.message.reply_text.await_args.args[0])

    async def test_bad_or_missing_telegram_date_fails_closed_before_ai(self):
        for date in (None, NOW.replace(tzinfo=None), NOW + timedelta(days=1)):
            self.message.date = date
            await self.invoke()
        self.assertEqual(self.requests, [])
        self.assertEqual(self.ns["send_error_response"].await_count, 3)
        self.assertFalse(self.ns["DIALOG_HISTORY_FILE"].exists())

    async def test_bad_archive_is_not_sent_to_ai_or_partially_rewritten(self):
        history = self.ns["DIALOG_HISTORY_FILE"]
        history.write_text(json.dumps(dict(pair(NOW), timestamp="PRIVATE_CORRUPT_DATE")))
        before = history.read_bytes()
        with self.assertLogs(__name__, level="ERROR") as logs:
            await self.invoke()
        self.assertEqual(self.requests, [])
        self.assertEqual(history.read_bytes(), before)
        self.assertNotIn("PRIVATE_CORRUPT_DATE", str(logs.output))

    async def test_startup_cleans_both_stores_and_records_owner_diagnostic(self):
        _, mid = source(self.db, 101, "expired", NOW - timedelta(days=366))
        history = self.ns["DIALOG_HISTORY_FILE"]
        history.write_text(json.dumps(pair(NOW - timedelta(days=366))) + "\n")
        await self.ns["start_retention_cleanup"](self.app)
        try:
            task = self.app.bot_data["_retention_task"]
            await self.ns["start_retention_cleanup"](self.app)
            self.assertIs(self.app.bot_data["_retention_task"], task)
            self.assertEqual(len(self.records), 1)
            self.assertIsNone(storage.get_memory(101, mid, db_path=self.db))
            self.assertEqual(history.read_text(), "")
            self.assertEqual(self.records[0]["expired_dialog_messages"], 2)
            diagnostic = self.ns["format_diagnostic"](dict(self.records[0], question="PRIVATE_EXTRA"))
            self.assertIn("365", diagnostic)
            self.assertNotIn("PRIVATE_EXTRA", diagnostic)
        finally:
            await self.ns["stop_retention_cleanup"](self.app)
            await self.ns["stop_retention_cleanup"](self.app)
        self.assertNotIn("_retention_task", self.app.bot_data)

    def controlled_interval(self):
        queue = asyncio.Queue()
        intervals = []
        async def sleep(seconds):
            intervals.append(seconds)
            await queue.get()
        self.ns["asyncio"] = SimpleNamespace(sleep=sleep, create_task=asyncio.create_task,
            to_thread=asyncio.to_thread, shield=asyncio.shield, CancelledError=asyncio.CancelledError)
        return queue, intervals

    async def test_periodic_cleanup_recovers_after_error_without_private_error_body(self):
        queue, intervals = self.controlled_interval()
        second = asyncio.Event()
        def save(record):
            self.records.append(dict(record))
            if len(self.records) == 2:
                second.set()
        self.ns["save_diagnostic_record"] = save
        with patch.object(worker, "cleanup_once", side_effect=[OSError("PRIVATE_ERROR_BODY"),
                {"expired_messages":0,"deleted_memories":0}]) as cleanup:
            with self.assertLogs(__name__, level="ERROR") as logs:
                await self.ns["start_retention_cleanup"](self.app)
            try:
                queue.put_nowait(None)
                await asyncio.wait_for(second.wait(), 3)
                self.assertEqual(cleanup.call_count, 2)
                self.assertEqual(intervals[0], 3600)
                self.assertIn("ошибка", self.records[0]["status"])
                self.assertEqual(self.records[1]["status"], "выполнено")
                self.assertNotIn("PRIVATE_ERROR_BODY", str(logs.output) + str(self.records))
            finally:
                await self.ns["stop_retention_cleanup"](self.app)

    async def test_shutdown_waits_for_inflight_filesystem_work_before_unlocking(self):
        queue, _ = self.controlled_interval()
        entered, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test synchronization timeout")
            return {"expired_messages":0,"deleted_memories":0}
        with patch.object(worker, "cleanup_once", return_value={"expired_messages":0,"deleted_memories":0}):
            await self.ns["start_retention_cleanup"](self.app)
        with patch.object(worker, "cleanup_once", side_effect=blocked):
            queue.put_nowait(None)
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            stopping = asyncio.create_task(self.ns["stop_retention_cleanup"](self.app))
            try:
                await asyncio.sleep(0.02)
                self.assertFalse(stopping.done())
                self.assertTrue(self.ns["dialog_lock"].locked())
            finally:
                release.set()
                await asyncio.wait_for(stopping, 3)
        self.assertFalse(self.ns["dialog_lock"].locked())


if __name__ == "__main__":
    unittest.main()
