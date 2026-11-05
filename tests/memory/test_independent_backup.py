"""Seven-file capture, encrypted native restic round trips, and runtime safety.

All databases, credentials and repositories are disposable synthetic fixtures.
Native tests require the pinned tools, mandatory in CI. No external services.
"""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import socket
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import urllib.error
from unittest.mock import AsyncMock, Mock, patch

from memory import backup, cleanup_worker as worker, lifecycle, storage
from test_retention_operation import pair, source
import test_bot_memory_round_trip as fixtures

NOW = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)
TOOLS = Path(os.environ.get("TEST_BACKUP_TOOLS_DIR", "/nonexistent"))
NATIVE = all((TOOLS / name).is_file() for name in ("restic", "rclone"))
if os.environ.get("REQUIRE_NATIVE_BACKUP_TESTS") == "1" and not NATIVE:
    raise RuntimeError("Required native backup tools are missing")


class Dataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "live"
        self.data.mkdir()
        self.db = self.data / "memory.db"
        self.sid, self.mid = source(self.db, 101, "original", NOW - timedelta(days=1))
        (self.data / "private_dialog_state.json").write_text('{"101":0}\n')
        (self.data / "private_history.jsonl").write_text(json.dumps(pair(NOW - timedelta(days=1))) + "\n")
        (self.data / "messages.jsonl").write_text('{"message_id":1,"text":"PRIVATE_TEST_SENTINEL"}\n')
        (self.data / "messages.txt").write_text("PRIVATE_TEST_SENTINEL\n")
        (self.data / "topics.json").write_text('{"1":"General"}\n')
        (self.data / "diagnostics.jsonl").write_text('{"status":"test"}\n')
        self.config = backup.Config(self.data, "DUMMY_LOGIN", "DUMMY_PASSWORD", "ab" * 32, TOOLS)

    def hashes(self):
        return {name: hashlib.sha256((self.data / name).read_bytes()).hexdigest()
                for name in worker.FULL_BACKUP_FILES}

    def capture(self, name="stage", now=NOW):
        stage = self.root / name
        result = worker.capture_full_dataset_quiescent(self.data, stage, now)
        return stage, result


class CaptureTests(Dataset):
    def test_captures_exact_seven_files_excludes_credentials_and_preserves_live(self):
        before = self.hashes()
        (self.data / ".env").write_text("DO_NOT_CAPTURE_SECRET")
        (self.data / "unexpected.bin").write_text("NOT_DATA")
        stage, result = self.capture()
        self.assertEqual(set(p.name for p in stage.iterdir()),
                         {*worker.FULL_BACKUP_FILES, worker.FULL_BACKUP_MANIFEST})
        self.assertEqual(result["verified_files"], 7)
        self.assertEqual(self.hashes(), before)
        worker.validate_full_dataset(stage, NOW)

    def test_new_snapshot_has_committed_wal_rows(self):
        holder = sqlite3.connect(self.db)
        self.addCleanup(holder.close)
        holder.execute("PRAGMA journal_mode=WAL")
        holder.execute("BEGIN")
        holder.execute("SELECT count(*) FROM memories").fetchone()
        _, new_id = source(self.db, 202, "wal", NOW - timedelta(hours=1))
        self.assertTrue(Path(str(self.db) + "-wal").exists())
        stage, _ = self.capture()
        self.assertIsNotNone(storage.get_memory(202, new_id, db_path=stage / "memory.db"))

    def test_expired_sources_are_removed_only_from_staging(self):
        _, old = source(self.db, 202, "expired", NOW - timedelta(days=366))
        worker.append_archive_record(self.data / "private_history.jsonl", pair(NOW - timedelta(days=366), user=202))
        before = self.hashes()
        stage, _ = self.capture()
        self.assertIsNone(storage.get_memory(202, old, db_path=stage / "memory.db"))
        self.assertIsNotNone(storage.get_memory(202, old, db_path=self.db))
        self.assertEqual(self.hashes(), before)

    def test_missing_component_refuses_incomplete_backup(self):
        (self.data / "topics.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.capture()
        self.assertFalse((self.root / "stage").exists())

    def test_symlink_component_is_refused(self):
        (self.data / "topics.json").unlink()
        external = self.root / "external.json"
        external.write_text('{"secret":"do not follow"}')
        (self.data / "topics.json").symlink_to(external)
        with self.assertRaises(ValueError):
            self.capture()

    def test_corrupt_auxiliary_file_leaves_live_unchanged_and_removes_staging(self):
        (self.data / "messages.jsonl").write_text("not json\n")
        before = self.hashes()
        with self.assertRaises(json.JSONDecodeError):
            self.capture()
        self.assertEqual(self.hashes(), before)
        self.assertFalse((self.root / "stage").exists())

    def test_existing_target_is_never_replaced(self):
        target = self.root / "stage"
        target.mkdir()
        (target / "sentinel").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.capture()
        self.assertEqual((target / "sentinel").read_text(), "keep")

    def test_disk_admission_refuses_before_writing(self):
        with patch.object(worker.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaises(OSError):
                self.capture()
        self.assertFalse((self.root / "stage").exists())

    def test_manifest_corruption_is_detected(self):
        stage, _ = self.capture()
        (stage / "messages.txt").write_text("changed")
        with self.assertRaises(ValueError):
            worker.validate_full_dataset(stage, NOW)

    def test_diagnostic_write_waits_until_capture_releases_its_lock(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        original = worker.shutil.copyfile
        def copy(a, b):
            if Path(a).name == "private_history.jsonl":
                entered.set()
                self.assertTrue(release.wait(5))
            return original(a, b)
        errors = []
        def snapshot():
            try:
                with patch.object(worker.shutil, "copyfile", side_effect=copy):
                    self.capture()
            except BaseException as error:
                errors.append(error)
        def append():
            with worker.archive_lock(self.data / "diagnostics.jsonl"):
                with (self.data / "diagnostics.jsonl").open("a") as stream:
                    stream.write('{"late":true}\n')
            finished.set()
        first = threading.Thread(target=snapshot)
        first.start()
        self.assertTrue(entered.wait(5))
        second = threading.Thread(target=append)
        second.start()
        self.assertFalse(finished.wait(.05))
        release.set()
        first.join(5)
        second.join(5)
        self.assertFalse(errors)
        self.assertTrue(finished.is_set())
        self.assertNotIn("late", (self.root / "stage" / "diagnostics.jsonl").read_text())


class ConfigurationTests(Dataset):
    def environment(self):
        return dict(YANDEX_DISK_LOGIN="DUMMY_LOGIN", YANDEX_DISK_APP_PASSWORD="DUMMY_PASSWORD",
                    BACKUP_ENCRYPTION_KEY="ab" * 32)

    def test_defaults_do_not_enable_destructive_rotation(self):
        config = backup.Config.from_environment(self.data, self.environment())
        self.assertEqual(config.interval, 3600)
        self.assertEqual(config.keep_days, 30)
        self.assertFalse(config.prune)
        self.assertNotIn("DUMMY_PASSWORD", repr(config))
        self.assertNotIn("ab" * 32, repr(config))

    def test_missing_or_invalid_encryption_key_is_refused(self):
        for value in ("", "abc", "z" * 64, "ab" * 33):
            env = self.environment()
            env["BACKUP_ENCRYPTION_KEY"] = value
            with self.assertRaises(ValueError):
                backup.Config.from_environment(self.data, env)

    def test_invalid_interval_retention_and_header_injection_are_refused(self):
        for key, value in (("BACKUP_INTERVAL_SECONDS", "0"), ("BACKUP_KEEP_DAYS", "999"),
                           ("BACKUP_MIN_FREE_BYTES", "-1"), ("YANDEX_DISK_LOGIN", "user\r\nX:bad")):
            env = self.environment()
            env[key] = value
            with self.assertRaises(ValueError):
                backup.Config.from_environment(self.data, env)

    def test_authentication_error_does_not_create_repository_or_state(self):
        stage, _ = self.capture()
        repository = Mock()
        repository.cloud_status.side_effect = PermissionError("DUMMY_PASSWORD")
        with self.assertRaises(PermissionError):
            backup.transfer(self.config, stage, repository_factory=lambda c: repository)
        repository.ensure.assert_not_called()
        self.assertFalse((self.data / ".backup-work/state.json").exists())

    def test_process_error_is_redacted_and_exit_is_checked(self):
        child = Mock()
        child.communicate.return_value = ("SECRET_USER_DATA", "DUMMY_PASSWORD")
        child.returncode = 1
        with patch.object(backup.subprocess, "Popen", return_value=child):
            with self.assertRaises(backup.BackupProcessError) as raised:
                backup._run(["unused"], {})
        self.assertNotIn("DUMMY_PASSWORD", str(raised.exception))
        self.assertNotIn("SECRET_USER_DATA", str(raised.exception))

    def test_subprocess_timeout_kills_the_entire_process_group(self):
        child = Mock(pid=123456)
        child.communicate.side_effect = [subprocess.TimeoutExpired("control", 1), ("", "")]
        with patch.object(backup.subprocess, "Popen", return_value=child), \
             patch.object(backup.os, "killpg") as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                backup._run(["unused"], {}, timeout=1)
        kill.assert_called_once_with(123456, backup.signal.SIGKILL)

    def test_no_credential_is_followed_to_a_redirect(self):
        self.assertIsNone(backup.NoRedirect().redirect_request(None, None, None, None, None, None))

    def test_repository_state_does_not_silently_accept_invalid_identifiers(self):
        path = self.root / "state.json"
        path.write_text('{"snapshot_id":"not-a-snapshot"}')
        with self.assertRaises(ValueError):
            backup._read_state(path)

    def quota_response(self, free=900 * 1024 ** 3, used=1):
        response = io.BytesIO((f'<d:multistatus xmlns:d="DAV:"><d:response><d:propstat>'
            f'<d:prop><d:quota-available-bytes>{free}</d:quota-available-bytes>'
            f'<d:quota-used-bytes>{used}</d:quota-used-bytes></d:prop>'
            '</d:propstat></d:response></d:multistatus>').encode())
        response.status = 207
        return response

    def quota_repository(self):
        repository = object.__new__(backup.Repository)
        repository.config = self.config
        return repository

    def test_cloud_status_reads_quota_and_only_404_allows_initialization(self):
        opener = Mock()
        opener.open.side_effect = [self.quota_response(),
            urllib.error.HTTPError("https://webdav.yandex.ru/fixture", 404, "missing", {}, None)]
        with patch.object(backup.urllib.request, "build_opener", return_value=opener):
            quota, exists = self.quota_repository().cloud_status()
        self.assertFalse(exists)
        self.assertEqual(quota["free_bytes"], 900 * 1024 ** 3)
        requests = [call.args[0] for call in opener.open.call_args_list]
        self.assertEqual([request.method for request in requests], ["PROPFIND", "HEAD"])
        self.assertEqual(requests[0].headers["Depth"], "0")
        self.assertEqual(requests[1].full_url,
                         "https://webdav.yandex.ru/PsychologiaPRO_backups/restic-v1/config")

    def test_cloud_authentication_and_server_errors_are_not_missing_repository(self):
        for code in (401, 403, 429, 500):
            opener = Mock()
            opener.open.side_effect = [self.quota_response(),
                urllib.error.HTTPError("https://webdav.yandex.ru/fixture", code, "denied", {}, None)]
            with patch.object(backup.urllib.request, "build_opener", return_value=opener):
                with self.assertRaises(urllib.error.HTTPError):
                    self.quota_repository().cloud_status()

    def test_cloud_low_or_unavailable_quota_refuses_before_repository_lookup(self):
        for free in (0, "unknown"):
            opener = Mock()
            opener.open.return_value = self.quota_response(free=free)
            with patch.object(backup.urllib.request, "build_opener", return_value=opener):
                with self.assertRaises((ValueError, OSError)):
                    self.quota_repository().cloud_status()
            self.assertEqual(opener.open.call_count, 1)


@unittest.skipUnless(NATIVE, "pinned restic and rclone are required")
class NativeResticTests(Dataset):
    def local_factory(self):
        target = self.root / "encrypted-repository"
        class LocalRepository(backup.Repository):
            def __init__(self, config):
                super().__init__(config, local_repository=target)
            def cloud_status(self):
                return {"free_bytes": 900 * 1024 ** 3, "used_bytes": 1}, (target / "config").is_file()
        return LocalRepository

    def transfer(self, stage):
        return backup.transfer(self.config, stage, prove=True, repository_factory=self.local_factory())

    def test_native_encrypted_round_trip_preserves_all_seven_files_and_blocks_activation(self):
        before = self.hashes()
        stage, _ = self.capture()
        result = self.transfer(stage)
        self.assertTrue(result["restore_verified"])
        self.assertEqual(result["verified_files"], 7)
        self.assertEqual(self.hashes(), before)
        state = backup._read_state(self.data / ".backup-work/state.json")
        restored = self.root / "restore-again"
        repository = self.local_factory()(self.config)
        repository.restore_isolated(state["snapshot_id"], restored, NOW + timedelta(hours=1))
        for name in worker.FULL_BACKUP_FILES:
            self.assertEqual((stage / name).read_bytes(), (restored / name).read_bytes())
        with self.assertRaises(PermissionError):
            worker.ensure_recovery_ready(restored)
        needle = b"PRIVATE_TEST_SENTINEL"
        for path in (self.root / "encrypted-repository").rglob("*"):
            if path.is_file():
                self.assertNotIn(needle, path.read_bytes())
                self.assertNotIn(self.config.key.encode(), path.read_bytes())

    def test_native_second_backup_reuses_unchanged_data(self):
        (self.data / "messages.txt").write_text(os.urandom(256 * 1024).hex())
        stage, _ = self.capture()
        self.transfer(stage)
        repository = self.local_factory()(self.config)
        initial = json.loads(repository.command("stats", "--json", "--mode", "raw-data"))["total_size"]
        with (self.data / "diagnostics.jsonl").open("a") as stream:
            stream.write('{"new":"record"}\n')
        stage, _ = self.capture("second")
        self.transfer(stage)
        final = json.loads(repository.command("stats", "--json", "--mode", "raw-data"))["total_size"]
        self.assertLess(final, initial * 1.2)
        snapshots = json.loads(repository.command("snapshots", "--json"))
        self.assertEqual(len(snapshots), 2)

    def test_wrong_key_refuses_existing_repository_without_reinitialization(self):
        stage, _ = self.capture()
        self.transfer(stage)
        config = (self.root / "encrypted-repository/config").read_bytes()
        repository = self.local_factory()(replace(self.config, key="cd" * 32))
        with self.assertRaises(backup.BackupProcessError):
            repository.ensure(True)
        self.assertEqual((self.root / "encrypted-repository/config").read_bytes(), config)

    def test_restored_expired_sources_are_purged_without_modifying_live(self):
        before = self.hashes()
        stage, _ = self.capture()
        self.transfer(stage)
        state = backup._read_state(self.data / ".backup-work/state.json")
        target = self.root / "expired"
        result = self.local_factory()(self.config).restore_isolated(
            state["snapshot_id"], target, NOW + timedelta(days=365))
        self.assertGreater(result["cleanup"]["expired_messages"], 0)
        self.assertIsNone(storage.get_memory(101, self.mid, db_path=target / "memory.db"))
        self.assertEqual(self.hashes(), before)
        with self.assertRaises(PermissionError):
            worker.ensure_recovery_ready(target)

    def test_restore_refuses_live_or_existing_destination(self):
        stage, _ = self.capture()
        self.transfer(stage)
        state = backup._read_state(self.data / ".backup-work/state.json")
        repository = self.local_factory()(self.config)
        with self.assertRaises(PermissionError):
            repository.restore_isolated(state["snapshot_id"], self.data)
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            repository.restore_isolated(state["snapshot_id"], existing)

    def test_corrupted_encrypted_pack_is_detected(self):
        stage, _ = self.capture()
        self.transfer(stage)
        pack = next(p for p in (self.root / "encrypted-repository/data").rglob("*") if p.is_file())
        body = bytearray(pack.read_bytes())
        body[len(body) // 2] ^= 1
        pack.chmod(0o600)
        pack.write_bytes(body)
        with self.assertRaises(backup.BackupProcessError):
            self.local_factory()(self.config).command("check", "--read-data")

    def test_actual_rclone_webdav_transport_round_trip(self):
        # A local authenticated WebDAV fixture; never a real account or API.
        cloud = self.root / "webdav-cloud"
        cloud.mkdir()
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        process = subprocess.Popen([str(TOOLS / "rclone"), "serve", "webdav", str(cloud),
            "--addr", f"127.0.0.1:{port}", "--user", "DUMMY_LOGIN", "--pass", "DUMMY_PASSWORD",
            "--config", os.devnull], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(process.wait, 10)
        self.addCleanup(process.terminate)
        deadline = time.monotonic() + 10
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.1):
                    break
            except OSError:
                if time.monotonic() > deadline or process.poll() is not None:
                    self.fail("WebDAV test server did not start")
                time.sleep(.02)
        url = f"http://127.0.0.1:{port}/"
        class WebDAVRepository(backup.Repository):
            def __init__(inner, config):
                super().__init__(config)
                inner.env["RCLONE_CONFIG_PSYCHOLOGIA_URL"] = url
            def cloud_status(inner):
                return {"free_bytes": 900 * 1024 ** 3, "used_bytes": 1}, \
                       (cloud / "PsychologiaPRO_backups/restic-v1/config").is_file()
        stage, _ = self.capture()
        result = backup.transfer(self.config, stage, prove=True, repository_factory=WebDAVRepository)
        self.assertTrue(result["restore_verified"])
        self.assertTrue((cloud / "PsychologiaPRO_backups/restic-v1/config").is_file())

    def test_opt_in_retention_keeps_latest_and_does_not_remove_other_tags(self):
        repository = self.local_factory()(self.config)
        repository.ensure(False)
        stage, _ = self.capture()
        old = (datetime.now(timezone.utc) - timedelta(days=45)).strftime("%Y-%m-%d %H:%M:%S")
        repository.command("backup", "--host", backup.HOST, "--tag", backup.TAG,
                           "--time", old, ".", cwd=stage)
        repository.command("backup", "--host", backup.HOST, "--tag", "unrelated",
                           "--time", old, ".", cwd=stage)
        config = replace(self.config, prune=True)
        result = backup.transfer(config, stage, prove=True, repository_factory=self.local_factory())
        self.assertEqual(result["prune_status"], "completed")
        snapshots = json.loads(repository.command("snapshots", "--json"))
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(sum(backup.TAG in snap.get("tags", []) for snap in snapshots), 1)
        self.assertEqual(sum("unrelated" in snap.get("tags", []) for snap in snapshots), 1)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_holds_dialog_lock_but_upload_releases_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "live"
            data.mkdir()
            config = backup.Config(data, "DUMMY_LOGIN", "DUMMY_PASSWORD", "ab" * 32, TOOLS)
            dialog_lock = asyncio.Lock()
            application = SimpleNamespace(bot_data={})
            records = []
            def capture(directory, stage, **kwargs):
                self.assertTrue(dialog_lock.locked())
                stage.mkdir()
                return {"source_bytes": 42}
            def transfer(config, stage, **kwargs):
                self.assertFalse(dialog_lock.locked())
                self.assertTrue(stage.is_dir())
                return {"verified_files": 7, "restore_verified": True}
            with patch.object(backup.Config, "from_environment", return_value=config), \
                 patch.object(worker, "capture_full_dataset_quiescent", side_effect=capture), \
                 patch.object(backup, "transfer", side_effect=transfer):
                result = await backup.run_backup_pass(application, data_dir=data,
                    db_path=data / "memory.db", dialog_lock=dialog_lock,
                    record=records.append, prove=True)
            self.assertEqual(result["status"], "выполнено")
            self.assertEqual(records, [result])
            self.assertEqual(list((data / ".backup-work").iterdir()), [])

    async def test_failure_is_redacted_and_temporary_plaintext_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "live"
            data.mkdir()
            config = backup.Config(data, "DUMMY_LOGIN", "DUMMY_PASSWORD", "ab" * 32, TOOLS)
            application = SimpleNamespace(bot_data={})
            def capture(directory, stage, **kwargs):
                stage.mkdir()
                (stage / "fixture").write_text("PRIVATE_TEST_SENTINEL")
                return {"source_bytes": 42}
            with patch.object(backup.Config, "from_environment", return_value=config), \
                 patch.object(worker, "capture_full_dataset_quiescent", side_effect=capture), \
                 patch.object(backup, "transfer", side_effect=OSError("DUMMY_PASSWORD")):
                result = await backup.run_backup_pass(application, data_dir=data,
                    db_path=data / "memory.db", dialog_lock=asyncio.Lock(), record=lambda r:None)
            self.assertEqual(result["status"], "ошибка")
            self.assertEqual(result["error_class"], "OSError")
            self.assertNotIn("DUMMY_PASSWORD", str(result))
            self.assertEqual(list((data / ".backup-work").iterdir()), [])

    async def test_start_is_idempotent_and_performs_a_pass_before_sleep(self):
        application = SimpleNamespace(bot_data={})
        started = asyncio.Event()
        async def run(*args, **kwargs):
            started.set()
        with patch.object(backup.Config, "from_environment", return_value=SimpleNamespace(interval=3600)), \
             patch.object(backup, "run_backup_pass", side_effect=run) as runner:
            await backup.start_runtime(application, data_dir=Path("unused"))
            first = application.bot_data["_backup_task"]
            await backup.start_runtime(application, data_dir=Path("unused"))
            self.assertIs(first, application.bot_data["_backup_task"])
            await asyncio.wait_for(started.wait(), timeout=2)
            self.assertEqual(runner.call_count, 1)
            await backup.stop_runtime(application)

    async def test_thread_cancellation_waits_for_the_actual_operation(self):
        entered, release, ended = threading.Event(), threading.Event(), threading.Event()
        def operation():
            entered.set()
            release.wait(5)
            ended.set()
        task = asyncio.create_task(backup._drain_thread(operation))
        self.assertTrue(await asyncio.to_thread(entered.wait, 5))
        task.cancel()
        await asyncio.sleep(.01)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(ended.is_set())

    async def test_owner_command_does_not_export_for_other_users(self):
        ns = dict(Update=object, ContextTypes=SimpleNamespace(DEFAULT_TYPE=object),
                  has_memory_test_access=lambda u:False, independent_backup=Mock())
        fixtures.load_function("backup_now_command", ns)
        update = SimpleNamespace(effective_message=SimpleNamespace(reply_text=AsyncMock()))
        await ns["backup_now_command"](update, Mock())
        update.effective_message.reply_text.assert_not_awaited()
        ns["independent_backup"].run_backup_pass.assert_not_called()

    async def test_stop_drains_the_registered_background_task(self):
        stopped = asyncio.Event()
        async def loop():
            try:
                await asyncio.sleep(3600)
            finally:
                stopped.set()
        task = asyncio.create_task(loop())
        await asyncio.sleep(0)
        application = SimpleNamespace(bot_data={"_backup_task":task})
        await backup.stop_runtime(application)
        self.assertTrue(stopped.is_set())
        self.assertNotIn("_backup_task", application.bot_data)


if __name__ == "__main__":
    unittest.main()
