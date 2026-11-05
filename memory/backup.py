"""Encrypted, deduplicated off-host backups via pinned restic and rclone.

Only capture is under the bot's dialog lock. Network work runs afterwards.
Secrets are supplied through a minimal child environment, never command
arguments, manifests, Telegram replies or diagnostic exception strings.
Automation and pruning require separate explicit configuration switches.
"""
import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as XML

from memory import cleanup_worker as worker

LOGGER = logging.getLogger(__name__)
REPOSITORY = "rclone:psychologia:PsychologiaPRO_backups/restic-v1"
TAG = "psychologia-pro-full-v1"
HOST = "PsychologiaProClubBot"
HEX_ID = re.compile(r"[0-9a-f]{64}")


class BackupProcessError(RuntimeError):
    """Intentionally contains no child output, credentials or personal paths."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


@dataclass(frozen=True, repr=False)
class Config:
    data_dir: Path
    login: str
    password: str
    key: str
    tools_dir: Path
    interval: int = 3600
    keep_days: int = 30
    prune: bool = False
    min_free: int = 5 * 1024 ** 3

    @classmethod
    def from_environment(cls, data_dir, environment=None):
        env = os.environ if environment is None else environment
        login, password, key = (env.get(name, "").strip() for name in
                               ("YANDEX_DISK_LOGIN", "YANDEX_DISK_APP_PASSWORD", "BACKUP_ENCRYPTION_KEY"))
        if (not login or not password or not HEX_ID.fullmatch(key)
                or any(c in login + password for c in "\r\n")):
            raise ValueError("Backup credentials are missing or invalid")
        interval = int(env.get("BACKUP_INTERVAL_SECONDS", "3600"))
        keep = int(env.get("BACKUP_KEEP_DAYS", "30"))
        minimum = int(env.get("BACKUP_MIN_FREE_BYTES", str(5 * 1024 ** 3)))
        if not 60 <= interval <= 86400 or not 1 <= keep <= 90 or minimum < 0:
            raise ValueError("Invalid backup limits")
        prune = env.get("BACKUP_PRUNE_ENABLED", "0")
        if prune not in ("0", "1"):
            raise ValueError("Invalid pruning switch")
        directory = Path(data_dir).resolve()
        tools = Path(env.get("BACKUP_TOOLS_DIR", str(directory / ".backup-tools"))).resolve()
        return cls(directory, login, password, key, tools, interval, keep, prune == "1", minimum)


def _run(arguments, env, *, cwd=None, input_text=None, timeout=900):
    """Bound every process tree; raw child output is never used in errors."""
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        child = subprocess.Popen(arguments, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                 stdout=output, stderr=errors,
                                 text=True, start_new_session=True)
        try:
            child.communicate(input=input_text, timeout=timeout)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.communicate()
            raise
        if child.returncode != 0 or output.tell() > 8 * 1024 ** 2:
            raise BackupProcessError("Backup subprocess failed")
        output.seek(0)
        return output.read(8 * 1024 ** 2).decode("utf-8")


def _private_directory(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Backup work directory must not be a link")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


class Repository:
    def __init__(self, config, *, local_repository=None):
        self.config = config
        self.restic, self.rclone = (config.tools_dir / name for name in ("restic", "rclone"))
        for binary in (self.restic, self.rclone):
            worker._regular_file(binary)
            if not os.access(binary, os.X_OK):
                raise PermissionError("Backup tool is not executable")
        safe = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR",
                "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")
        self.env = {name: os.environ[name] for name in safe if name in os.environ}
        self.env.update(RCLONE_CONFIG=os.devnull, GOMAXPROCS="2", GOGC="50")
        obscured = _run([str(self.rclone), "obscure", "-"], self.env,
                        input_text=config.password, timeout=30).strip()
        self.env.update({"RESTIC_PASSWORD": config.key,
                         "RCLONE_CONFIG_PSYCHOLOGIA_TYPE": "webdav",
                         "RCLONE_CONFIG_PSYCHOLOGIA_URL": "https://webdav.yandex.ru/",
                         "RCLONE_CONFIG_PSYCHOLOGIA_VENDOR": "other",
                         "RCLONE_CONFIG_PSYCHOLOGIA_USER": config.login,
                         "RCLONE_CONFIG_PSYCHOLOGIA_PASS": obscured})
        self.repository = str(local_repository) if local_repository is not None else REPOSITORY
        cache = _private_directory(config.data_dir / ".backup-cache")
        self.base = [str(self.restic), "-r", self.repository, "--cache-dir", str(cache),
                     "--compression", "auto", "--pack-size", "8", "-o",
                     "rclone.program=" + str(self.rclone), "-o",
                     "rclone.args=serve restic --stdio --transfers=2 --checkers=2 "
                     "--timeout=30s --contimeout=15s --retries=3 --low-level-retries=3 --log-level=ERROR"]

    def command(self, *arguments, cwd=None, timeout=900):
        return _run([*self.base, *arguments], self.env, cwd=cwd, timeout=timeout)

    def cloud_status(self):
        """Read only quota properties, never list the account's personal files."""
        auth = "Basic " + base64.b64encode(
            (self.config.login + ":" + self.config.password).encode()).decode()
        headers = {"Authorization": auth, "Depth": "0", "Content-Type": "application/xml",
                   "User-Agent": "PsychologiaPRO-backup/1.0"}
        body = b'<d:propfind xmlns:d="DAV:"><d:prop><d:quota-available-bytes/>' \
               b'<d:quota-used-bytes/></d:prop></d:propfind>'
        client = urllib.request.build_opener(NoRedirect())
        request = urllib.request.Request("https://webdav.yandex.ru/", data=body,
                                         headers=headers, method="PROPFIND")
        with client.open(request, timeout=30) as response:
            if response.status != 207:
                raise OSError("Quota request failed")
            content = response.read(65537)
        if len(content) > 65536:
            raise ValueError("Quota response is too large")
        root = XML.fromstring(content)
        values = {}
        for key, name in (("free_bytes", "quota-available-bytes"), ("used_bytes", "quota-used-bytes")):
            value = root.find(".//{DAV:}" + name)
            if value is None or value.text is None or not value.text.isdecimal():
                raise ValueError("Storage quota is unavailable")
            values[key] = int(value.text)
        if values["free_bytes"] < self.config.min_free:
            raise OSError("Cloud storage reserve reached")
        path = "https://webdav.yandex.ru/PsychologiaPRO_backups/restic-v1/config"
        try:
            with client.open(urllib.request.Request(path, headers=headers, method="HEAD"), timeout=30) as response:
                if response.status != 200:
                    raise OSError("Repository lookup failed")
                exists = True
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            exists = False
        return values, exists

    def ensure(self, exists, expected_id=None):
        if not exists:
            if expected_id is not None:
                raise PermissionError("Previously initialized repository is missing")
            self.command("init", "--repository-version", "2")
        config = json.loads(self.command("cat", "config"))
        identity = config.get("id", "")
        if not HEX_ID.fullmatch(identity) or (expected_id is not None and identity != expected_id):
            raise PermissionError("Repository identity mismatch")
        return identity

    def backup(self, directory, parent=None):
        arguments = ["backup", "--json", "--tag", TAG, "--host", HOST]
        if parent is not None:
            if not HEX_ID.fullmatch(parent):
                raise ValueError("Invalid parent snapshot")
            arguments += ["--parent", parent]
        output = self.command(*arguments, ".", cwd=directory)
        summary = next((json.loads(line) for line in reversed(output.splitlines())
                        if line.startswith("{") and json.loads(line).get("message_type") == "summary"), None)
        identity = summary.get("snapshot_id", "") if summary else ""
        if not HEX_ID.fullmatch(identity):
            raise ValueError("Backup did not commit a snapshot")
        # Read back the authenticated encrypted snapshot metadata.
        snapshot = json.loads(self.command("cat", "snapshot", identity))
        if snapshot.get("hostname") != HOST or TAG not in snapshot.get("tags", []):
            raise ValueError("Unexpected committed snapshot")
        return identity, int(summary.get("data_added_packed", 0))

    def restore_isolated(self, snapshot_id, destination, now=None):
        """Return a NEW flat dataset; no flag exists to activate it."""
        if not HEX_ID.fullmatch(snapshot_id):
            raise ValueError("Invalid snapshot ID")
        target = Path(destination).resolve()
        live = self.config.data_dir
        if target == live or target.is_relative_to(live) or live.is_relative_to(target):
            raise PermissionError("Restore must be separate from live storage")
        if os.path.lexists(target):
            raise FileExistsError("Restore destination must be new")
        moment = worker._moment(now if now is not None else datetime.now(timezone.utc))
        metadata = json.loads(self.command("cat", "snapshot", snapshot_id))
        if metadata.get("hostname") != HOST or TAG not in metadata.get("tags", []):
            raise ValueError("Snapshot is outside the platform backup set")
        target.parent.mkdir(parents=True, exist_ok=True)
        with worker.data_directory_lock(target):
            with tempfile.TemporaryDirectory(prefix=".backup-restore-", dir=target.parent) as temporary:
                root = Path(temporary)
                self.command("restore", snapshot_id, "--target", str(root / "restic"))
                manifests = list((root / "restic").rglob(worker.FULL_BACKUP_MANIFEST))
                if len(manifests) != 1:
                    raise ValueError("Backup contains an ambiguous dataset")
                dataset = manifests[0].parent
                if {p.name for p in dataset.iterdir()} != {*worker.FULL_BACKUP_FILES, worker.FULL_BACKUP_MANIFEST}:
                    raise ValueError("Unexpected files in restored dataset")
                counts = worker.validate_full_dataset(dataset, moment)
                cleanup = worker.cleanup_once(dataset / "memory.db", moment,
                                               archive_path=dataset / "private_history.jsonl")
                (dataset / "private_history.jsonl.lock").unlink(missing_ok=True)
                worker.validate_full_dataset(dataset, moment, verify_manifest=False)
                marker = dataset / worker.RECOVERY_REVIEW_FILE
                marker.write_text(json.dumps({"version": 1, "snapshot_id": snapshot_id,
                    "restored_at": moment.isoformat(), "review_required": True,
                    "status": "isolated; reconcile subsequent deletions, consent and sessions"}) + "\n")
                marker.chmod(0o600)
                for path in dataset.iterdir():
                    worker._regular_file(path)
                    path.chmod(0o600)
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
                worker._sync_directory(dataset)
                # The target lease prevents supported entrypoints racing us;
                # reject any non-cooperating pre-existing destination too.
                if os.path.lexists(target):
                    raise FileExistsError("Restore destination appeared")
                os.rename(dataset, target)
                worker._sync_directory(target.parent)
        return {"verified_files": 7, "review_required": True, "counts": counts, "cleanup": cleanup}


def _read_state(path):
    if not path.exists():
        return {}
    worker._regular_file(path)
    if path.stat().st_size > 65536:
        raise ValueError("Invalid backup state")
    state = json.loads(path.read_text())
    if not isinstance(state, dict):
        raise ValueError("Invalid backup state")
    for key in ("repository_id", "snapshot_id"):
        if key in state and not HEX_ID.fullmatch(state[key]):
            raise ValueError("Invalid backup state identity")
    return state


def _write_state(path, state):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o600)
    try:
        os.replace(temporary, path)
        worker._sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def transfer(config, directory, *, prove=False, repository_factory=Repository):
    """Encrypt and deduplicate with restic; no raw data are uploaded by us."""
    work = _private_directory(config.data_dir / ".backup-work")
    state_file = work / "state.json"
    state = _read_state(state_file)
    repository = repository_factory(config)
    quota, exists = repository.cloud_status()
    identity = repository.ensure(exists, state.get("repository_id"))
    snapshot, added = repository.backup(directory, state.get("snapshot_id"))
    # Never publish an initialized repository without retaining its identity.
    state.update(repository_id=identity, snapshot_id=snapshot,
                 completed_at=datetime.now(timezone.utc).isoformat())
    verified = False
    if prove or not state.get("restore_verified_at"):
        with tempfile.TemporaryDirectory(prefix=".backup-proof-", dir=config.data_dir.parent) as temporary:
            result = repository.restore_isolated(snapshot, Path(temporary) / "restored")
            worker.ensure_recovery_ready(config.data_dir)
            try:
                worker.ensure_recovery_ready(Path(temporary) / "restored")
            except PermissionError:
                pass
            else:
                raise RuntimeError("Restore activation barrier missing")
            if result["verified_files"] != 7:
                raise ValueError("Incomplete restore proof")
        state["restore_verified_at"] = state["completed_at"]
        verified = True
    _write_state(state_file, state)
    record = {"snapshot_id": snapshot[:12], "uploaded_bytes": added,
              "verified_files": 7, "restore_verified": verified, **quota,
              "prune_status": "disabled"}
    # Destructive retention is opt-in and scoped to this tagged repository.
    # It is not run after an unsuccessful backup, and a restore proof must exist.
    if config.prune and state.get("restore_verified_at"):
        last = state.get("pruned_at")
        if last is None or worker._moment(last) + timedelta(days=1) <= datetime.now(timezone.utc):
            try:
                repository.command("check")
                repository.command("forget", "--tag", TAG, "--host", HOST,
                                   "--group-by", "host,tags", "--keep-within",
                                   str(config.keep_days) + "d", "--prune")
                state["pruned_at"] = datetime.now(timezone.utc).isoformat()
                _write_state(state_file, state)
                record["prune_status"] = "completed"
            except Exception as error:
                record.update(prune_status="error", maintenance_error=type(error).__name__)
        else:
            record["prune_status"] = "not_due"
    return record


async def _drain_thread(function, *args, **kwargs):
    operation = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        try:
            await operation
        except Exception as error:
            LOGGER.error("Backup stopped after filesystem operation: %s", type(error).__name__)
        raise


async def run_backup_pass(application, *, data_dir, db_path, dialog_lock, record, prove=False):
    lock = application.bot_data.setdefault("_backup_lock", asyncio.Lock())
    async with lock:
        started = time.monotonic()
        status = {"record_type": "backup", "timestamp": datetime.now(timezone.utc).isoformat()}
        try:
            config = Config.from_environment(data_dir)
            base = _private_directory(config.data_dir / ".backup-work")
            with tempfile.TemporaryDirectory(prefix="run-", dir=base) as temporary:
                stage = Path(temporary) / "dataset"
                async with dialog_lock:
                    captured = await _drain_thread(worker.capture_full_dataset_quiescent,
                        data_dir, stage, db_path=db_path)
                uploaded = await _drain_thread(transfer, config, stage, prove=prove)
            status.update(status="выполнено", source_bytes=captured["source_bytes"], **uploaded)
        except Exception as error:
            status.update(status="ошибка", error_class=type(error).__name__)
            LOGGER.error("Independent backup failed: %s", type(error).__name__)
        status["duration_seconds"] = round(time.monotonic() - started, 3)
        application.bot_data["backup_status"] = status
        try:
            record(status)
        except Exception as error:
            LOGGER.error("Cannot record backup status: %s", type(error).__name__)
        return status


async def start_runtime(application, **kwargs):
    previous = application.bot_data.get("_backup_task")
    if previous is not None and not previous.done():
        return
    config = Config.from_environment(kwargs["data_dir"])

    async def loop():
        deadline = asyncio.get_running_loop().time()
        while True:
            await run_backup_pass(application, **kwargs)
            deadline += config.interval
            current = asyncio.get_running_loop().time()
            while deadline <= current:
                deadline += config.interval
            await asyncio.sleep(deadline - current)

    application.bot_data["_backup_task"] = asyncio.create_task(loop(), name="independent-backup")


async def stop_runtime(application):
    task = application.bot_data.pop("_backup_task", None)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Restore an encrypted backup into a new isolated directory")
    parser.add_argument("snapshot_id")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = Config.from_environment(Path(os.environ.get("BOT_DATA_DIR", "/app/data")))
        Repository(config).restore_isolated(args.snapshot_id, args.destination)
        print("Restored 7 files into an isolated copy; activation requires reconciliation.")
    except Exception as error:
        print("Backup restore failed:", type(error).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
