"""Run periodic retention cleanup in the same persistent environment as bot.py.

bot.py starts maintenance itself. This CLI can also run on the same host:
    python -m memory.cleanup_worker --db /app/data/memory.db

GitHub Actions cannot clean the production database on the bot host.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import stat
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from memory.lifecycle import initialize_lifecycle, purge_expired, _moment, RETENTION_DAYS
from memory.storage import DATABASE_PATH

LOGGER = logging.getLogger(__name__)
SNAPSHOT_FILES = ("memory.db", "private_history.jsonl", "private_dialog_state.json")
RECOVERY_REVIEW_FILE = "recovery_review_required.json"
MAX_SNAPSHOT_FILE_BYTES = 512 * 1024 * 1024


@contextmanager
def data_directory_lock(data_dir):
    """One bot or offline maintenance operation per canonical data directory.

    Keep the lease outside the directory: restore publishes a new directory
    by rename. All supported bot/CLI entrypoints use this same stable inode.
    Never unlink or replace the lease while another process may hold it.
    """
    path = Path(data_dir).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path.parent / ("." + path.name + ".memory-maintenance.lock"),
                 os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def ensure_recovery_ready(data_dir):
    """A point-in-time copy must not undo later deletion/consent decisions."""
    if os.path.lexists(Path(data_dir) / RECOVERY_REVIEW_FILE):
        raise PermissionError("Restored data require deletion and consent reconciliation")


def message_time(created_at, now):
    """Validate the original timestamp; processing/replay never renews it."""
    created, moment = _moment(created_at), _moment(now)
    if created > moment:
        raise ValueError("Message timestamp is in the future")
    if created + timedelta(days=RETENTION_DAYS) <= moment:
        raise PermissionError("Message retention has expired")
    return created


@contextmanager
def archive_lock(path, *, shared=False):
    """Lock a stable sidecar, not the inode replaced during compaction.

    All readers, appenders and cleaners use this lock, including other
    processes. No archive content is copied into the sidecar.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _record_times(record):
    if not isinstance(record, dict):
        raise ValueError("Invalid dialog record")
    if type(record.get("user_id")) is not int or record["user_id"] <= 0:
        raise ValueError("Invalid dialog owner")
    if type(record.get("session_id")) is not int or record["session_id"] < 0:
        raise ValueError("Invalid dialog session")
    # Legacy pairs have one stored date. Keep that anchor; never replace it
    # with migration/restart time or guess an unavailable Telegram date.
    if "question_timestamp" in record or "answer_timestamp" in record:
        question_at = _moment(record["question_timestamp"])
        answer_at = _moment(record["answer_timestamp"])
    else:
        question_at = answer_at = _moment(record.get("timestamp"))
    if question_at > answer_at:
        raise ValueError("Invalid dialog chronology")
    for key in ("question", "answer"):
        if record.get(key) is not None and not isinstance(record[key], str):
            raise ValueError("Invalid dialog message")
    return question_at, answer_at


def append_archive_record(path, record):
    """Append under the same lock used by retention; fsync before returning."""
    _record_times(record)
    with archive_lock(path):
        with Path(path).open("ab+") as archive:
            if archive.tell():
                archive.seek(-1, os.SEEK_END)
                if archive.read(1) != b"\n":
                    archive.write(b"\n")
            archive.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            archive.flush()
            os.fsync(archive.fileno())


def live_dialog_records(path, now):
    """Yield complete, unexpired pairs only; never send orphan replies to AI."""
    path, moment = Path(path), _moment(now)
    if not path.exists():
        return
    with archive_lock(path, shared=True):
        with path.open(encoding="utf-8") as archive:
            for line in archive:
                if not line.strip():
                    continue
                record = json.loads(line)
                dates = _record_times(record)
                if all(d <= moment < d + timedelta(days=RETENTION_DAYS) for d in dates):
                    if all(isinstance(record.get(k), str) for k in ("question", "answer")):
                        yield record


def prune_dialog_archive(path, now):
    """Atomically remove expired message bodies, preserving fresh bytes/dates.

    A pair may temporarily retain only its fresh answer. It is not readable
    as AI context without its question. Invalid/undated input or I/O failure
    aborts replacement, leaving the original available for diagnosis.
    """
    path, moment = Path(path), _moment(now)
    counts = {"expired_dialog_messages": 0, "deleted_dialog_records": 0}
    if not path.exists():
        return counts
    temporary = None
    with archive_lock(path):
        try:
            # A killed cleaner may leave a private staging file. Recover only
            # our own regular files while holding the same interprocess lock.
            prefix = path.name + ".retention-"
            for entry in path.parent.iterdir():
                if (entry.name.startswith(prefix) and entry.name.endswith(".tmp")
                        and stat.S_ISREG(entry.lstat().st_mode)):
                    entry.unlink()
            with path.open(encoding="utf-8", newline="") as original:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                        dir=path.parent, prefix=prefix, suffix=".tmp",
                        delete=False) as output:
                    temporary = Path(output.name)
                    os.fchmod(output.fileno(), stat.S_IMODE(os.fstat(original.fileno()).st_mode))
                    for line in original:
                        if not line.strip():
                            output.write(line)
                            continue
                        record = json.loads(line)
                        dates = _record_times(record)
                        expired = [key for key, date in zip(("question", "answer"), dates)
                                   if record.get(key) is not None
                                   and date + timedelta(days=RETENTION_DAYS) <= moment]
                        if not expired:
                            output.write(line)
                            continue
                        counts["expired_dialog_messages"] += len(expired)
                        for key in expired:
                            record[key] = None
                        if record.get("question") is None and record.get("answer") is None:
                            counts["deleted_dialog_records"] += 1
                        else:
                            output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
            if counts["expired_dialog_messages"]:
                os.replace(temporary, path)
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return counts


def cleanup_once(db_path=DATABASE_PATH, now=None, *, archive_path=None):
    """Purge SQLite and, when supplied, the full private archive.

    The two stores are independently repeatable; a failure is not reported
    as a successful full pass. Read guards remain active between passes.
    """
    moment = _moment(now if now is not None else datetime.now(timezone.utc))
    initialize_lifecycle(db_path)
    result = purge_expired(moment, db_path=db_path)
    if archive_path is not None:
        result.update(prune_dialog_archive(archive_path, moment))
    return result


def run_forever(db_path=DATABASE_PATH, interval_seconds=3600, *,
                sleep=time.sleep, clock=None, archive_path=None):
    """Log safe failure classes and retry on the next maintenance interval."""
    if type(interval_seconds) not in (int, float) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if clock is None:
        clock = lambda: datetime.now(timezone.utc)
    while True:
        try:
            result = cleanup_once(db_path, clock(), archive_path=archive_path)
            LOGGER.info("Retention cleanup: %s", result)
        except Exception as error:
            LOGGER.error("Retention cleanup failed: %s", type(error).__name__)
        sleep(interval_seconds)


def _regular_file(path):
    if not stat.S_ISREG(Path(path).lstat().st_mode):
        raise ValueError("Expected a regular private data file")


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_only_database(path):
    _regular_file(path)
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def _validate_private_dataset(directory, now):
    """Validate a closed copy without initialization, migration or AI calls."""
    directory, moment = Path(directory), _moment(now)
    for name in SNAPSHOT_FILES:
        path = directory / name
        _regular_file(path)
        if path.stat().st_size > MAX_SNAPSHOT_FILE_BYTES:
            raise ValueError("Private snapshot component is too large")
    state_file = directory / SNAPSHOT_FILES[2]
    if state_file.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Session state is too large")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or any(
        not isinstance(user, str) or not user.isascii() or not user.isdigit()
        or str(int(user)) != user or int(user) <= 0
        or type(session) is not int or session < 0
        for user, session in state.items()
    ):
        raise ValueError("Invalid private session state")
    dialogs = 0
    with (directory / SNAPSHOT_FILES[1]).open(encoding="utf-8") as archive:
        for line in archive:
            if not line.strip():
                continue
            if len(line) > 1024 * 1024:
                raise ValueError("Private dialog record is too large")
            record = json.loads(line)
            dates = _record_times(record)
            if any(date > moment for date in dates):
                raise ValueError("Future private dialog timestamp")
            if record["session_id"] > state.get(str(record["user_id"]), 0):
                raise ValueError("Archive is ahead of its session state")
            dialogs += 1
    db = _read_only_database(directory / SNAPSHOT_FILES[0])
    try:
        db.execute("PRAGMA trusted_schema=OFF")
        if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("Snapshot database integrity check failed")
        required = {"memories", "private_messages", "memory_sources", "memory_consents",
                    "memory_revisions", "memory_merges"}
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not required.issubset(tables):
            raise ValueError("Unsupported private database schema")
        if db.execute("SELECT 1 FROM sqlite_master WHERE type IN ('trigger','view')").fetchone():
            raise ValueError("Unsupported executable database schema")
        if db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Invalid snapshot source references")
        if db.execute("""SELECT 1 FROM memory_sources s LEFT JOIN memories m
                ON m.id=s.memory_id AND m.user_id=s.user_id WHERE m.id IS NULL""").fetchone():
            raise ValueError("Invalid snapshot memory ownership")
        for created, expires in db.execute("SELECT created_at,expires_at FROM private_messages"):
            created, expires = _moment(created), _moment(expires)
            if created > moment or expires != created + timedelta(days=RETENTION_DAYS):
                raise ValueError("Invalid original source lifetime")
        for enabled, updated in db.execute("SELECT enabled,updated_at FROM memory_consents"):
            if type(enabled) is not int or enabled not in (0, 1) or _moment(updated) > moment:
                raise ValueError("Invalid snapshot consent")
        counts = {"memories": db.execute("SELECT count(*) FROM memories").fetchone()[0],
                  "sources": db.execute("SELECT count(*) FROM private_messages").fetchone()[0],
                  "dialogs": dialogs, "sessions": len(state)}
    finally:
        db.close()
    return counts


def _capture_snapshot_quiescent(data_dir, destination, now, *, db_path=None):
    """Caller owns the runtime lease and has drained private handlers.

    The bot additionally holds dialog_lock; the public CLI/API acquires the
    exclusive directory lease. Copy only the three named private components,
    never tokens, environment files, group archives or diagnostic bodies.
    """
    data_dir, destination, moment = Path(data_dir).resolve(), Path(destination), _moment(now)
    ensure_recovery_ready(data_dir)
    if os.path.lexists(destination):
        raise FileExistsError("Snapshot destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    database = Path(db_path) if db_path is not None else data_dir / "memory.db"
    if database.resolve().parent != data_dir or destination.suffix.lower() != ".zip":
        raise ValueError("Snapshot requires the leased data directory and a new ZIP file")
    _regular_file(database)
    if database.stat().st_size > MAX_SNAPSHOT_FILE_BYTES:
        raise ValueError("Private snapshot component is too large")
    with tempfile.TemporaryDirectory(prefix=".memory-snapshot-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        with archive_lock(data_dir / SNAPSHOT_FILES[1], shared=True):
            source = _read_only_database(database)
            output = sqlite3.connect(stage / SNAPSHOT_FILES[0])
            try:
                source.backup(output)
                output.commit()
            finally:
                output.close()
                source.close()
            for name, empty in ((SNAPSHOT_FILES[1], b""), (SNAPSHOT_FILES[2], b"{}\n")):
                path = data_dir / name
                if os.path.lexists(path):
                    _regular_file(path)
                    if path.stat().st_size > MAX_SNAPSHOT_FILE_BYTES:
                        raise ValueError("Private snapshot component is too large")
                    shutil.copyfile(path, stage / name)
                else:
                    (stage / name).write_bytes(empty)
        _validate_private_dataset(stage, moment)
        cleanup = cleanup_once(stage / "memory.db", moment,
                               archive_path=stage / "private_history.jsonl")
        counts = _validate_private_dataset(stage, moment)
        files = {}
        for name in SNAPSHOT_FILES:
            path = stage / name
            path.chmod(0o600)
            files[name] = {"sha256": _file_digest(path), "size": path.stat().st_size}
        manifest = {"format": "psychologia-private-memory", "version": 1,
                    "created_at": moment.isoformat(), "retention_days": RETENTION_DAYS,
                    "files": files}
        bundle = stage / "snapshot.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
            for name in SNAPSHOT_FILES:
                archive.write(stage / name, name)
        bundle.chmod(0o600)
        with bundle.open("rb") as stream:
            os.fsync(stream.fileno())
        # Same-filesystem hard link publishes one complete file without ever
        # overwriting an existing destination, even in a competing process.
        os.link(bundle, destination)
        _sync_directory(destination.parent)
    return {"counts": counts, "cleanup": cleanup, "version": 1}


def create_snapshot(data_dir, destination, now=None, *, db_path=None):
    """Offline consistent snapshot. Refuse while the bot/CLI owns the lease."""
    with data_directory_lock(data_dir):
        return _capture_snapshot_quiescent(data_dir, destination,
            now if now is not None else datetime.now(timezone.utc), db_path=db_path)


def _unpack_snapshot(snapshot, stage, now):
    snapshot, stage, moment = Path(snapshot), Path(stage), _moment(now)
    _regular_file(snapshot)
    with zipfile.ZipFile(snapshot) as archive:
        names = archive.namelist()
        if len(names) != 4 or set(names) != {"manifest.json", *SNAPSHOT_FILES}:
            raise ValueError("Invalid snapshot members")
        if archive.getinfo("manifest.json").file_size > 65536:
            raise ValueError("Invalid snapshot manifest size")
        manifest = json.loads(archive.read("manifest.json"))
        if (manifest.get("format") != "psychologia-private-memory"
                or type(manifest.get("version")) is not int or manifest["version"] != 1
                or manifest.get("retention_days") != RETENTION_DAYS
                or not isinstance(manifest.get("files"), dict)
                or set(manifest["files"]) != set(SNAPSHOT_FILES)
                or _moment(manifest.get("created_at")) > moment):
            raise ValueError("Unsupported snapshot manifest")
        for name in SNAPSHOT_FILES:
            spec, info = manifest["files"][name], archive.getinfo(name)
            if (not isinstance(spec, dict) or type(spec.get("size")) is not int
                    or spec["size"] != info.file_size or not 0 <= info.file_size <= MAX_SNAPSHOT_FILE_BYTES
                    or not isinstance(spec.get("sha256"), str) or len(spec["sha256"]) != 64
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError("Invalid snapshot component metadata")
            path = stage / name
            with archive.open(name) as source, path.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            path.chmod(0o600)
            if path.stat().st_size != spec["size"] or _file_digest(path) != spec["sha256"]:
                raise ValueError("Snapshot checksum mismatch")
    _validate_private_dataset(stage, _moment(manifest["created_at"]))
    return manifest


def verify_snapshot(snapshot, now=None):
    moment = _moment(now if now is not None else datetime.now(timezone.utc))
    with tempfile.TemporaryDirectory(prefix="memory-verify-") as temporary:
        manifest = _unpack_snapshot(snapshot, temporary, moment)
    return {"version": manifest["version"], "verified_files": len(SNAPSHOT_FILES)}


def restore_snapshot(snapshot, destination, now=None):
    """Restore into a NEW isolated directory, never over live data.

    Purge by original timestamps before publication. Leave an activation
    barrier: a historical snapshot cannot establish later deletion, consent
    revocation or account state. There is deliberately no auto-activate flag.
    """
    target = Path(destination).resolve()
    moment = _moment(now if now is not None else datetime.now(timezone.utc))
    if Path(destination).is_symlink():
        raise FileExistsError("Restore destination already exists")
    with data_directory_lock(target):
        if os.path.lexists(target):
            raise FileExistsError("Restore destination already exists")
        stage = Path(tempfile.mkdtemp(prefix=".memory-restore-", dir=target.parent))
        try:
            manifest = _unpack_snapshot(snapshot, stage, moment)
            cleanup = cleanup_once(stage / "memory.db", moment,
                                   archive_path=stage / "private_history.jsonl")
            counts = _validate_private_dataset(stage, moment)
            marker = stage / RECOVERY_REVIEW_FILE
            marker.write_text(json.dumps({"version": 1, "restored_at": moment.isoformat(),
                "snapshot_created_at": manifest["created_at"],
                "status": "isolated; reconcile later deletions and consent before activation"}) + "\n")
            marker.chmod(0o600)
            for name in (*SNAPSHOT_FILES, RECOVERY_REVIEW_FILE):
                with (stage / name).open("rb") as stream:
                    os.fsync(stream.fileno())
            _sync_directory(stage)
            if os.path.lexists(target):
                raise FileExistsError("Restore destination already exists")
            os.rename(stage, target)
            _sync_directory(target.parent)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return {"counts": counts, "cleanup": cleanup, "review_required": True}


def check_recovery_quiescent(data_dir, now=None, *, db_path=None):
    """Owner-only host rehearsal under the bot's existing dialog/runtime locks.

    Temporary private copies stay on this host and are removed on exit.
    Nothing is uploaded, sent to AI, activated or written over live storage.
    """
    data_dir = Path(data_dir)
    moment = _moment(now if now is not None else datetime.now(timezone.utc))
    with tempfile.TemporaryDirectory(prefix=".recovery-check-", dir=data_dir) as temporary:
        root = Path(temporary)
        snapshot = root / "snapshot.zip"
        _capture_snapshot_quiescent(data_dir, snapshot, moment, db_path=db_path)
        verified = verify_snapshot(snapshot, moment)
        restored = restore_snapshot(snapshot, root / "restored", moment)
        try:
            ensure_recovery_ready(root / "restored")
        except PermissionError:
            pass
        else:
            raise RuntimeError("Recovery activation barrier missing")
    return {"verified_files": verified["verified_files"], **restored,
            "temporary_copies_removed": True, "live_data_replaced": False}


def main():
    parser = argparse.ArgumentParser(description="Private memory retention cleanup")
    parser.add_argument("--db", type=Path, default=DATABASE_PATH)
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--archive", type=Path,
                        help="Private JSONL archive; defaults to private_history.jsonl beside --db")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--data-dir", type=Path,
                        help="Runtime data directory, defaults to --db parent")
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument("--backup", type=Path, metavar="NEW_ZIP",
                            help="Offline snapshot; stop the bot first")
    operations.add_argument("--verify-snapshot", type=Path, metavar="ZIP")
    operations.add_argument("--restore", type=Path, metavar="ZIP",
                            help="Restore into --destination; never activates data")
    parser.add_argument("--destination", type=Path,
                        help="New isolated directory for --restore; must not exist")
    args = parser.parse_args()
    if args.destination and not args.restore:
        parser.error("--destination requires --restore")
    if args.restore and args.destination is None:
        parser.error("--restore requires --destination")
    logging.basicConfig(level=logging.INFO)
    archive = args.archive or args.db.parent / "private_history.jsonl"
    data_dir = args.data_dir or args.db.parent
    try:
        if args.backup:
            result = create_snapshot(data_dir, args.backup, db_path=args.db)
        elif args.verify_snapshot:
            result = verify_snapshot(args.verify_snapshot)
        elif args.restore:
            result = restore_snapshot(args.restore, args.destination)
        else:
            with data_directory_lock(data_dir):
                ensure_recovery_ready(data_dir)
                if args.once:
                    result = cleanup_once(args.db, archive_path=archive)
                else:
                    run_forever(args.db, args.interval, archive_path=archive)
                    return
        LOGGER.info("Private memory maintenance: %s", result)
    except Exception as error:
        # File errors may contain private paths/content: expose class only.
        LOGGER.error("Private memory maintenance failed: %s", type(error).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
