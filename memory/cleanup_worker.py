"""Run periodic retention cleanup in the same persistent environment as bot.py.

bot.py starts maintenance itself. This CLI can also run on the same host:
    python -m memory.cleanup_worker --db /app/data/memory.db

GitHub Actions cannot clean the production database on the bot host.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import logging
import os
import stat
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from memory.lifecycle import initialize_lifecycle, purge_expired, _moment, RETENTION_DAYS
from memory.storage import DATABASE_PATH

LOGGER = logging.getLogger(__name__)


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


def main():
    parser = argparse.ArgumentParser(description="Private memory retention cleanup")
    parser.add_argument("--db", type=Path, default=DATABASE_PATH)
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--archive", type=Path,
                        help="Private JSONL archive; defaults to private_history.jsonl beside --db")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    archive = args.archive or args.db.parent / "private_history.jsonl"
    if args.once:
        LOGGER.info("Retention cleanup: %s", cleanup_once(args.db, archive_path=archive))
    else:
        run_forever(args.db, args.interval, archive_path=archive)


if __name__ == "__main__":
    main()
