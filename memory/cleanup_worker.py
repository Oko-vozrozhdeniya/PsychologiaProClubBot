"""Run periodic retention cleanup in the same persistent environment as bot.py.

Launch as a separate long-running process on the bot host:
    python -m memory.cleanup_worker

GitHub Actions cannot clean the production database on the bot host.
"""
import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from memory.lifecycle import initialize_lifecycle, purge_expired
from memory.storage import DATABASE_PATH

LOGGER = logging.getLogger(__name__)


def cleanup_once(db_path=DATABASE_PATH, now=None):
    """Initialize and purge expired sources in one repeatable operation."""
    initialize_lifecycle(db_path)
    return purge_expired(now or datetime.now(timezone.utc), db_path=db_path)


def run_forever(db_path=DATABASE_PATH, interval_seconds=3600, *,
                sleep=time.sleep, clock=None):
    """Periodically purge; never suppress errors or silently stop on failure."""
    if type(interval_seconds) not in (int, float) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if clock is None:
        clock = lambda: datetime.now(timezone.utc)
    while True:
        result = cleanup_once(db_path, clock())
        LOGGER.info("Retention cleanup: %s", result)
        sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Private memory retention cleanup")
    parser.add_argument("--db", type=Path, default=DATABASE_PATH)
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.once:
        LOGGER.info("Retention cleanup: %s", cleanup_once(args.db))
    else:
        run_forever(args.db, args.interval)


if __name__ == "__main__":
    main()
