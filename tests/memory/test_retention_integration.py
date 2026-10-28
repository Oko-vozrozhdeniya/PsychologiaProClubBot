import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from memory import lifecycle, storage
from memory.cleanup_worker import cleanup_once
from memory.coordinator import AccessPolicy, SourcePermission, read_context

NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
FUTURE = (NOW + timedelta(days=400)).isoformat()


class RetentionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "memory.db"
        lifecycle.initialize_lifecycle(self.db)
        self.policy = AccessPolicy(1, True, True, True, FUTURE)

    def tearDown(self):
        self.temp.cleanup()

    def read(self, now, sources):
        return read_context(
            1, "Любит рисовать", self.policy,
            [SourcePermission("private_dialog", FUTURE)],
            now.isoformat(), db_path=self.db, enforce_lifecycle=True
        )["memories"]

    def test_expired_memory_hidden_before_cleanup(self):
        lifecycle.register_message(1, "old", "Любит рисовать",
                                   NOW - timedelta(days=364), db_path=self.db)
        mid = storage.add_memory(1, "Любит рисовать", "fact",
                                 "private_dialog", self.db)
        lifecycle.link_memory_sources(1, mid, ["old"], NOW, db_path=self.db)
        self.assertEqual(len(self.read(NOW, ["old"])), 1)
        self.assertEqual(self.read(NOW + timedelta(days=2), ["old"]), [])
        self.assertIsNotNone(storage.get_memory(1, mid, self.db))

    def test_cleanup_once_removes_expired_and_is_idempotent(self):
        lifecycle.register_message(1, "old", "Любит рисовать",
                                   NOW - timedelta(days=364), db_path=self.db)
        mid = storage.add_memory(1, "Любит рисовать", "fact",
                                 "private_dialog", self.db)
        lifecycle.link_memory_sources(1, mid, ["old"], NOW, db_path=self.db)
        result = cleanup_once(self.db, NOW + timedelta(days=2))
        self.assertEqual(result, {"expired_messages": 1, "deleted_memories": 1})
        self.assertIsNone(storage.get_memory(1, mid, self.db))
        self.assertEqual(cleanup_once(self.db, NOW + timedelta(days=2)),
                         {"expired_messages": 0, "deleted_memories": 0})

    def test_legacy_unlinked_record_is_hidden(self):
        storage.add_memory(1, "Любит рисовать", "fact",
                           "private_dialog", self.db)
        self.assertEqual(self.read(NOW, []), [])

    def test_other_participant_not_visible(self):
        lifecycle.register_message(2, "a", "Любит рисовать", NOW, db_path=self.db)
        mid = storage.add_memory(2, "Любит рисовать", "fact",
                                 "private_dialog", self.db)
        lifecycle.link_memory_sources(2, mid, ["a"], NOW, db_path=self.db)
        self.assertEqual(self.read(NOW, []), [])

    def test_restart_keeps_valid_source(self):
        lifecycle.register_message(1, "a", "Любит рисовать", NOW, db_path=self.db)
        mid = storage.add_memory(1, "Любит рисовать", "fact",
                                 "private_dialog", self.db)
        lifecycle.link_memory_sources(1, mid, ["a"], NOW, db_path=self.db)
        lifecycle.initialize_lifecycle(self.db)
        self.assertEqual(len(self.read(NOW, ["a"])), 1)


if __name__ == "__main__":
    unittest.main()
