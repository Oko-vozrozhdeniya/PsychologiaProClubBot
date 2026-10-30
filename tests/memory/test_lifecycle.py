import tempfile
import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from memory import lifecycle, storage

NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "test.db"
        lifecycle.initialize_lifecycle(self.db)

    def tearDown(self):
        self.temp.cleanup()

    def message(self, user, name, days_ago=0):
        return lifecycle.register_message(
            user, name, "Message " + name,
            NOW - timedelta(days=days_ago), db_path=self.db)

    def memory(self, user, sources):
        mid = storage.add_memory(user, "Любит читать книги", "fact",
                                 "private_dialog", self.db)
        lifecycle.link_memory_sources(user, mid, sources, NOW, db_path=self.db)
        return mid

    def test_each_message_has_own_365_day_expiry(self):
        a = self.message(1, "a", 364)
        b = self.message(1, "b", 10)
        self.assertEqual(datetime.fromisoformat(a), NOW + timedelta(days=1))
        self.assertEqual(datetime.fromisoformat(b), NOW + timedelta(days=355))
        result = lifecycle.purge_expired(NOW + timedelta(days=2), db_path=self.db)
        self.assertEqual(result["expired_messages"], 1)

    def test_dependent_memory_deleted_after_last_source_expires(self):
        self.message(1, "old", 364)
        self.message(1, "new", 10)
        mid = self.memory(1, ["old", "new"])
        lifecycle.purge_expired(NOW + timedelta(days=2), db_path=self.db)
        self.assertIsNotNone(storage.get_memory(1, mid, self.db))
        self.assertEqual(len(lifecycle.list_live_memories(
            1, NOW + timedelta(days=2), db_path=self.db)), 1)
        result = lifecycle.purge_expired(NOW + timedelta(days=356), db_path=self.db)
        self.assertEqual(result["deleted_memories"], 1)
        self.assertIsNone(storage.get_memory(1, mid, self.db))

    def test_deleting_only_source_cascades(self):
        self.message(1, "a")
        mid = self.memory(1, ["a"])
        result = lifecycle.delete_message(1, "a", db_path=self.db)
        self.assertEqual(result, {"deleted_messages": 1, "deleted_memories": 1})
        self.assertIsNone(storage.get_memory(1, mid, self.db))

    def test_deleting_one_of_two_sources_preserves_memory(self):
        self.message(1, "a")
        self.message(1, "b")
        mid = self.memory(1, ["a", "b"])
        lifecycle.delete_message(1, "a", db_path=self.db)
        self.assertIsNotNone(storage.get_memory(1, mid, self.db))

    def test_cross_participant_source_rejected(self):
        self.message(2, "a")
        mid = storage.add_memory(1, "fact", "fact", "private_dialog", self.db)
        with self.assertRaises(PermissionError):
            lifecycle.link_memory_sources(1, mid, ["a"], NOW, db_path=self.db)
        self.assertEqual(lifecycle.list_live_memories(1, NOW, db_path=self.db), [])

    def test_cross_participant_deletion_isolated(self):
        self.message(1, "same")
        self.message(2, "same")
        mid = self.memory(2, ["same"])
        lifecycle.delete_message(1, "same", db_path=self.db)
        self.assertIsNotNone(storage.get_memory(2, mid, self.db))

    def test_duplicate_registration_cannot_extend_expiry(self):
        self.message(1, "a", 10)
        with self.assertRaises(ValueError):
            self.message(1, "a", 0)
        result = lifecycle.purge_expired(NOW + timedelta(days=356), db_path=self.db)
        self.assertEqual(result["expired_messages"], 1)

    def test_expired_source_cannot_be_linked(self):
        self.message(1, "old", 366)
        mid = storage.add_memory(1, "fact", "fact", "private_dialog", self.db)
        with self.assertRaises(PermissionError):
            lifecycle.link_memory_sources(1, mid, ["old"], NOW, db_path=self.db)

    def test_legacy_unlinked_memories_not_deleted(self):
        mid = storage.add_memory(1, "legacy", "fact", "legacy", self.db)
        lifecycle.purge_expired(NOW, db_path=self.db)
        self.assertIsNotNone(storage.get_memory(1, mid, self.db))
        self.assertEqual(lifecycle.list_live_memories(1, NOW, db_path=self.db), [])

    def test_restart_preserves_message_and_links(self):
        self.message(1, "a")
        mid = self.memory(1, ["a"])
        lifecycle.initialize_lifecycle(self.db)
        self.assertEqual(lifecycle.list_live_memories(1, NOW, db_path=self.db)[0]["id"], mid)

    def test_invalid_naive_timestamp(self):
        with self.assertRaises(ValueError):
            self.message(1, "a", 0) if False else lifecycle.register_message(
                1, "a", "content", "2026-10-28T12:00:00", db_path=self.db)

    def test_atomic_save_rechecks_consent_and_source(self):
        source = "telegram:private:1:12"
        self.message(1, source)
        with self.assertRaises(PermissionError):
            lifecycle.add_memory(1, "Любит рисовать", "preference", source,
                                 NOW, db_path=self.db)
        lifecycle.set_memory_consent(1, True, NOW, db_path=self.db)
        lifecycle.delete_message(1, source, db_path=self.db)
        with self.assertRaises(PermissionError):
            lifecycle.add_memory(1, "Любит рисовать", "preference", source,
                                 NOW, db_path=self.db)
        self.assertEqual(storage.list_memories(1, db_path=self.db), [])

    def test_failed_linkage_rolls_back_the_new_memory(self):
        source = "telegram:private:1:12"
        self.message(1, source)
        lifecycle.set_memory_consent(1, True, NOW, db_path=self.db)
        with storage._connection(self.db) as db:
            db.execute("""CREATE TRIGGER fail_link BEFORE INSERT ON memory_sources
                          BEGIN SELECT RAISE(ABORT, 'test link failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            lifecycle.add_memory(1, "Любит рисовать", "preference", source,
                                 NOW, db_path=self.db)
        self.assertEqual(storage.list_memories(1, db_path=self.db), [])
        self.assertEqual(lifecycle.list_live_memories(1, NOW, db_path=self.db), [])


if __name__ == "__main__":
    unittest.main()
