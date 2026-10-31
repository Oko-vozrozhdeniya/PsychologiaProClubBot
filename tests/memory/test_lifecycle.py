import tempfile
import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from memory import lifecycle, storage
import json

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


class MemoryConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "memory.db"
        lifecycle.initialize_lifecycle(self.db)
        lifecycle.set_memory_consent(101, True, NOW, db_path=self.db)

    def seed(self, mid, content="Предпочитает выращивать жёлтые кактусы",
             message="Мне нравится выращивать жёлтые кактусы", age=0, user=101, kind="preference"):
        source = f"telegram:private:{user}:{mid}"
        lifecycle.register_message(user, source, message, NOW - timedelta(days=age), db_path=self.db)
        record = storage.add_memory(user, content, kind, source, db_path=self.db)
        lifecycle.link_memory_sources(user, record, [source], NOW, db_path=self.db)
        return storage.get_memory(user, record, db_path=self.db)

    def merge(self, *records, now=NOW):
        return lifecycle.merge_memories(101, list(records), now, db_path=self.db)

    def table(self, name):
        with storage._connection(self.db) as db:
            return [dict(r) for r in db.execute("SELECT * FROM " + name + " ORDER BY rowid")]

    def test_merge_preserves_messages_original_dates_and_audits_original_wording(self):
        first = self.seed(1, "Пользователь теперь предпочитает выращивать жёлтые кактусы вместо фиолетовых")
        second = self.seed(2, "Пользователь увлекается выращиванием желтых кактусов")
        messages = self.table("private_messages")
        self.assertEqual(self.merge(first, second), 1)
        current = storage.list_memories(101, db_path=self.db)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["id"], first["id"])
        self.assertNotIn("фиолетовых", current[0]["content"])
        self.assertEqual((current[0]["created_at"], current[0]["updated_at"]),
                         (first["created_at"], first["updated_at"]))
        self.assertEqual(self.table("private_messages"), messages)
        audit = lifecycle.list_memory_merges(101, first["id"], NOW, db_path=self.db)
        self.assertEqual({r["content"] for r in audit}, {first["content"], second["content"]})
        self.assertEqual({r["operation"] for r in audit}, {"merge", "current_state"})
        self.assertEqual({r["message_id"] for r in self.table("memory_sources")},
                         {first["source"], second["source"]})

    def test_question_derived_duplicate_is_not_current_support(self):
        first = self.seed(1, age=364)
        second = self.seed(2, message="Какие кактусы мне нравятся?")
        self.assertEqual(self.merge(first, second), 1)
        self.assertEqual([r["message_id"] for r in self.table("memory_sources")], [first["source"]])
        self.assertEqual(storage.get_memory(101, first["id"], self.db)["source"], first["source"])
        lifecycle.purge_expired(NOW + timedelta(days=2), db_path=self.db)
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])
        self.assertEqual(self.table("memory_merges"), [])
        self.assertEqual([r["message_id"] for r in self.table("private_messages")], [second["source"]])

    def test_no_grounded_source_means_no_merge(self):
        first = self.seed(1, message="Что мне нравится?")
        second = self.seed(2, message="Какие кактусы я люблю?")
        self.assertEqual(self.merge(first, second), 0)
        self.assertEqual(len(storage.list_memories(101, db_path=self.db)), 2)
        self.assertEqual(self.table("memory_merges"), [])

    def test_conflicting_values_kinds_and_other_subjects_are_kept(self):
        first = self.seed(1)
        for index, (content, kind) in enumerate([
            ("Предпочитает выращивать фиолетовые кактусы", "preference"),
            ("Предпочитает выращивать жёлтые кактусы", "hypothesis"),
            ("Мама предпочитает выращивать жёлтые кактусы", "preference"),
            ("Не предпочитает выращивать жёлтые кактусы", "preference"),
        ], 2):
            other = self.seed(index, content, kind=kind)
            self.assertEqual(self.merge(first, other), 0)
        self.assertEqual(len(storage.list_memories(101, db_path=self.db)), 5)

    def test_stale_snapshot_does_not_change_any_row(self):
        first, second = self.seed(1), self.seed(2)
        storage.update_memory(101, second["id"], "Предпочитает розы", self.db)
        before = self.table("memories")
        self.assertEqual(self.merge(first, second), 0)
        self.assertEqual(self.table("memories"), before)
        self.assertEqual(self.table("memory_merges"), [])

    def test_revoked_consent_rejects_before_writes(self):
        first, second = self.seed(1), self.seed(2)
        lifecycle.set_memory_consent(101, False, NOW, db_path=self.db)
        with self.assertRaises(PermissionError):
            self.merge(first, second)
        self.assertEqual(len(self.table("memories")), 2)
        self.assertEqual(self.table("memory_merges"), [])

    def test_cross_user_snapshot_cannot_merge_or_expose_other_users_data(self):
        first, second = self.seed(1), self.seed(2, user=202)
        before = self.table("memories")
        with self.assertRaises(PermissionError):
            self.merge(first, second)
        self.assertEqual(self.table("memories"), before)
        self.assertEqual(lifecycle.list_memory_merges(202, first["id"], NOW, db_path=self.db), [])

    def test_expired_primary_source_cannot_participate(self):
        first, second = self.seed(1), self.seed(2)
        self.assertEqual(self.merge(first, second, now=NOW + timedelta(days=366)), 0)
        self.assertEqual(len(self.table("memories")), 2)

    def test_failed_delete_rolls_back_audit_links_versions_and_active_rows(self):
        first, second = self.seed(1), self.seed(2)
        before = {name: self.table(name) for name in
                  ("memories", "memory_sources", "memory_revisions", "memory_merges", "private_messages")}
        with storage._connection(self.db) as db:
            db.execute("""CREATE TRIGGER reject_merge BEFORE DELETE ON memories
                          BEGIN SELECT RAISE(ABORT, 'rollback merge'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.merge(first, second)
        self.assertEqual({name: self.table(name) for name in before}, before)

    def test_idempotent_after_restart(self):
        first, second = self.seed(1), self.seed(2)
        self.assertEqual(self.merge(first, second), 1)
        lifecycle.initialize_lifecycle(self.db)
        before = self.table("memory_merges")
        self.assertEqual(self.merge(first, second), 0)
        self.assertEqual(self.table("memory_merges"), before)

    def test_real_reaffirmation_keeps_original_message_expiries(self):
        first, second = self.seed(1, age=364), self.seed(2, age=10)
        before = self.table("private_messages")
        self.assertEqual(self.merge(first, second), 1)
        lifecycle.purge_expired(NOW + timedelta(days=2), db_path=self.db)
        current = lifecycle.list_live_memories(101, NOW + timedelta(days=2), db_path=self.db)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["source"], second["source"])
        self.assertEqual(self.table("private_messages"), [before[1]])
        lifecycle.purge_expired(NOW + timedelta(days=356), db_path=self.db)
        self.assertEqual(self.table("memories"), [])

    def test_deleting_freshest_support_restores_other_live_primary_source(self):
        first, second = self.seed(1, age=10), self.seed(2)
        self.merge(first, second)
        lifecycle.delete_message(101, second["source"], db_path=self.db)
        self.assertEqual(storage.get_memory(101, first["id"], self.db)["source"], first["source"])
        self.assertEqual(len(lifecycle.list_live_memories(101, NOW, db_path=self.db)), 1)
        self.assertEqual(lifecycle.list_memory_merges(101, first["id"], NOW, db_path=self.db), [])
        lifecycle.delete_message(101, first["source"], db_path=self.db)
        self.assertEqual(self.table("memories"), [])

    def test_removed_memory_revision_history_is_reassigned_without_changing_content(self):
        first = self.seed(1)
        second = self.seed(2, "Предпочитает выращивать фиолетовые кактусы",
                           "Мне нравятся фиолетовые кактусы")
        lifecycle.revise_memory(101, second["id"], second["content"], second["source"],
            "Предпочитает выращивать жёлтые кактусы", "telegram:private:101:3",
            "Теперь мне нравится выращивать жёлтые кактусы", NOW, "change", db_path=self.db)
        changed = storage.get_memory(101, second["id"], self.db)
        before = self.table("memory_revisions")[0]
        self.assertEqual(self.merge(first, changed), 1)
        after = self.table("memory_revisions")[0]
        self.assertEqual(after, {**before, "memory_id": first["id"]})
        audit = lifecycle.list_memory_merges(101, first["id"], NOW, db_path=self.db)
        self.assertEqual(json.loads(audit[0]["revision_ids_json"]), [before["id"]])
        self.assertEqual(len(lifecycle.list_memory_revisions(101, first["id"], NOW, db_path=self.db)), 1)

    def test_forget_canonical_memory_removes_merge_audit_but_not_message_receipts(self):
        first, second = self.seed(1), self.seed(2)
        self.merge(first, second)
        self.assertTrue(storage.delete_memory(101, first["id"], self.db))
        self.assertEqual(self.table("memory_merges"), [])
        self.assertEqual(self.table("memory_sources"), [])
        self.assertEqual(len(self.table("private_messages")), 2)

    def test_new_confirmation_does_not_extend_past_embedded_in_old_representation(self):
        first = self.seed(1, "Теперь предпочитает выращивать жёлтые кактусы вместо фиолетовых", age=364)
        source = "telegram:private:101:2"
        lifecycle.register_message(101, source, "Мне нравятся выращивать жёлтые кактусы", NOW, db_path=self.db)
        self.assertTrue(lifecycle.confirm_memory(101, first, source, NOW, db_path=self.db))
        lifecycle.purge_expired(NOW + timedelta(days=2), db_path=self.db)
        current = storage.get_memory(101, first["id"], self.db)
        self.assertNotIn("фиолетовых", current["content"])
        self.assertEqual(current["source"], source)
        self.assertEqual(self.table("memory_merges"), [])

    def test_question_cannot_reconfirm_or_refresh_old_memory(self):
        first = self.seed(1, age=364)
        source = "telegram:private:101:2"
        lifecycle.register_message(101, source, "Какие кактусы мне нравятся?", NOW, db_path=self.db)
        self.assertFalse(lifecycle.confirm_memory(101, first, source, NOW, db_path=self.db))
        self.assertEqual(storage.get_memory(101, first["id"], self.db), first)
        self.assertEqual(len(self.table("memory_sources")), 1)

    def test_deleting_primary_does_not_promote_legacy_question_to_evidence(self):
        first = self.seed(1)
        question = "telegram:private:101:2"
        lifecycle.register_message(101, question, "Какие кактусы я люблю?", NOW, db_path=self.db)
        lifecycle.link_memory_sources(101, first["id"], [question], NOW, db_path=self.db)
        lifecycle.delete_message(101, first["source"], db_path=self.db)
        current = storage.get_memory(101, first["id"], self.db)
        self.assertNotEqual(current["source"], question)

    def test_failed_confirmation_rolls_back_normalization_and_audit(self):
        first = self.seed(1, "Теперь предпочитает выращивать жёлтые кактусы вместо фиолетовых")
        source = "telegram:private:101:2"
        lifecycle.register_message(101, source, "Мне нравится выращивать жёлтые кактусы", NOW, db_path=self.db)
        before = {name: self.table(name) for name in ("memories", "memory_merges", "memory_sources", "private_messages")}
        with storage._connection(self.db) as db:
            db.execute("""CREATE TRIGGER reject_confirmation BEFORE INSERT ON memory_sources
                          BEGIN SELECT RAISE(ABORT, 'rollback confirmation'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            lifecycle.confirm_memory(101, first, source, NOW, db_path=self.db)
        self.assertEqual({name: self.table(name) for name in before}, before)


if __name__ == "__main__":
    unittest.main()
