import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch
from memory import coordinator, lifecycle, storage
from memory.coordinator import (AccessPolicy, SourcePermission, read_context,
                                prepare_candidates, prepare_correction)

NOW = "2026-10-28T12:00:00+00:00"
FUTURE = "2026-10-29T12:00:00+00:00"
PAST = "2026-10-27T12:00:00+00:00"


def policy(user=1, consent=True, entitled=True, verified=True, expiry=FUTURE):
    return AccessPolicy(user, consent, entitled, verified, expiry)


def source(name="message:1", expiry=FUTURE, verified=True):
    return SourcePermission(name, expiry, verified)


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "memory.db"
        storage.initialize_database(self.db)
        storage.add_memory(1, "Любит читать книги", "preference", "message:1", self.db)
        storage.add_memory(1, "Любит читать книги про море", "fact", "message:old", self.db)
        storage.add_memory(2, "Любит читать книги", "fact", "message:1", self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, pol=None, sources=None, user=1):
        return read_context(user, "Какие книги любит?", pol or policy(user),
                            [source()] if sources is None else sources,
                            NOW, db_path=self.db)

    def test_authorized_only(self):
        result = self.read()
        self.assertEqual(len(result["memories"]), 1)
        self.assertEqual(result["memories"][0]["source"], "message:1")

    def test_authorized_source_not_crowded_out_by_disallowed_matches(self):
        for index in range(30):
            storage.add_memory(
                1, "Любит читать книги", "fact",
                f"message:blocked:{index}", self.db
            )
        result = self.read()
        self.assertEqual(len(result["memories"]), 1)
        self.assertEqual(result["memories"][0]["source"], "message:1")

    def test_authorized_source_survives_over_500_newer_records(self):
        for index in range(510):
            storage.add_memory(
                1, "Любит читать книги", "fact",
                f"message:blocked:{index}", self.db
            )
        result = self.read()
        self.assertEqual(len(result["memories"]), 1)
        self.assertEqual(result["memories"][0]["source"], "message:1")

    def test_cross_user_policy_rejected(self):
        with self.assertRaises(PermissionError):
            self.read(policy(2))

    def test_no_consent(self):
        with self.assertRaises(PermissionError):
            self.read(policy(consent=False))

    def test_no_entitlement(self):
        with self.assertRaises(PermissionError):
            self.read(policy(entitled=False))

    def test_unverified_policy(self):
        with self.assertRaises(PermissionError):
            self.read(policy(verified=False))

    def test_expired_policy(self):
        with self.assertRaises(PermissionError):
            self.read(policy(expiry=PAST))

    def test_expired_source(self):
        self.assertEqual(self.read(sources=[source(expiry=PAST)])["memories"], [])

    def test_unverified_source(self):
        self.assertEqual(self.read(sources=[source(verified=False)])["memories"], [])

    def test_missing_source(self):
        self.assertEqual(self.read(sources=[])["memories"], [])

    def test_duplicate_sources_rejected(self):
        with self.assertRaises(ValueError):
            self.read(sources=[source(), source()])

    def test_invalid_user(self):
        with self.assertRaises(ValueError):
            self.read(user=True)

    def test_candidate_preparation(self):
        data = [{"content": "Любит море", "kind": "preference",
                 "source": "message:1", "confidence": 0.95}]
        self.assertEqual(len(prepare_candidates(1, data, policy(), [source()], NOW)), 1)

    def test_candidate_unverified_source(self):
        data = [{"content": "Любит море", "kind": "preference",
                 "source": "message:old", "confidence": 0.95}]
        with self.assertRaises(PermissionError):
            prepare_candidates(1, data, policy(), [source()], NOW)

    def test_explicit_correction(self):
        decision = prepare_correction(1, "old", "new", "message:1",
                                      policy(), [source()], NOW,
                                      explicitly_corrected=True)
        self.assertEqual(decision.action, "update")

    def test_correction_unverified_source(self):
        with self.assertRaises(PermissionError):
            prepare_correction(1, "old", "new", "message:old",
                               policy(), [source()], NOW)


class CoordinatorPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "memory.db"
        lifecycle.initialize_lifecycle(self.db)
        lifecycle.set_memory_consent(1, True, NOW, db_path=self.db)
        self.source = "telegram:private:1:10"
        self.content = "Любит выращивать фиолетовые кактусы"

    def remember(self, **overrides):
        args = dict(user_id=1, content=self.content, kind="preference",
                    source=self.source, message_text="Мне нравятся фиолетовые кактусы",
                    policy=policy(), now=NOW, db_path=self.db)
        args.update(overrides)
        return coordinator.remember_memory(**args)

    def automatic(self, message_id=20, **overrides):
        args = dict(user_id=1, source=f"telegram:private:1:{message_id}",
                    message_text="Мне нравятся фиолетовые кактусы",
                    raw_candidates=[{"content": self.content, "kind": "preference",
                                     "confidence": 0.99}],
                    visible_memories=(), policy=policy(), now=NOW,
                    db_path=self.db)
        args.update(overrides)
        return coordinator.save_dialog_candidates(**args)

    def revise(self, mid, **overrides):
        args = dict(user_id=1, memory_id=mid, expected_content=self.content,
                    expected_source=self.source,
                    new_content="Теперь предпочитает жёлтые кактусы",
                    new_source="telegram:private:1:11", message_text="Теперь люблю жёлтые",
                    policy=policy(), now=NOW, change_kind="change",
                    explicitly_corrected=True, db_path=self.db)
        args.update(overrides)
        return coordinator.revise_memory(**args)

    def test_persist_recall_then_source_expiry(self):
        mid = self.remember()
        permissions = [SourcePermission(**item) for item in
                       lifecycle.list_live_sources(1, NOW, db_path=self.db)]
        result = read_context(1, "Какие кактусы нравится выращивать?", policy(),
                              permissions, NOW, db_path=self.db, enforce_lifecycle=True)
        self.assertEqual([item["id"] for item in result["memories"]], [mid])
        expiry = datetime.fromisoformat(NOW) + timedelta(days=365)
        self.assertEqual(permissions[0].expires_at, expiry.isoformat())
        lifecycle.purge_expired(expiry, db_path=self.db)
        self.assertEqual(lifecycle.list_live_memories(1, expiry, db_path=self.db), [])

    def test_automatic_replay_never_extends_source_or_duplicates_memory(self):
        self.assertEqual(self.automatic(), 1)
        sources = lifecycle.list_live_sources(1, NOW, db_path=self.db)
        self.assertEqual(self.automatic(now="2026-10-28T13:00:00+00:00"), 0)
        self.assertEqual(lifecycle.list_live_sources(1, NOW, db_path=self.db), sources)
        self.assertEqual(len(storage.list_memories(1, db_path=self.db)), 1)

    def test_delete_after_opt_out_removes_current_and_revision_without_replay(self):
        self.assertEqual(self.automatic(), 1)
        old = storage.list_memories(1, db_path=self.db)[0]
        self.assertTrue(self.revise(old["id"], expected_source=old["source"]))
        lifecycle.set_memory_consent(1, False, NOW, db_path=self.db)
        self.assertTrue(coordinator.forget_memory(
            1, old["id"], policy(consent=False), NOW, db_path=self.db))
        lifecycle.set_memory_consent(1, True, NOW, db_path=self.db)
        self.assertEqual(self.automatic(), 0)
        self.assertEqual(storage.list_memories(1, db_path=self.db), [])
        with storage._connection(self.db) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM memory_revisions").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM memory_sources").fetchone()[0], 0)

    def test_fresh_policy_cannot_override_revoked_persistent_consent(self):
        mid = self.remember()
        lifecycle.set_memory_consent(1, False, NOW, db_path=self.db)
        for operation in (lambda: self.remember(source="telegram:private:1:12"),
                          lambda: self.automatic(), lambda: self.revise(mid)):
            with self.subTest(operation=operation), self.assertRaises(PermissionError):
                operation()
        self.assertEqual(storage.get_memory(1, mid, self.db)["content"], self.content)
        self.assertEqual(len(lifecycle.list_live_sources(1, NOW, db_path=self.db)), 1)

    def test_denied_policies_cannot_create_change_or_delete(self):
        mid = self.remember()
        for denied in (policy(2), policy(consent=False), policy(entitled=False),
                       policy(verified=False), policy(expiry=PAST)):
            with self.subTest(policy=denied), self.assertRaises(PermissionError):
                self.remember(policy=denied, source="telegram:private:1:12")
            with self.subTest(policy=denied), self.assertRaises(PermissionError):
                self.revise(mid, policy=denied)
            if denied.consent:  # Opt-out must never prevent deleting one's own data.
                with self.subTest(policy=denied), self.assertRaises(PermissionError):
                    coordinator.forget_memory(1, mid, denied, NOW, db_path=self.db)
        self.assertEqual(storage.get_memory(1, mid, self.db)["content"], self.content)

    def test_foreign_sources_and_foreign_memory_id_cannot_be_used(self):
        mid = self.remember()
        with self.assertRaises(PermissionError):
            self.remember(source="telegram:private:2:10")
        with self.assertRaises(PermissionError):
            self.revise(mid, new_source="telegram:private:2:11")
        self.assertFalse(coordinator.forget_memory(2, mid, policy(2), NOW, db_path=self.db))
        self.assertIsNotNone(storage.get_memory(1, mid, self.db))

    def test_stale_or_expired_memory_cannot_be_revised(self):
        mid = self.remember()
        with self.assertRaises(ValueError):
            self.revise(mid, expected_content="Другая версия")
        later = (datetime.fromisoformat(NOW) + timedelta(days=365)).isoformat()
        with self.assertRaises(PermissionError):
            self.revise(mid, now=later, policy=policy(expiry="2028-01-01T00:00:00+00:00"))
        self.assertEqual(len(lifecycle.list_live_sources(1, NOW, db_path=self.db)), 1)

    def test_unconfirmed_revision_does_not_mutate(self):
        mid = self.remember()
        self.assertFalse(self.revise(mid, explicitly_corrected=False))
        self.assertEqual(storage.get_memory(1, mid, self.db)["content"], self.content)

    def test_source_must_be_independently_verified(self):
        with patch.object(lifecycle, "list_live_sources", return_value=[]):
            with self.assertRaises(PermissionError):
                self.automatic()
        self.assertEqual(storage.list_memories(1, db_path=self.db), [])

    def test_deleted_snapshot_cannot_be_recreated_as_a_replacement(self):
        mid = self.remember()
        visible = lifecycle.list_live_memories(1, NOW, db_path=self.db)
        coordinator.forget_memory(1, mid, policy(), NOW, db_path=self.db)
        raw = [{"content": "Теперь любит жёлтые кактусы", "kind": "preference",
                "confidence": 0.99, "replaces_memory_id": mid,
                "replaces_content": self.content, "revision_kind": "change"}]
        self.assertEqual(self.automatic(message_text="Теперь люблю жёлтые кактусы",
                                       raw_candidates=raw, visible_memories=visible), 0)
        self.assertEqual(storage.list_memories(1, db_path=self.db), [])

    def test_normalized_candidate_still_checks_replacement_metadata(self):
        mid = self.remember()
        raw = [{"content": "Любит  жёлтые кактусы", "kind": "preference",
                "confidence": 0.99, "replaces_memory_id": mid + 999,
                "replaces_content": self.content, "revision_kind": "change"}]
        self.assertEqual(self.automatic(message_text="Теперь люблю жёлтые кактусы",
                                       raw_candidates=raw), 0)
        self.assertEqual(len(storage.list_memories(1, db_path=self.db)), 1)


if __name__ == "__main__":
    unittest.main()
