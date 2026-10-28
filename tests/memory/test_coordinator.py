import tempfile
import unittest
from pathlib import Path
from memory import storage
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


if __name__ == "__main__":
    unittest.main()
