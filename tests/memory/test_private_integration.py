
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory import lifecycle, private_integration, storage
from memory.coordinator import AccessPolicy, SourcePermission


NOW = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)


def make_policy(
    user_id=1,
    consent=True,
    entitled=True,
    verified=True,
    days=30,
):
    return AccessPolicy(
        user_id,
        consent,
        entitled,
        verified,
        (NOW + timedelta(days=days)).isoformat(),
    )


def make_permissions(days=400):
    return [
        SourcePermission(
            "private_dialog",
            (NOW + timedelta(days=days)).isoformat(),
        )
    ]


class PrivateIntegrationTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "memory.db"

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, **overrides):
        arguments = {
            "user_id": 1,
            "message_id": 42,
            "text": "I like books",
            "timestamp": NOW,
            "policy": make_policy(),
            "source_permissions": make_permissions(),
            "db_path": self.db,
        }
        arguments.update(overrides)
        return private_integration.prepare_private_memory(
            **arguments
        )

    def test_authorized_message_registration(self):
        result = self.prepare()

        self.assertEqual(
            result["source"]["source_id"],
            "telegram:private:1:42",
        )
        self.assertIn("memories", result["context"])

    def test_missing_consent_denied(self):
        with self.assertRaises(PermissionError):
            self.prepare(
                policy=make_policy(consent=False)
            )

        self.assertFalse(self.db.exists())

    def test_missing_entitlement_denied(self):
        with self.assertRaises(PermissionError):
            self.prepare(
                policy=make_policy(entitled=False)
            )

        self.assertFalse(self.db.exists())

    def test_unverified_policy_denied(self):
        with self.assertRaises(PermissionError):
            self.prepare(
                policy=make_policy(verified=False)
            )

        self.assertFalse(self.db.exists())

    def test_cross_user_policy_denied(self):
        with self.assertRaises(PermissionError):
            self.prepare(
                policy=make_policy(user_id=2)
            )

        self.assertFalse(self.db.exists())

    def test_expired_policy_denied(self):
        with self.assertRaises(PermissionError):
            self.prepare(
                policy=make_policy(days=-1)
            )

        self.assertFalse(self.db.exists())

    def test_missing_source_permissions_denied(self):
        with self.assertRaises(PermissionError):
            self.prepare(source_permissions=None)

        self.assertFalse(self.db.exists())

    def test_invalid_message_id_denied(self):
        with self.assertRaises(ValueError):
            self.prepare(message_id=True)

        self.assertFalse(self.db.exists())

    def test_naive_timestamp_denied(self):
        with self.assertRaises(ValueError):
            self.prepare(
                timestamp=NOW.replace(tzinfo=None)
            )

        self.assertFalse(self.db.exists())

    def test_authorized_memory_retrieval(self):
        source = self.prepare()["source"]["source_id"]

        memory_id = storage.add_memory(
            1,
            "I like books",
            "fact",
            "private_dialog",
            self.db,
        )

        lifecycle.link_memory_sources(
            1,
            memory_id,
            [source],
            NOW,
            db_path=self.db,
        )

        result = self.prepare(
            message_id=43,
            policy=make_policy(days=400),
        )

        self.assertEqual(
            len(result["context"]["memories"]),
            1,
        )

    def test_expired_source_not_returned(self):
        source = self.prepare()["source"]["source_id"]

        memory_id = storage.add_memory(
            1,
            "I like books",
            "fact",
            "private_dialog",
            self.db,
        )

        lifecycle.link_memory_sources(
            1,
            memory_id,
            [source],
            NOW,
            db_path=self.db,
        )

        result = self.prepare(
            message_id=44,
            timestamp=NOW + timedelta(days=366),
            policy=make_policy(days=400),
            source_permissions=make_permissions(days=400),
        )

        self.assertEqual(
            result["context"]["memories"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
