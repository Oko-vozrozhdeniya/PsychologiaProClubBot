import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from memory import dialog_bridge, lifecycle, storage
from memory.coordinator import AccessPolicy, SourcePermission

NOW = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)


def policy(user=1, consent=True, entitled=True, verified=True, days=30):
    return AccessPolicy(user, consent, entitled, verified,
                        (NOW + timedelta(days=days)).isoformat())


class DialogBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.db"

    def tearDown(self):
        self.tmp.cleanup()

    def register(self, user=1, message_id=42, access=None):
        return dialog_bridge.register_private_message(
            user, message_id, "A message", NOW,
            access if access is not None else policy(user),
            db_path=self.db)

    def test_registration_and_retention(self):
        result = self.register()
        self.assertEqual(result["source_id"], "telegram:private:1:42")
        self.assertEqual(datetime.fromisoformat(result["expires_at"]),
                         NOW + timedelta(days=365))
        self.assertEqual(self.register(), result)

    def test_denied_policies_do_not_write(self):
        for access in (policy(consent=False), policy(entitled=False),
                       policy(verified=False), policy(user=2),
                       policy(days=-1)):
            with self.assertRaises(PermissionError):
                self.register(access=access)
        self.assertFalse(self.db.exists())

    def test_invalid_id_and_timestamp(self):
        with self.assertRaises(ValueError):
            self.register(message_id=True)
        with self.assertRaises(ValueError):
            dialog_bridge.register_private_message(
                1, 42, "hello", NOW.replace(tzinfo=None), policy(),
                db_path=self.db)
        self.assertFalse(self.db.exists())

    def test_context_requires_live_owned_source(self):
        source = self.register()["source_id"]
        mid = storage.add_memory(1, "Likes books", "fact",
                                 "private_dialog", self.db)
        lifecycle.link_memory_sources(1, mid, [source], NOW, db_path=self.db)
        permission = SourcePermission(
            "private_dialog", (NOW + timedelta(days=400)).isoformat())
        context = dialog_bridge.authorized_context(
            1, "books", policy(days=400), [permission], NOW, db_path=self.db)
        self.assertEqual(len(context["memories"]), 1)
        later = dialog_bridge.authorized_context(
            1, "books", policy(days=400), [permission],
            NOW + timedelta(days=366), db_path=self.db)
        self.assertEqual(later["memories"], [])

    def test_cross_user_context_denied(self):
        self.register()
        with self.assertRaises(PermissionError):
            dialog_bridge.authorized_context(
                2, "books", policy(), [], NOW, db_path=self.db)


if __name__ == "__main__":
    unittest.main()
