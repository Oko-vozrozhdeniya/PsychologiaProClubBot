import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from memory.coordinator import AccessPolicy, SourcePermission
from memory.telegram_access import prepare_with_verified_access


NOW = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)


def access(user_id=1, *, consent=True, entitled=True, verified=True):
    return AccessPolicy(
        user_id, consent, entitled, verified,
        (NOW + timedelta(days=30)).isoformat(),
    )


def sources(*, verified=True):
    return [
        SourcePermission(
            "private_dialog",
            (NOW + timedelta(days=30)).isoformat(),
            verified=verified,
        )
    ]


class TelegramAccessTests(unittest.TestCase):
    def invoke(self, provider, **overrides):
        kwargs = dict(
            user_id=1,
            message_id=42,
            text="Hello",
            timestamp=NOW,
            permission_provider=provider,
        )
        kwargs.update(overrides)
        return prepare_with_verified_access(**kwargs)

    @patch("memory.telegram_access.prepare_private_memory")
    def test_authorized_provider_passes_exact_permissions(self, bridge):
        policy = access()
        permissions = sources()
        provider = lambda user_id, now: (policy, permissions)
        self.invoke(provider)
        bridge.assert_called_once_with(
            user_id=1, message_id=42, text="Hello", timestamp=NOW,
            policy=policy, source_permissions=permissions, db_path=None,
        )

    @patch("memory.telegram_access.prepare_private_memory")
    def test_missing_provider_denied(self, bridge):
        with self.assertRaises(PermissionError):
            self.invoke(None)
        bridge.assert_not_called()

    @patch("memory.telegram_access.prepare_private_memory")
    def test_cross_user_policy_denied(self, bridge):
        with self.assertRaises(PermissionError):
            self.invoke(lambda *_: (access(2), sources()))
        bridge.assert_not_called()

    @patch("memory.telegram_access.prepare_private_memory")
    def test_unverified_or_unconsented_policy_denied(self, bridge):
        for policy in (
            access(consent=False),
            access(entitled=False),
            access(verified=False),
        ):
            with self.subTest(policy=policy):
                with self.assertRaises(PermissionError):
                    self.invoke(lambda *_: (policy, sources()))
        bridge.assert_not_called()

    @patch("memory.telegram_access.prepare_private_memory")
    def test_missing_or_unverified_sources_denied(self, bridge):
        for permissions in (None, [], sources(verified=False)):
            with self.subTest(permissions=permissions):
                with self.assertRaises(PermissionError):
                    self.invoke(lambda *_: (access(), permissions))
        bridge.assert_not_called()

    @patch("memory.telegram_access.prepare_private_memory")
    def test_invalid_user_or_timestamp_denied(self, bridge):
        provider = lambda *_: (access(), sources())
        with self.assertRaises(ValueError):
            self.invoke(provider, user_id=True)
        with self.assertRaises(ValueError):
            self.invoke(provider, timestamp=NOW.replace(tzinfo=None))
        bridge.assert_not_called()

    @patch("memory.telegram_access.prepare_private_memory")
    def test_provider_error_is_not_silenced(self, bridge):
        def unavailable(*_):
            raise RuntimeError("Provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "Provider unavailable"):
            self.invoke(unavailable)
        bridge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
