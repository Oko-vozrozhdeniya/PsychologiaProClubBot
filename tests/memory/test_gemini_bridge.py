import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from memory.gemini_bridge import private_memory_for_gemini


class GeminiBridgeTests(unittest.TestCase):
    def setUp(self):
        self.timestamp = datetime(2026, 10, 29, tzinfo=timezone.utc)
        self.args = dict(
            user_id=42, message_id=123, text="Мой запрос",
            timestamp=self.timestamp, permission_provider=lambda u, t: None,
        )

    @patch("memory.gemini_bridge.prepare_with_verified_access")
    def test_passes_verified_context_to_prompt(self, prepare):
        prepare.return_value = {
            "source": {"source_id": "telegram:private:42:123"},
            "context": {"memories": [{"content": "Предпочитает краткость"}]},
        }
        result = private_memory_for_gemini(**self.args)
        self.assertIn("Предпочитает краткость", result["prompt"])
        self.assertEqual(result["source"]["source_id"], "telegram:private:42:123")
        prepare.assert_called_once()

    @patch("memory.gemini_bridge.prepare_with_verified_access")
    def test_empty_authorized_memory(self, prepare):
        prepare.return_value = {
            "source": {"source_id": "telegram:private:42:123"},
            "context": {"memories": []},
        }
        self.assertEqual(private_memory_for_gemini(**self.args)["prompt"], "")

    @patch("memory.gemini_bridge.prepare_with_verified_access")
    def test_denied_access_does_not_return_context(self, prepare):
        prepare.side_effect = PermissionError("No verified offer")
        with self.assertRaises(PermissionError):
            private_memory_for_gemini(**self.args)

    @patch("memory.gemini_bridge.prepare_with_verified_access")
    def test_passes_provider_unchanged(self, prepare):
        prepare.return_value = {"source": {}, "context": {"memories": []}}
        private_memory_for_gemini(**self.args)
        self.assertIs(
            prepare.call_args.kwargs["permission_provider"],
            self.args["permission_provider"],
        )


if __name__ == "__main__":
    unittest.main()
