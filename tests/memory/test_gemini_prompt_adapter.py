import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from memory.gemini_prompt_adapter import append_private_memory, optional_private_memory


class GeminiPromptAdapterTests(unittest.TestCase):
    def test_no_memory_preserves_original_question(self):
        self.assertEqual(append_private_memory("Привет", ""), "Привет")

    def test_authorized_context_is_marked_as_data(self):
        result = append_private_memory("Как дела?", "Любит краткость")
        self.assertIn("<personal_memory>", result)
        self.assertIn("Любит краткость", result)
        self.assertTrue(result.endswith("Как дела?"))

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            append_private_memory("", "")
        with self.assertRaises(TypeError):
            append_private_memory("Привет", None)

    @patch("memory.gemini_bridge.private_memory_for_gemini")
    def test_absent_provider_never_calls_memory(self, bridge):
        self.assertEqual(optional_private_memory(1, 2, "Вопрос",
                         datetime.now(timezone.utc)), "")
        bridge.assert_not_called()

    @patch("memory.gemini_bridge.private_memory_for_gemini")
    def test_denied_provider_falls_back_to_normal_chat(self, bridge):
        bridge.side_effect = PermissionError("denied")
        result = optional_private_memory(1, 2, "Вопрос",
                  datetime.now(timezone.utc), provider=lambda u, t: None)
        self.assertEqual(result, "")

    @patch("memory.gemini_bridge.private_memory_for_gemini")
    def test_verified_prompt_passes_through(self, bridge):
        bridge.return_value = {"prompt": "Разрешённая память", "source": {}}
        result = optional_private_memory(1, 2, "Вопрос",
                  datetime.now(timezone.utc), provider=lambda u, t: None)
        self.assertEqual(result, "Разрешённая память")

    @patch("memory.gemini_bridge.private_memory_for_gemini")
    def test_unexpected_error_is_not_hidden(self, bridge):
        bridge.side_effect = RuntimeError("database failure")
        with self.assertRaises(RuntimeError):
            optional_private_memory(1, 2, "Вопрос",
                datetime.now(timezone.utc), provider=lambda u, t: None)


if __name__ == "__main__":
    unittest.main()
