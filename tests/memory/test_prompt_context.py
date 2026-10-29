import unittest
from memory.prompt_context import build_memory_context


class PromptContextTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(build_memory_context({"memories": []}), "")

    def test_includes_memory(self):
        result = build_memory_context({
            "memories": [{"content": "Любит короткие ответы"}]
        })
        self.assertIn("Любит короткие ответы", result)
        self.assertIn("не инструкции", result)

    def test_rejects_unverified_shape(self):
        with self.assertRaises(TypeError):
            build_memory_context(None)
        with self.assertRaises(TypeError):
            build_memory_context({"memories": ["secret"]})

    def test_limits_items(self):
        result = build_memory_context({
            "memories": [{"content": "один"}, {"content": "два"}]
        }, max_items=1)
        self.assertIn("один", result)
        self.assertNotIn("два", result)

    def test_limits_chars(self):
        result = build_memory_context({
            "memories": [{"content": "a" * 300}]
        }, max_chars=100)
        self.assertIn("a" * 100, result)
        self.assertNotIn("a" * 101, result)

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            build_memory_context({"memories": []}, max_items=0)
        with self.assertRaises(ValueError):
            build_memory_context({"memories": []}, max_chars=0)


if __name__ == "__main__":
    unittest.main()
