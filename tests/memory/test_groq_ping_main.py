"""Smoke-test command safety and routing, without making external calls."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class GroqMainEntryPointTests(unittest.TestCase):
    def test_groq_ping_runs_only_for_owner_in_private_chat(self):
        source = (ROOT / "bot.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        func = next(node for node in tree.body
                    if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "groq_ping")
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        body = ast.unparse(func)
        registration = ast.unparse(main)
        self.assertIn("is_owner(update)", body)
        self.assertIn("update.effective_chat.type != 'private'", body)
        self.assertIn("request_groq", body)
        self.assertIn("[]", body)
        self.assertIn("None", body)
        self.assertIn("CommandHandler('groq_ping', groq_ping", registration)
        self.assertIn("filters.ChatType.PRIVATE", registration)
        self.assertIn("filters.User(user_id=OWNER_ID)", registration)

    def test_groq_ping_does_not_mutate_memory_or_change_ai_mode(self):
        source = (ROOT / "bot.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        func = next(node for node in tree.body
                    if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "groq_ping")
        body = ast.unparse(func)
        self.assertNotIn("remember_dialog_facts", body)
        self.assertNotIn("personal_storage", body)
        self.assertNotIn("test_ai_mode", body)


if __name__ == "__main__":
    unittest.main()
