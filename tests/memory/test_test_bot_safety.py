"""Safety checks of the isolated Telegram test entrypoint without credentials."""
import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load_config():
    script = (ROOT / "test_bot.py").read_text(encoding="utf-8")
    tree = ast.parse(script)
    names = {"configure_test_runtime", "verify_test_bot_identity"}
    pieces = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
              and node.name in names]
    assert len(pieces) == 2
    ns = {"os": os, "Path": Path,
          "TEST_DATA_DIR": Path("/app/data/psychologia_live_test"),
          "EXPECTED_PRODUCTION_USERNAME": "psychologiaproclub_bot"}
    exec(compile(ast.Module(body=pieces, type_ignores=[]), "test_bot.py", "exec"), ns)
    return ns, tree


class IsolatedTelegramTestModeTests(unittest.TestCase):
    def setUp(self):
        self.config, self.tree = load_config()

    def base(self):
        return {
            "BOT_TEST_MODE": "1",
            "BOT_TOKEN": "TEST_TOKEN_NOT_FOR_NETWORK",
            "GEMINI_API_KEY": "TEST_GEMINI_KEY",
            "TEST_BOT_USERNAME": "PsychologiaProMemoryTest_bot",
            "TEST_BOT_ALLOWED_USER_IDS": "885005330",
        }

    def test_test_mode_refuses_launch_without_explicit_flag(self):
        opts = self.base()
        opts.pop("BOT_TEST_MODE")
        with patch.dict(os.environ, opts, clear=True):
            with self.assertRaises(RuntimeError):
                self.config["configure_test_runtime"]()

    def test_production_username_is_always_refused(self):
        opts = self.base()
        opts["TEST_BOT_USERNAME"] = "PsychologiaProClub_bot"
        with patch.dict(os.environ, opts, clear=True):
            with self.assertRaises(RuntimeError):
                self.config["configure_test_runtime"]()

    def test_data_folders_are_separate_from_production(self):
        with patch.dict(os.environ, self.base(), clear=True):
            name, ids = self.config["configure_test_runtime"]()
            self.assertEqual(name, "psychologiapromemorytest_bot")
            self.assertEqual(ids, {885005330})
            self.assertEqual(os.environ["BOT_DATA_DIR"],
                             "/app/data/psychologia_live_test")
            self.assertEqual(os.environ["MEMORY_DB_PATH"],
                             "/app/data/psychologia_live_test/memory.db")
            self.assertNotEqual(os.environ["MEMORY_DB_PATH"], "/app/data/memory.db")

    def test_explicit_data_override_cannot_point_to_production(self):
        opts = self.base()
        opts["MEMORY_DB_PATH"] = "/app/data/memory.db"
        with patch.dict(os.environ, opts, clear=True):
            with self.assertRaises(RuntimeError):
                self.config["configure_test_runtime"]()

    def test_missing_testers_and_gemini_key_fail_closed(self):
        for name in ("BOT_TOKEN", "GEMINI_API_KEY", "TEST_BOT_ALLOWED_USER_IDS"):
            with self.subTest(name=name):
                opts = self.base()
                opts.pop(name)
                with patch.dict(os.environ, opts, clear=True):
                    with self.assertRaises(RuntimeError):
                        self.config["configure_test_runtime"]()

    def test_no_shared_group_handlers_or_normal_bot_main(self):
        funcs = {node.name: node for node in self.tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        build = ast.unparse(funcs["create_test_application"])
        self.assertIn("filters.ChatType.PRIVATE", build)
        self.assertIn("filters.User", build)
        self.assertIn("bot.private_message", build)
        self.assertNotIn("filters.ChatType.GROUPS", build)
        self.assertNotIn("bot.main(", build)
        self.assertNotIn("bot.save_message", build)

    def test_bot_identity_checked_before_polling(self):
        build = ast.unparse(next(node for node in self.tree.body
                                 if isinstance(node, ast.FunctionDef)
                                 and node.name == "create_test_application"))
        self.assertIn("post_init", build)
        self.assertIn("verify_test_bot_identity", build)

    def test_bot_database_path_reads_environment_with_legacy_default(self):
        bot_code = (ROOT / "bot.py").read_text(encoding="utf-8")
        storage_code = (ROOT / "memory/storage.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("BOT_DATA_DIR", "/app/data")', bot_code)
        self.assertIn('os.environ.get("MEMORY_DB_PATH", "/app/data/memory.db")',
                      storage_code)


if __name__ == "__main__":
    unittest.main()
