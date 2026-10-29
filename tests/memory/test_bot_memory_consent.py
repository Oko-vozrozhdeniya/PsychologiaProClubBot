"""Integration-oriented checks for real bot.py coordinator wiring and consent."""
import ast
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from memory import coordinator, lifecycle, storage
from memory.prompt_context import build_memory_context


NOW = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)


class PrivateMemoryConsentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "memory.db"
        lifecycle.initialize_lifecycle(self.db)

    def add_memory(self, user_id=101, message_id=11, content="Любит читать книги", created_at=NOW):
        source = f"telegram:private:{user_id}:{message_id}"
        lifecycle.register_message(user_id, source, content, created_at, db_path=self.db)
        record_id = storage.add_memory(user_id, content, "fact", source, db_path=self.db)
        lifecycle.link_memory_sources(user_id, record_id, [source], created_at, db_path=self.db)
        return record_id

    def call_bot_read(self, user_id, query):
        source = (Path(__file__).resolve().parents[2] / "bot.py").read_text(encoding="utf-8")
        fn = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef)
                  and x.name == "read_explicit_memory_prompt")
        verified = lambda uid, query, policy, permissions, now, **opts: coordinator.read_context(
            uid, query, policy, permissions, now, db_path=self.db, **opts)
        local_lifecycle = SimpleNamespace(
            initialize_lifecycle=lambda: lifecycle.initialize_lifecycle(self.db),
            has_memory_consent=lambda user: lifecycle.has_memory_consent(user, db_path=self.db),
            purge_expired=lambda now: lifecycle.purge_expired(now, db_path=self.db),
            list_live_memories=lambda user, now: lifecycle.list_live_memories(user, now, db_path=self.db),
            list_live_sources=lambda user, now: lifecycle.list_live_sources(user, now, db_path=self.db),
        )
        ns = {
            "personal_storage": SimpleNamespace(
                initialize_database=lambda: storage.initialize_database(self.db)),
            "personal_lifecycle": local_lifecycle,
            "OWNER_ID": 101, "TEST_USER_IDS": {202},
            "AccessPolicy": coordinator.AccessPolicy,
            "SourcePermission": coordinator.SourcePermission,
            "read_context": verified, "build_memory_context": build_memory_context,
            "datetime": datetime, "timezone": timezone, "timedelta": timedelta,
        }
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "bot.py", "exec"), ns)
        return ns["read_explicit_memory_prompt"](user_id, query)

    def test_opt_in_and_opt_out_are_persistent_and_separate_per_user(self):
        self.assertFalse(lifecycle.has_memory_consent(101, db_path=self.db))
        lifecycle.set_memory_consent(101, True, NOW, db_path=self.db)
        lifecycle.initialize_lifecycle(self.db)  # simulate restart
        self.assertTrue(lifecycle.has_memory_consent(101, db_path=self.db))
        self.assertFalse(lifecycle.has_memory_consent(202, db_path=self.db))
        lifecycle.set_memory_consent(101, False, NOW, db_path=self.db)
        self.assertFalse(lifecycle.has_memory_consent(101, db_path=self.db))

    def test_without_consent_never_retrieves_memory(self):
        self.add_memory()
        self.assertEqual(self.call_bot_read(101, "читать книги"), "")

    def test_explicit_consent_reads_only_live_source_backed_user_memory(self):
        self.add_memory()
        self.add_memory(user_id=202, message_id=11, content="Любит бегать утром")
        lifecycle.set_memory_consent(101, True, NOW, db_path=self.db)
        response = self.call_bot_read(101, "читать книги")
        self.assertIn("Любит читать книги", response)
        self.assertNotIn("Любит бегать", response)

    def test_revoke_hides_existing_memory_without_deleting(self):
        record_id = self.add_memory()
        lifecycle.set_memory_consent(101, True, NOW, db_path=self.db)
        self.assertIn("Любит читать книги", self.call_bot_read(101, "читать книги"))
        lifecycle.set_memory_consent(101, False, NOW, db_path=self.db)
        self.assertEqual(self.call_bot_read(101, "читать книги"), "")
        self.assertIsNotNone(storage.get_memory(101, record_id, db_path=self.db))

    def test_no_permissions_from_legacy_unlinked_memory(self):
        storage.add_memory(101, "Любит читать книги", "fact", "legacy", db_path=self.db)
        lifecycle.set_memory_consent(101, True, NOW, db_path=self.db)
        self.assertEqual(self.call_bot_read(101, "читать книги"), "")

    def test_source_liveness_and_ownership_verified_by_source_table(self):
        self.add_memory(user_id=101, message_id=11)
        self.add_memory(user_id=202, message_id=11, content="Любит бегать")
        sources = lifecycle.list_live_sources(101, NOW, db_path=self.db)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source"], "telegram:private:101:11")
        self.assertEqual(lifecycle.list_live_sources(
            101, NOW + timedelta(days=366), db_path=self.db), [])

    def test_bot_registered_consent_commands_and_uses_coordinator(self):
        source = (Path(__file__).resolve().parents[2] / "bot.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertIn("memory_on_command", names)
        self.assertIn("memory_off_command", names)
        self.assertTrue(any(
            isinstance(n, ast.Name) and n.id == "read_context"
            for n in ast.walk(names["read_explicit_memory_prompt"])
        ))
        self.assertTrue(any(
            isinstance(n, ast.Attribute) and n.attr == "has_memory_consent"
            for n in ast.walk(names["remember_dialog_facts"])
        ))
        main = names["main"]
        registered = {
            n.args[0].value
            for n in ast.walk(main)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "CommandHandler" and n.args
            and isinstance(n.args[0], ast.Constant)
        }
        self.assertTrue({"memory_on", "memory_off"}.issubset(registered))


if __name__ == "__main__":
    unittest.main()
