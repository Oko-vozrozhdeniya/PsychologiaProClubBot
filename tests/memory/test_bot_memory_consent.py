"""Integration-oriented checks for real bot.py coordinator wiring and consent."""
import ast
import asyncio
import logging
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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


class PrivateDialogConsentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "memory.db"
        lifecycle.initialize_lifecycle(self.db)
        code = (Path(__file__).resolve().parents[2] / "bot.py").read_text(encoding="utf-8")
        names = {node.name: node for node in ast.parse(code).body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.request = Mock(return_value=("Ответ по текущему вопросу", "Gemini"))
        self.extract = Mock(return_value=1)
        self.lookup = Mock(return_value="Разрешённая память")
        self.ns = {
            "Update": object, "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object),
            "OWNER_ID": 101, "TEST_USER_IDS": {202},
            "DATA_DIR": self.root, "DIALOG_STATE_FILE": self.root / "state.json",
            "DIALOG_HISTORY_FILE": self.root / "history.jsonl",
            "MAX_DIALOG_PAIRS": 30, "MAX_DIALOG_CHARS": 30000,
            "datetime": datetime, "timezone": timezone, "asyncio": asyncio,
            "dialog_lock": asyncio.Lock(), "logger": logging.getLogger(__name__),
            "personal_lifecycle": SimpleNamespace(
                has_memory_consent=lambda uid: lifecycle.has_memory_consent(uid, db_path=self.db)),
            "read_explicit_memory_prompt": self.lookup,
            "request_personal_ai": self.request, "request_gemini": Mock(),
            "remember_dialog_facts": self.extract,
            "send_long_message": AsyncMock(), "send_error_response": AsyncMock(),
        }
        import json
        import os
        self.ns.update(json=json, os=os)
        for name in ("has_personal_ai_access", "read_dialog_state", "save_dialog_state",
                     "get_session_id", "append_dialog_record", "read_recent_dialog", "private_message"):
            exec(compile(ast.Module(body=[names[name]], type_ignores=[]), "bot.py", "exec"), self.ns)
        self.ns["append_dialog_record"](101, 0, "Чужая прежняя реплика", "Чужой ответ")
        self.ns["append_dialog_record"](202, 0, "Моя прежняя реплика", "Мой прежний ответ")
        self.history_before = self.ns["DIALOG_HISTORY_FILE"].read_bytes()
        self.append = Mock(wraps=self.ns["append_dialog_record"])
        self.history = Mock(wraps=self.ns["read_recent_dialog"])
        self.ns.update(append_dialog_record=self.append, read_recent_dialog=self.history)
        self.reply = AsyncMock()
        message = SimpleNamespace(text="Мой новый вопрос", message_id=12, reply_text=self.reply)
        self.update = SimpleNamespace(
            effective_user=SimpleNamespace(id=202),
            effective_chat=SimpleNamespace(type="private"),
            effective_message=message, message=message,
        )
        self.context = SimpleNamespace(application=SimpleNamespace(bot_data={}))

    async def invoke(self):
        await self.ns["private_message"](self.update, self.context)

    async def test_without_consent_neither_history_nor_ai_nor_archive_is_used(self):
        await self.invoke()
        self.history.assert_not_called()
        self.lookup.assert_not_called()
        self.request.assert_not_called()
        self.append.assert_not_called()
        self.extract.assert_not_called()
        self.ns["send_long_message"].assert_not_awaited()
        self.assertIn("согласие", self.reply.await_args.args[0].lower())
        self.assertEqual(self.ns["DIALOG_HISTORY_FILE"].read_bytes(), self.history_before)

    async def test_consent_allows_current_users_dialog_without_changing_old_archive(self):
        lifecycle.set_memory_consent(202, True, NOW, db_path=self.db)
        await self.invoke()
        self.ns["send_error_response"].assert_not_awaited()
        self.request.assert_called_once()
        args = self.request.call_args.args
        self.assertEqual(args[0], "Мой новый вопрос")
        self.assertIn("Моя прежняя реплика", str(args[1]))
        self.assertNotIn("Чужая прежняя реплика", str(args[1]))
        self.append.assert_called_once_with(202, 0, "Мой новый вопрос", "Ответ по текущему вопросу")
        self.extract.assert_called_once_with(202, 12, "Мой новый вопрос")
        self.assertTrue(self.ns["DIALOG_HISTORY_FILE"].read_bytes().startswith(self.history_before))

    async def test_consent_revoked_during_request_prevents_archiving_and_delivery(self):
        lifecycle.set_memory_consent(202, True, NOW, db_path=self.db)

        def revoke(*args):
            lifecycle.set_memory_consent(202, False, NOW, db_path=self.db)
            return "Ответ, который уже не должен сохраняться", "Gemini"

        self.request.side_effect = revoke
        await self.invoke()
        self.request.assert_called_once()
        self.append.assert_not_called()
        self.extract.assert_not_called()
        self.ns["send_long_message"].assert_not_awaited()
        self.assertEqual(self.ns["DIALOG_HISTORY_FILE"].read_bytes(), self.history_before)

    async def test_unavailable_consent_store_fails_closed(self):
        self.ns["personal_lifecycle"].has_memory_consent = Mock(side_effect=OSError("unavailable"))
        await self.invoke()
        self.history.assert_not_called()
        self.lookup.assert_not_called()
        self.request.assert_not_called()
        self.append.assert_not_called()
        self.ns["send_error_response"].assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
