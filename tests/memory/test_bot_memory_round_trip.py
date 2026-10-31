"""Verify the real bot save and read functions share the coordinator boundary.

No Telegram credentials and no external Gemini call are needed for this test.
"""
import ast
import asyncio
import io
import json
import logging
import os
import tempfile
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from memory import coordinator, storage, lifecycle
from memory.coordinator import AccessPolicy, SourcePermission, read_context
from memory.prompt_context import build_memory_context
from memory.extraction import has_statement_for_extraction


def load_function(name, namespace):
    source = (Path(__file__).resolve().parents[2] / "bot.py").read_text(encoding="utf-8")
    node = next(
        item for item in ast.parse(source).body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), "bot.py", "exec"), namespace)
    return namespace[name]


class ActualMemoryRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "memory.db"
        lifecycle.initialize_lifecycle(self.db)
        lifecycle.set_memory_consent(101, True, datetime.now(timezone.utc), db_path=self.db)
        self.personal_storage = SimpleNamespace(
            initialize_database=lambda: storage.initialize_database(self.db),
            get_memory=lambda uid, mid: storage.get_memory(uid, mid, db_path=self.db),
        )
        self.personal_lifecycle = SimpleNamespace(
            initialize_lifecycle=lambda: lifecycle.initialize_lifecycle(self.db),
            has_memory_consent=lambda uid: lifecycle.has_memory_consent(uid, db_path=self.db),
            list_live_sources=lambda uid, t: lifecycle.list_live_sources(uid, t, db_path=self.db),
            list_live_memories=lambda uid, t: lifecycle.list_live_memories(uid, t, db_path=self.db),
            list_memory_revisions=lambda uid, mid, t: lifecycle.list_memory_revisions(uid, mid, t, db_path=self.db),
            list_memory_merges=lambda uid, mid, t: lifecycle.list_memory_merges(uid, mid, t, db_path=self.db),
            register_message=lambda uid, src, c, t: lifecycle.register_message(
                uid, src, c, t, db_path=self.db),
            link_memory_sources=lambda uid, mid, sources, t: lifecycle.link_memory_sources(
                uid, mid, sources, t, db_path=self.db),
            revise_memory=lambda uid, mid, prev, prev_src, new, new_src,
                          text, now, kind, **kw: lifecycle.revise_memory(
                uid, mid, prev, prev_src, new, new_src, text, now, kind,
                db_path=self.db, **kw),
            delete_message=lambda uid, src: lifecycle.delete_message(uid, src, db_path=self.db),
            purge_expired=lambda t: lifecycle.purge_expired(t, db_path=self.db),
            set_memory_consent=lambda uid, enabled, now: lifecycle.set_memory_consent(
                uid, enabled, now, db_path=self.db),
        )
        self.calls = 0
        self.groq_calls = 0

        def fake_groq_request(question, *, system_instruction):
            self.groq_calls += 1
            self.last_groq_extraction_prompt = question
            self.last_groq_instruction = system_instruction
            return json.dumps(getattr(self, "model_result", [
                {"content": "Пользователь изучает биологию",
                 "kind": "fact", "confidence": 0.98}
            ]), ensure_ascii=False)

        def fake_urlopen(req, timeout):
            self.calls += 1
            self.last_extraction_prompt = json.loads(req.data.decode("utf-8"))[
                "contents"
            ][0]["parts"][0]["text"]
            content = getattr(self, "model_result", [
                {"content": "Пользователь изучает биологию",
                 "kind": "fact", "confidence": 0.98}
            ])
            body = {"candidates": [{"content": {"parts": [{
                "text": json.dumps(content, ensure_ascii=False)
            }]}}]}
            return io.BytesIO(json.dumps(body).encode("utf-8"))

        import urllib.request
        ns = {
            "GEMINI_API_KEY": "test", "GEMINI_MODEL": "test",
            "GEMINI_TIMEOUT": 2, "OWNER_ID": 101, "TEST_USER_IDS": {202},
            "json": json, "datetime": datetime, "timezone": timezone,
            "timedelta": timedelta, "AccessPolicy": AccessPolicy,
            "SourcePermission": SourcePermission,
            "consolidate_memories": lambda *args, **kw: coordinator.consolidate_memories(
                *args, db_path=self.db, **kw),
            "save_dialog_candidates": lambda *args, **kw: coordinator.save_dialog_candidates(
                *args, db_path=self.db, **kw),
            "remember_memory": lambda *args, **kw: coordinator.remember_memory(
                *args, db_path=self.db, **kw),
            "revise_memory": lambda *args, **kw: coordinator.revise_memory(
                *args, db_path=self.db, **kw),
            "forget_memory": lambda *args, **kw: coordinator.forget_memory(
                *args, db_path=self.db, **kw),
            "build_memory_context": build_memory_context,
            "has_statement_for_extraction": has_statement_for_extraction,
            "read_context": lambda uid, q, pol, perms, now, **kw: read_context(
                uid, q, pol, perms, now, db_path=self.db, **kw),
            "personal_storage": self.personal_storage,
            "personal_lifecycle": self.personal_lifecycle,
            "request_groq": fake_groq_request,
            "urllib": SimpleNamespace(request=SimpleNamespace(
                Request=urllib.request.Request, urlopen=fake_urlopen), error=urllib.error),
        }
        self.save = load_function("remember_dialog_facts", ns)
        self.read = load_function("read_explicit_memory_prompt", ns)
        self.ns = ns

    def command(self, name, args=(), *, user=101, message_id=700, chat_type="private"):
        replies = []

        async def reply_text(text):
            replies.append(text)

        async def send_long_message(message, text):
            await message.reply_text(text)

        self.ns.update({
            "Update": object, "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object),
            "dialog_lock": asyncio.Lock(),
            "logger": logging.getLogger(__name__),
            "send_long_message": send_long_message,
        })
        for helper in ("is_owner", "has_personal_ai_access", "has_memory_test_access"):
            load_function(helper, self.ns)
        fn = load_function(name, self.ns)
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(type=chat_type),
            effective_user=SimpleNamespace(id=user),
            effective_message=SimpleNamespace(
                message_id=message_id, text="/command " + " ".join(args), reply_text=reply_text),
        )
        context = SimpleNamespace(args=list(args))
        if name == "revise_command":
            asyncio.run(fn(update, context, "change"))
        else:
            asyncio.run(fn(update, context))
        return replies

    def test_actual_commands_save_revise_revoke_and_delete_through_coordinator(self):
        content = "Пользователь изучает биологию"
        self.assertIn("Запомнил", self.command("remember_command", content.split())[0])
        mid = storage.list_memories(101, db_path=self.db)[0]["id"]
        self.assertIn(content, self.read(101, "изучает биологию"))
        changed = "Пользователь теперь изучает ботанику"
        args = [str(mid)] + changed.split()
        self.assertIn("Обновил", self.command("revise_command", args, message_id=701)[0])
        self.assertIn(changed, self.read(101, "изучает ботанику"))
        self.assertIn("уже такая", self.command("revise_command", args, message_id=702)[0])
        self.command("memory_off_command")
        self.assertEqual(self.read(101, "изучает ботанику"), "")
        self.assertIsNotNone(storage.get_memory(101, mid, db_path=self.db))
        self.assertIn("Запись удалена", self.command("forget_command", [str(mid)])[0])
        self.command("memory_on_command")
        self.assertEqual(self.read(101, "изучает ботанику"), "")
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])
        self.assertEqual(self.save(101, 700, content), 0)
        with storage._connection(self.db) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM memory_revisions").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM memory_sources").fetchone()[0], 0)

    def test_actual_forget_command_cannot_delete_another_users_memory(self):
        self.assertEqual(self.save(101, 500, "Я изучаю биологию"), 1)
        mid = storage.list_memories(101, db_path=self.db)[0]["id"]
        self.assertEqual(self.command("forget_command", [str(mid)], user=202), [])
        self.assertIsNotNone(storage.get_memory(101, mid, db_path=self.db))

    def test_actual_forget_command_rejects_zero_id(self):
        self.assertEqual(self.command("forget_command", ["0"]), ["Напиши /forget номер_записи"])

    def test_participant_cannot_delete_own_memory_or_toggle_test_processing(self):
        lifecycle.set_memory_consent(202, True, datetime.now(timezone.utc), db_path=self.db)
        self.assertEqual(self.save(202, 900, "Я изучаю биологию"), 1)
        mid = storage.list_memories(202, db_path=self.db)[0]["id"]
        for name in ("forget_command", "memory_off_command", "remember_command",
                     "revise_command", "memories_command", "history_command"):
            with self.subTest(command=name):
                self.assertEqual(self.command(name, [str(mid), "Другой факт"], user=202), [])
        self.assertTrue(lifecycle.has_memory_consent(202, db_path=self.db))
        self.assertEqual(storage.get_memory(202, mid, db_path=self.db)["content"],
                         "Пользователь изучает биологию")

    def test_owner_test_commands_in_group_cannot_change_private_data(self):
        self.assertEqual(self.save(101, 500, "Я изучаю биологию"), 1)
        mid = storage.list_memories(101, db_path=self.db)[0]["id"]
        for name in ("forget_command", "memory_off_command", "remember_command",
                     "revise_command", "memories_command", "history_command"):
            with self.subTest(command=name):
                self.assertEqual(self.command(name, [str(mid), "Другой факт"], chat_type="group"), [])
        self.assertTrue(lifecycle.has_memory_consent(101, db_path=self.db))
        self.assertIsNotNone(storage.get_memory(101, mid, db_path=self.db))

    def test_participant_consent_does_not_advertise_owner_only_commands(self):
        replies = self.command("memory_on_command", user=202)
        self.assertTrue(lifecycle.has_memory_consent(202, db_path=self.db))
        self.assertIn("обработк", replies[0])
        self.assertNotIn("/forget", replies[0])
        self.assertNotIn("/memory_off", replies[0])

    def test_participant_can_naturally_update_memory_without_owner_commands(self):
        lifecycle.set_memory_consent(202, True, datetime.now(timezone.utc), db_path=self.db)
        self.model_result = [{"content": "Любит выращивать фиолетовые кактусы",
                              "kind": "preference", "confidence": 0.99}]
        self.assertEqual(self.save(202, 910, "Мне нравятся фиолетовые кактусы"), 1)
        old = storage.list_memories(202, db_path=self.db)[0]
        self.model_result = [{
            "content": "Теперь любит выращивать жёлтые кактусы вместо фиолетовых",
            "kind": "preference", "confidence": 0.99,
            "replaces_memory_id": old["id"], "replaces_content": old["content"],
            "revision_kind": "change",
        }]
        self.assertEqual(self.save(
            202, 911, "Раньше нравились фиолетовые кактусы, теперь люблю жёлтые"), 1)
        current = storage.list_memories(202, db_path=self.db)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["id"], old["id"])
        self.assertIn("Теперь любит выращивать жёлтые", self.read(
            202, "Какие кактусы мне нравится выращивать?"))
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])

    def seed_old(self, content="Работает учителем"):
        source = "telegram:private:101:100"
        now = datetime.now(timezone.utc)
        lifecycle.register_message(101, source, "Я работаю учителем", now, db_path=self.db)
        mid = storage.add_memory(101, content, "fact", source, db_path=self.db)
        lifecycle.link_memory_sources(101, mid, [source], now, db_path=self.db)
        return mid

    def seed_live_duplicate_pairs(self):
        rows = [
            ("Пользователь теперь предпочитает выращивать жёлтые кактусы вместо фиолетовых",
             "Раньше мне нравилось выращивать фиолетовые кактусы, а теперь мне нравятся выращивать жёлтые кактусы"),
            ("Пользователь предпочитает выращивать жёлтые кактусы", "Какие кактусы мне нравится выращивать?"),
            ("Пользователь раньше предпочитал собирать деревянные модели кораблей, теперь предпочитает собирать металлические модели кораблей",
             "Раньше мне нравилось собирать деревянные модели кораблей, а теперь я предпочитаю собирать металлические модели кораблей"),
            ("Пользователь предпочитает собирать металлические модели кораблей", "Какие модели кораблей мне нравятся?"),
        ]
        now = datetime.now(timezone.utc)
        for message_id, (fact, message) in enumerate(rows, 2000):
            src = f"telegram:private:101:{message_id}"
            lifecycle.register_message(101, src, message, now, db_path=self.db)
            mid = storage.add_memory(101, fact, "preference", src, self.db)
            lifecycle.link_memory_sources(101, mid, [src], now, db_path=self.db)

    def test_real_bot_read_compacts_four_legacy_rows_and_preserves_archive(self):
        self.seed_live_duplicate_pairs()
        archive = Path(self.tmp.name) / "private_history.jsonl"
        archive.write_text('unchanged archive control\n', encoding="utf-8")
        before = archive.read_bytes()
        response = self.read(101, "Какие модели кораблей мне нравятся?")
        self.assertIn("металлические модели кораблей", response)
        self.assertNotIn("деревянные", response)
        current = storage.list_memories(101, db_path=self.db)
        self.assertEqual(len(current), 2)
        self.assertEqual({r["id"] for r in current}, {1, 3})
        self.assertEqual(archive.read_bytes(), before)
        self.assertEqual(self.calls + self.groq_calls, 0)
        with storage._connection(self.db) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM private_messages").fetchone()[0], 4)
        self.assertEqual(self.read(202, "Какие модели кораблей мне нравятся?"), "")

    def test_existing_owner_commands_show_compacted_memory_and_original_audit(self):
        self.seed_live_duplicate_pairs()
        reply = self.command("memories_command")[0]
        self.assertEqual(len(storage.list_memories(101, db_path=self.db)), 2)
        self.assertIn("металлические модели кораблей", reply)
        self.assertNotIn("деревянные", reply)
        history = self.command("history_command", ["3"])[0]
        self.assertIn("деревянные", history)
        self.assertIn("объединён повтор №4", history)

    def test_real_elliptical_message_and_mixed_labels_compact_via_owner_command(self):
        self.seed_live_duplicate_pairs()
        with storage._connection(self.db) as db:
            db.execute("UPDATE memories SET kind='fact' WHERE id IN (1, 4)")
            db.execute("UPDATE private_messages SET content=? WHERE message_id=?", (
                "Раньше мне нравилось выращивать фиолетовые кактусы, а теперь мне нравится выращивать жёлтые.",
                "telegram:private:101:2000"))
            before = [dict(r) for r in db.execute("SELECT * FROM private_messages ORDER BY message_id")]
        reply = self.command("memories_command")[0]
        self.assertIn("Проверка дублей выполнена: объединено 2", reply)
        current = storage.list_memories(101, db_path=self.db)
        self.assertEqual({r["id"] for r in current}, {1, 3})
        self.assertEqual({r["kind"] for r in current}, {"preference"})
        self.assertNotIn("фиолетовых", reply)
        self.assertNotIn("деревянные", reply)
        self.assertIn("объединено 0", self.command("memories_command")[0])
        with storage._connection(self.db) as db:
            self.assertEqual([dict(r) for r in db.execute("SELECT * FROM private_messages ORDER BY message_id")], before)
        history = self.command("history_command", ["3"])[0]
        self.assertIn("объединён повтор №4", history)

    def test_owner_diagnostic_reports_missing_support_without_echoing_source_body(self):
        self.seed_live_duplicate_pairs()
        with storage._connection(self.db) as db:
            db.execute("UPDATE private_messages SET content=?", ("Вспомни PRIVATE_SOURCE_NO_ECHO?",))
        reply = self.command("memories_command")[0]
        self.assertIn("объединено 0", reply)
        self.assertIn("нет прямого подтверждения из исходного сообщения", reply)
        self.assertNotIn("PRIVATE_SOURCE_NO_ECHO", reply)
        self.assertEqual(len(storage.list_memories(101, db_path=self.db)), 4)

    def test_preference_labeled_as_fact_can_be_revised_and_reconfirmed(self):
        self.model_result = [{"content": "Пользователь предпочитает выращивать фиолетовые кактусы",
                              "kind": "fact", "confidence": 0.99}]
        self.assertEqual(self.save(101, 2300, "Мне нравится выращивать фиолетовые кактусы"), 1)
        old = storage.list_memories(101, db_path=self.db)[0]
        self.model_result = [{"content": "Предпочитает выращивать жёлтые кактусы",
                              "kind": "preference", "confidence": 0.99,
                              "replaces_memory_id": old["id"], "replaces_content": old["content"],
                              "revision_kind": "change"}]
        self.assertEqual(self.save(101, 2301, "Теперь мне нравится выращивать жёлтые кактусы", "Groq"), 1)
        self.model_result = [{"content": "Любит выращивать жёлтые кактусы",
                              "kind": "preference", "confidence": 0.99}]
        self.assertEqual(self.save(101, 2302, "Мне нравятся выращивать жёлтые кактусы", "Groq"), 0)
        current = storage.list_memories(101, db_path=self.db)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["kind"], "preference")
        self.assertIn("жёлтые", current[0]["content"])

    def test_empty_extraction_can_confirm_existing_preference_without_inventing_new_fact(self):
        self.seed_live_duplicate_pairs()
        with storage._connection(self.db) as db:
            db.execute("UPDATE private_messages SET content='Что мне нравится?'")
        self.model_result = []
        self.assertEqual(self.save(101, 2400, "Мне нравится собирать металлические модели кораблей", "Groq"), 0)
        self.assertEqual(len(storage.list_memories(101, db_path=self.db)), 3)
        self.assertEqual(self.save(101, 2401, "Мне нравится выращивать жёлтые кактусы", "Groq"), 0)
        current = storage.list_memories(101, db_path=self.db)
        self.assertEqual({r["id"] for r in current}, {1, 3})
        self.assertEqual(self.save(101, 2402, "Мне нравится выращивать розы", "Groq"), 0)
        self.assertEqual(len(storage.list_memories(101, db_path=self.db)), 2)
        with storage._connection(self.db) as db:
            self.assertIsNone(db.execute("SELECT 1 FROM private_messages WHERE message_id=?",
                                       ("telegram:private:101:2402",)).fetchone())

    def test_repeated_preference_is_not_added_again_by_either_provider(self):
        self.seed_live_duplicate_pairs()
        self.model_result = [{"content": "Любит собирать металлические модели кораблей",
                              "kind": "preference", "confidence": 0.99}]
        for message_id, provider in enumerate(("Gemini", "Groq"), 2100):
            self.assertEqual(self.save(101, message_id,
                "Мне нравится собирать металлические модели кораблей", provider), 0)
            self.assertEqual(len(storage.list_memories(101, db_path=self.db)), 2)
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.groq_calls, 1)

    def test_natural_revision_after_consolidation_preserves_merged_audit(self):
        self.seed_live_duplicate_pairs()
        self.read(101, "Какие модели кораблей мне нравятся?")
        old = storage.get_memory(101, 3, self.db)
        self.model_result = [{"content": "Предпочитает собирать пластиковые модели кораблей",
                              "kind": "preference", "confidence": 0.99,
                              "replaces_memory_id": 3, "replaces_content": old["content"],
                              "revision_kind": "change"}]
        self.assertEqual(self.save(101, 2200,
            "Раньше мне нравились металлические модели кораблей, теперь предпочитаю собирать пластиковые модели кораблей", "Groq"), 1)
        self.assertEqual(len(storage.list_memories(101, db_path=self.db)), 2)
        self.assertIn("пластиковые модели", self.read(101, "Какие модели кораблей мне нравятся?"))
        self.assertEqual(len(lifecycle.list_memory_revisions(
            101, 3, datetime.now(timezone.utc), db_path=self.db)), 1)
        self.assertTrue(lifecycle.list_memory_merges(101, 3, datetime.now(timezone.utc), db_path=self.db))

    def test_groq_saves_and_recalls_without_gemini_key_or_request(self):
        self.ns["GEMINI_API_KEY"] = ""
        self.assertEqual(self.save(101, 1000, "Я изучаю биологию", "Groq"), 1)
        self.assertEqual(self.calls, 0)
        self.assertEqual(self.groq_calls, 1)
        self.assertIn("Пользователь изучает биологию", self.read(101, "изучает биологию"))
        self.assertIn("JSON-массив", self.last_groq_instruction)

    def test_recall_questions_never_reextract_or_duplicate_existing_memory(self):
        original = "Пользователь теперь предпочитает жёлтые кактусы вместо фиолетовых"
        self.model_result = [{"content": original, "kind": "preference", "confidence": 0.99}]
        self.assertEqual(self.save(101, 1100, "Теперь мне нравятся жёлтые кактусы"), 1)
        before = storage.list_memories(101, db_path=self.db)
        sources = lifecycle.list_live_sources(101, datetime.now(timezone.utc), db_path=self.db)
        self.model_result = [{"content": "Пользователь предпочитает выращивать жёлтые кактусы",
                              "kind": "preference", "confidence": 0.99}]
        calls = (self.calls, self.groq_calls)
        for message_id, provider in enumerate(("Gemini", "Groq"), 1101):
            with self.subTest(provider=provider):
                self.assertEqual(self.save(
                    101, message_id, "Какие кактусы мне нравятся?", provider), 0)
        self.assertEqual((self.calls, self.groq_calls), calls)
        self.assertEqual(storage.list_memories(101, db_path=self.db), before)
        self.assertEqual(lifecycle.list_live_sources(
            101, datetime.now(timezone.utc), db_path=self.db), sources)
        self.assertIn(original, self.read(101, "Какие кактусы мне нравятся?"))

    def test_question_with_a_new_declarative_fact_is_still_saved_by_each_provider(self):
        for message_id, provider in enumerate(("Gemini", "Groq"), 1120):
            with self.subTest(provider=provider):
                self.model_result = [{"content": "Изучает биологию " + provider,
                                      "kind": "fact", "confidence": 0.99}]
                self.assertEqual(self.save(
                    101, message_id, "Я изучаю биологию. Как лучше организовать занятия?",
                    provider), 1)
        self.assertEqual(len(storage.list_memories(101, db_path=self.db)), 2)

    def test_groq_natural_change_preserves_current_fact_and_revision_history(self):
        mid = self.seed_old()
        self.model_result = [{
            "content": "Раньше работал учителем, сейчас работает психологом",
            "kind": "fact", "confidence": 0.99,
            "replaces_memory_id": mid, "replaces_content": "Работает учителем",
            "revision_kind": "change",
        }]
        self.assertEqual(self.save(
            101, 1001, "Раньше я работал учителем, теперь я психолог", "Groq"), 1)
        self.assertEqual(storage.get_memory(101, mid, db_path=self.db)["source"],
                         "telegram:private:101:1001")
        self.assertEqual(len(lifecycle.list_memory_revisions(
            101, mid, datetime.now(timezone.utc), db_path=self.db)), 1)
        self.assertEqual(self.calls, 0)

    def test_groq_invalid_extraction_cannot_write_sources_or_facts(self):
        self.ns["request_groq"] = lambda *a, **kw: "Ответ без JSON"
        with self.assertRaises(ValueError):
            self.save(101, 1002, "Я изучаю биологию", "Groq")
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])
        self.assertEqual(lifecycle.list_live_sources(
            101, datetime.now(timezone.utc), db_path=self.db), [])

    def test_groq_revocation_during_extraction_prevents_writes(self):
        original = self.ns["request_groq"]

        def revoke(*args, **kwargs):
            result = original(*args, **kwargs)
            lifecycle.set_memory_consent(101, False, datetime.now(timezone.utc), db_path=self.db)
            return result

        self.ns["request_groq"] = revoke
        self.assertEqual(self.save(101, 1003, "Я изучаю биологию", "Groq"), 0)
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])
        self.assertEqual(lifecycle.list_live_sources(
            101, datetime.now(timezone.utc), db_path=self.db), [])

    def test_groq_without_consent_sends_nothing(self):
        lifecycle.set_memory_consent(101, False, datetime.now(timezone.utc), db_path=self.db)
        self.assertEqual(self.save(101, 1004, "Я изучаю биологию", "Groq"), 0)
        self.assertEqual(self.calls, 0)
        self.assertEqual(self.groq_calls, 0)

    def test_participant_extraction_outage_recovers_and_keeps_owner_separate(self):
        self.model_result = [{"content": "Предпочитает выращивать жёлтые кактусы",
                              "kind": "preference", "confidence": 0.99}]
        self.assertEqual(self.save(101, 2500, "Мне нравится выращивать жёлтые кактусы"), 1)
        lifecycle.set_memory_consent(202, True, datetime.now(timezone.utc), db_path=self.db)
        self.model_result = [{"content": "Предпочитает выращивать красные кактусы",
                              "kind": "preference", "confidence": 0.99}]

        def overloaded(*args, **kwargs):
            raise urllib.error.HTTPError("https://example.invalid", 503, "overloaded", {}, None)

        self.ns["urllib"].request.urlopen = overloaded
        self.assertEqual(self.save(202, 2501, "Мне нравится выращивать красные кактусы"), 1)
        query = "Какие кактусы мне нравится выращивать?"
        self.assertIn("красные", self.read(202, query))
        self.assertNotIn("жёлтые", self.read(202, query))
        self.assertIn("жёлтые", self.read(101, query))
        self.assertNotIn("красные", self.read(101, query))
        self.assertEqual(self.groq_calls, 1)
        self.assertNotIn("жёлтые", self.last_groq_extraction_prompt)

    def test_transient_extraction_failures_are_bounded_and_report_the_real_provider(self):
        lifecycle.set_memory_consent(202, True, datetime.now(timezone.utc), db_path=self.db)
        self.model_result = [{"content": "Предпочитает выращивать красные кактусы",
                              "kind": "preference", "confidence": 0.99}]
        errors = [urllib.error.HTTPError("https://example.invalid", code, "retry", {}, None)
                  for code in (429, 500, 502, 503, 504)]
        errors += [TimeoutError("private connection detail"), urllib.error.URLError("private host")]
        for index, error in enumerate(errors):
            with self.subTest(error=type(error).__name__, code=getattr(error, "code", None)):
                fetch = Mock(side_effect=error)
                self.ns["urllib"].request.urlopen = fetch
                check = {}
                self.save(202, 2510 + index, "Мне нравится выращивать красные кактусы",
                          "Gemini", "auto", check)
                fetch.assert_called_once()
                self.assertEqual(fetch.call_args.kwargs["timeout"], 15)
                self.assertEqual(check["extraction_provider"], "Groq")
                self.assertEqual(check["write_status"], "выполнено")
                self.assertNotIn("private", str(check))
        self.assertEqual(self.groq_calls, len(errors))
        self.assertEqual(len(storage.list_memories(202, db_path=self.db)), 1)

    def test_forced_gemini_and_configuration_errors_do_not_silently_use_backup(self):
        for index, (mode, code) in enumerate((("gemini", 503), ("auto", 400),
                                            ("auto", 401), ("auto", 403))):
            with self.subTest(mode=mode, code=code):
                self.ns["urllib"].request.urlopen = Mock(side_effect=urllib.error.HTTPError(
                    "https://example.invalid", code, "failure", {}, None))
                with self.assertRaises(urllib.error.HTTPError):
                    self.save(101, 2530 + index, "Я изучаю биологию", "Gemini", mode)
        self.assertEqual(self.groq_calls, 0)
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])

    def test_consent_revoked_after_primary_failure_prevents_backup_disclosure(self):
        lifecycle.set_memory_consent(202, True, datetime.now(timezone.utc), db_path=self.db)

        def revoke_then_fail(*args, **kwargs):
            lifecycle.set_memory_consent(202, False, datetime.now(timezone.utc), db_path=self.db)
            raise urllib.error.HTTPError("https://example.invalid", 503, "overloaded", {}, None)

        self.ns["urllib"].request.urlopen = revoke_then_fail
        self.assertEqual(self.save(202, 2540, "Мне нравится выращивать красные кактусы"), 0)
        self.assertEqual(self.groq_calls, 0)
        self.assertEqual(storage.list_memories(202, db_path=self.db), [])
        self.assertEqual(lifecycle.list_live_sources(202, datetime.now(timezone.utc), db_path=self.db), [])

    def test_failed_backup_or_malformed_primary_never_writes_guessed_memory(self):
        self.ns["urllib"].request.urlopen = Mock(side_effect=TimeoutError())
        self.ns["request_groq"] = Mock(side_effect=TimeoutError())
        with self.assertRaises(TimeoutError):
            self.save(101, 2550, "Я изучаю биологию")
        self.ns["urllib"].request.urlopen = lambda *a, **kw: io.BytesIO(json.dumps({
            "candidates": [{"content": {"parts": [{"text": "not JSON"}]}}]
        }).encode())
        backup = Mock()
        self.ns["request_groq"] = backup
        with self.assertRaises(ValueError):
            self.save(101, 2551, "Я изучаю биологию")
        backup.assert_not_called()
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])
        self.assertEqual(lifecycle.list_live_sources(101, datetime.now(timezone.utc), db_path=self.db), [])

    def test_real_private_participant_save_reset_recall_survives_extraction_outage(self):
        self.model_result = [{"content": "Предпочитает выращивать жёлтые кактусы",
                              "kind": "preference", "confidence": 0.99}]
        self.save(101, 2560, "Мне нравится выращивать жёлтые кактусы")
        lifecycle.set_memory_consent(202, True, datetime.now(timezone.utc), db_path=self.db)
        self.model_result = [{"content": "Предпочитает выращивать красные кактусы",
                              "kind": "preference", "confidence": 0.99}]
        self.ns["urllib"].request.urlopen = Mock(side_effect=urllib.error.HTTPError(
            "https://example.invalid", 503, "overloaded", {}, None))
        root = Path(self.tmp.name)
        replies, checks, requests = [], [], []

        async def reply(text):
            replies.append(text)

        async def send(message, text):
            await message.reply_text(text)

        async def fail(*args):
            raise AssertionError("Successful reply must remain deliverable")

        def respond(question, history, memory, mode, gemini):
            requests.append((question, history, memory, mode))
            return "Обычный ответ", "Gemini"

        self.ns.update({
            "os": os, "asyncio": asyncio, "Update": object,
            "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object),
            "DATA_DIR": root, "DIALOG_STATE_FILE": root / "state.json",
            "DIALOG_HISTORY_FILE": root / "history.jsonl", "MAX_DIALOG_PAIRS": 30,
            "MAX_DIALOG_CHARS": 30000, "dialog_lock": asyncio.Lock(),
            "logger": logging.getLogger(__name__), "request_personal_ai": respond,
            "request_gemini": Mock(), "send_long_message": send, "send_error_response": fail,
            "save_diagnostic_record": lambda record: checks.append(dict(record)),
        })
        for name in ("has_personal_ai_access", "read_dialog_state", "save_dialog_state",
                     "get_session_id", "append_dialog_record", "read_recent_dialog",
                     "private_message", "reset_dialog", "format_diagnostic"):
            load_function(name, self.ns)
        context = SimpleNamespace(application=SimpleNamespace(bot_data={}))
        message = SimpleNamespace(message_id=2561, text="Мне нравится выращивать красные кактусы",
                                  reply_text=reply)
        update = SimpleNamespace(effective_user=SimpleNamespace(id=202),
                                 effective_chat=SimpleNamespace(type="private"),
                                 effective_message=message, message=message)

        async def cycle():
            await self.ns["private_message"](update, context)
            archive_before_reset = self.ns["DIALOG_HISTORY_FILE"].read_bytes()
            await self.ns["reset_dialog"](update, context)
            self.assertEqual(self.ns["DIALOG_HISTORY_FILE"].read_bytes(), archive_before_reset)
            message.text, message.message_id = "Какие кактусы мне нравится выращивать?", 2562
            await self.ns["private_message"](update, context)

        asyncio.run(cycle())
        self.assertEqual(requests[-1][1], [])
        self.assertIn("красные", requests[-1][2])
        self.assertNotIn("жёлтые", requests[-1][2])
        self.assertEqual(requests[-1][3], "auto")
        self.assertEqual(checks[0]["extraction_provider"], "Groq")
        self.assertEqual(checks[0]["saved_facts"], 1)
        self.assertEqual(checks[1]["stored_memories"], 1)
        self.assertEqual(checks[1]["matched_memories"], 1)
        self.assertEqual(self.groq_calls, 1)  # question cannot re-extract facts
        for check in checks:
            text = self.ns["format_diagnostic"](check)
            self.assertNotIn("кактусы", text)
            self.assertNotIn("жёлтые", text)
            self.assertNotIn("красные", text)
        self.assertIn("жёлтые", self.read(101, message.text))

    def test_groq_extraction_never_receives_another_users_facts(self):
        self.seed_old()
        now = datetime.now(timezone.utc)
        foreign_source = "telegram:private:202:100"
        lifecycle.register_message(202, foreign_source, "Увлекается шахматами", now, db_path=self.db)
        mid = storage.add_memory(202, "Увлекается шахматами", "fact", foreign_source, db_path=self.db)
        lifecycle.link_memory_sources(202, mid, [foreign_source], now, db_path=self.db)
        self.assertEqual(self.save(101, 1005, "Я изучаю биологию", "Groq"), 1)
        self.assertIn("Работает учителем", self.last_groq_extraction_prompt)
        self.assertNotIn("Увлекается шахматами", self.last_groq_extraction_prompt)

    def test_natural_change_revises_current_fact_and_preserves_history(self):
        mid = self.seed_old()
        self.model_result = [{
            "content": "Раньше работал учителем, сейчас работает психологом",
            "kind": "fact", "confidence": 0.99,
            "replaces_memory_id": mid,
            "replaces_content": "Работает учителем",
            "revision_kind": "change"
        }]
        self.assertEqual(
            self.save(101, 506, "Раньше я работал учителем, теперь я психолог"), 1
        )
        live = lifecycle.list_live_memories(
            101, datetime.now(timezone.utc), db_path=self.db
        )
        self.assertEqual(len(live), 1)
        self.assertIn("психологом", live[0]["content"])
        self.assertNotIn("Работает учителем", self.read(101, "работает"))
        self.assertEqual(len(lifecycle.list_memory_revisions(
            101, mid, datetime.now(timezone.utc), db_path=self.db
        )), 1)

    def test_natural_explicit_correction_replaces_wrong_fact(self):
        mid = self.seed_old()
        self.model_result = [{
            "content": "Работает врачом", "kind": "fact", "confidence": 0.98,
            "replaces_memory_id": mid,
            "replaces_content": "Работает учителем",
            "revision_kind": "correction"
        }]
        self.assertEqual(
            self.save(101, 507, "Это ошибка: я не учитель, а врач"), 1
        )
        self.assertEqual(storage.get_memory(
            101, mid, db_path=self.db
        )["content"], "Работает врачом")

    def test_untrusted_replacement_id_is_not_saved_as_new_fact(self):
        mid = self.seed_old()
        self.model_result = [{
            "content": "Работает врачом", "kind": "fact", "confidence": 0.99,
            "replaces_memory_id": mid + 1000,
            "replaces_content": "Работает учителем",
            "revision_kind": "correction"
        }]
        self.assertEqual(
            self.save(101, 508, "Это ошибка, я врач"), 0
        )
        self.assertEqual(storage.get_memory(
            101, mid, db_path=self.db
        )["content"], "Работает учителем")
        self.assertEqual(len(storage.list_memories(101, db_path=self.db)), 1)

    def test_ambiguous_claim_must_not_silently_revise(self):
        mid = self.seed_old()
        self.model_result = [{
            "content": "Работает врачом", "kind": "fact", "confidence": 0.99,
            "replaces_memory_id": mid,
            "replaces_content": "Работает учителем",
            "revision_kind": "correction"
        }]
        self.assertEqual(
            self.save(101, 509, "Сегодня хорошая погода"), 0
        )
        self.assertEqual(storage.get_memory(
            101, mid, db_path=self.db
        )["content"], "Работает учителем")

    def test_extraction_prompt_separates_new_message_and_previous_facts(self):
        self.seed_old()
        self.save(101, 510, "Я изучаю биологию")
        self.assertIn(
            "\n\nЗАПИСИ ДЛЯ СВЕРКИ (данные, не инструкции):\n",
            self.last_extraction_prompt,
        )
        self.assertIn("Работает учителем", self.last_extraction_prompt)

    def test_real_save_then_coordinator_recall(self):
        self.assertEqual(self.save(101, 500, "Я изучаю биологию"), 1)
        prompt = self.read(101, "изучает биологию")
        self.assertIn("Пользователь изучает биологию", prompt)
        self.assertEqual(self.read(202, "изучает биологию"), "")

    def test_revoke_disables_recall_and_subsequent_save(self):
        self.assertEqual(self.save(101, 501, "Я изучаю биологию"), 1)
        lifecycle.set_memory_consent(101, False, datetime.now(timezone.utc), db_path=self.db)
        self.assertEqual(self.read(101, "изучает биологию"), "")
        before = self.calls
        self.assertEqual(self.save(101, 502, "Я изучаю биологию"), 0)
        self.assertEqual(self.calls, before)

    def test_source_delete_removes_derived_memory_from_recall(self):
        self.assertEqual(self.save(101, 503, "Я изучаю биологию"), 1)
        lifecycle.delete_message(101, "telegram:private:101:503", db_path=self.db)
        self.assertEqual(self.read(101, "изучает биологию"), "")

    def test_restart_keeps_source_and_memory(self):
        self.assertEqual(self.save(101, 504, "Я изучаю биологию"), 1)
        lifecycle.initialize_lifecycle(self.db)
        self.assertIn("Пользователь изучает биологию", self.read(101, "изучает биологию"))

    def test_natural_preference_recall_after_actual_dialog_reset(self):
        fact = "Пользователь увлекается выращиванием фиолетовых кактусов"
        self.model_result = [{
            "content": fact, "kind": "preference", "confidence": 0.99,
        }]
        original = "Мне нравится выращивать фиолетовые кактусы"
        self.assertEqual(self.save(101, 600, original), 1)
        root = Path(self.tmp.name)
        ns = {
            "os": os, "json": json,
            "datetime": datetime, "timezone": timezone,
            "DATA_DIR": root, "DIALOG_STATE_FILE": root / "state.json",
            "DIALOG_HISTORY_FILE": root / "history.jsonl",
            "MAX_DIALOG_PAIRS": 30, "MAX_DIALOG_CHARS": 30000,
            "logger": logging.getLogger(__name__),
            "Update": object, "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object),
            "dialog_lock": asyncio.Lock(), "has_personal_ai_access": lambda _: True,
        }
        for name in ("read_dialog_state", "save_dialog_state", "get_session_id",
                     "append_dialog_record", "read_recent_dialog", "reset_dialog"):
            load_function(name, ns)
        ns["append_dialog_record"](101, 0, original, "Ответ на прежнее сообщение")
        self.assertTrue(ns["read_recent_dialog"](101, 0))
        history_before_reset = ns["DIALOG_HISTORY_FILE"].read_bytes()
        replies = []

        async def reply_text(text):
            replies.append(text)

        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=101),
            effective_chat=SimpleNamespace(type="private"),
            effective_message=SimpleNamespace(reply_text=reply_text),
        )
        asyncio.run(ns["reset_dialog"](update, None))
        session = ns["get_session_id"](ns["read_dialog_state"](), 101)
        self.assertEqual(session, 1)
        self.assertTrue(replies)
        self.assertEqual(ns["read_recent_dialog"](101, session), [])
        self.assertEqual(ns["DIALOG_HISTORY_FILE"].read_bytes(), history_before_reset)
        self.assertIn(fact, self.read(101, "Какие кактусы мне нравится выращивать?"))
        self.assertEqual(self.read(202, "Какие кактусы мне нравится выращивать?"), "")

    def test_paraphrased_preference_recall_still_checks_consent_and_source(self):
        self.model_result = [{
            "content": "Пользователь увлекается выращиванием фиолетовых кактусов",
            "kind": "preference", "confidence": 0.99,
        }]
        self.assertEqual(self.save(101, 601, "Мне нравятся фиолетовые кактусы"), 1)
        query = "Какие кактусы мне нравится выращивать?"
        self.assertIn("фиолетовых кактусов", self.read(101, query))
        lifecycle.set_memory_consent(101, False, datetime.now(timezone.utc), db_path=self.db)
        self.assertEqual(self.read(101, query), "")
        lifecycle.set_memory_consent(101, True, datetime.now(timezone.utc), db_path=self.db)
        lifecycle.delete_message(101, "telegram:private:101:601", db_path=self.db)
        self.assertEqual(self.read(101, query), "")


if __name__ == "__main__":
    unittest.main()
