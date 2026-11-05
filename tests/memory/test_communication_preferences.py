"""Actual bot/coordinator personalization with temporary SQLite and fake AI."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

import test_bot_memory_round_trip as bot_harness
from memory import coordinator, lifecycle, storage
from memory.personalization import communication_length


class CommunicationPreferencesTests(unittest.TestCase):
    def setUp(self):
        self.bot = bot_harness.ActualMemoryRoundTripTests(methodName="runTest")
        self.bot.setUp()
        self.addCleanup(self.bot.doCleanups)
        self.db = self.bot.db
        self.bot.model_result = []

    def save(self, text="Мне удобнее короткие ответы", message_id=5000, **kwargs):
        diagnostic = {}
        result = self.bot.save(101, message_id, text, diagnostics=diagnostic, **kwargs)
        self.assertEqual(result, 0)
        return diagnostic

    def rows(self):
        with storage._connection(self.db) as db:
            return [dict(row) for row in db.execute("SELECT * FROM communication_preferences ORDER BY message_id")]

    def profile(self, user=101, now=None, sources=None):
        now = now or datetime.now(timezone.utc)
        if sources is None:
            sources = {row["source"] for row in lifecycle.list_live_sources(user, now, db_path=self.db)}
        return storage.get_communication_preference(user, sources, now, db_path=self.db)

    def test_natural_preference_survives_reset_and_applies_without_matching_facts(self):
        self.assertEqual(self.save()["saved_communication_preferences"], 1)
        state = {"101": 0}
        self.bot.ns.update(read_dialog_state=lambda: state,
                           save_dialog_state=lambda value: state.update(value),
                           get_session_id=lambda value, user: value.get(str(user), 0))
        self.bot.command("reset_dialog", message_id=5001)
        self.assertEqual(state["101"], 1)
        result = self.bot.read(101, "Какие бывают комнатные растения?")
        self.assertIn("краткие ответы", result)
        self.assertNotIn("ПОДТВЕРЖДЁННЫЕ СВЕДЕНИЯ", result)
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])
        lifecycle.initialize_lifecycle(self.db)
        self.assertEqual(self.profile()["response_length"], "brief")

    def test_new_explicit_preference_supersedes_old_without_rewriting_messages(self):
        self.save(message_id=5000)
        self.save("Теперь мне удобнее подробные ответы", 5001)
        self.assertEqual(self.profile()["response_length"], "detailed")
        self.assertEqual([row["status"] for row in self.rows()], ["superseded", "active"])
        with storage._connection(self.db) as db:
            self.assertEqual(db.execute("SELECT content FROM private_messages WHERE message_id=?",
                ("telegram:private:101:5000",)).fetchone()[0], "Мне удобнее короткие ответы")
        self.assertIn("подробные ответы", self.bot.read(101, "Расскажи о растениях"))

    def test_current_request_overrides_style_without_changing_saved_preference(self):
        self.save()
        result = self.bot.read(101, "Сейчас объясни подробно, как выращивать кактусы?")
        self.assertIn("подробные ответы", result)
        self.assertNotIn("краткие ответы", result)
        self.assertEqual(self.profile()["response_length"], "brief")
        self.assertIn("краткие ответы", self.bot.read(101, "Как выращивать кактусы?"))

    def test_one_answer_instruction_does_not_become_a_permanent_setting(self):
        for i, text in enumerate(("Сейчас объясни подробно", "Ответь кратко", "В этот раз пиши коротко")):
            self.save(text, 5010 + i)
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.bot.read(101, "Как выращивать кактусы?"), "")

    def test_only_bounded_explicit_self_statements_create_preferences(self):
        bad = ('«Мне удобнее короткие ответы»', "Моей жене удобнее короткие ответы",
               "Мне удобнее короткие ответы?", "Не отвечай кратко", "Короткие ответы",
               "Всегда отвечай кратко и подробно", "/remember Мне удобнее короткие ответы",
               "В статье написано: мне удобнее короткие ответы", "Мне удобнее короткие ответы, если я спешу")
        for i, text in enumerate(bad):
            self.save(text, 5020 + i)
        self.assertEqual(self.rows(), [])

    def test_future_instruction_and_politeness_are_explicit_durable_preferences(self):
        for text in ("Всегда отвечай кратко", "Пожалуйста, впредь пиши мне подробно.",
                     "Я предпочитаю развёрнутые ответы", "Мне нравятся краткие ответы, пожалуйста"):
            with self.subTest(text=text):
                self.assertIn(communication_length(text, persistent=True), ("brief", "detailed"))

    def test_model_generated_style_without_explicit_source_cannot_set_profile(self):
        self.bot.model_result = [dict(content="Предпочитает краткие ответы", kind="preference", confidence=1)]
        self.bot.save(101, 5040, "Я сегодня читал книгу")
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.bot.read(101, "Как выращивать кактусы?"), "")

    def test_without_consent_no_extraction_or_preference_write(self):
        lifecycle.set_memory_consent(101, False, datetime.now(timezone.utc), db_path=self.db)
        self.save()
        self.assertEqual(self.bot.calls, 0)
        self.assertEqual(self.rows(), [])

    def test_revocation_blocks_read_and_write_but_preserves_consented_history(self):
        self.save()
        self.bot.command("memory_off_command")
        self.assertEqual(self.bot.read(101, "Расскажи о растениях"), "")
        self.save("Мне удобнее подробные ответы", 5050)
        self.assertEqual(len(self.rows()), 1)
        self.bot.command("memory_on_command")
        self.assertEqual(self.profile()["response_length"], "brief")

    def test_atomic_write_rechecks_stored_consent(self):
        original = storage.save_communication_preference
        def revoke_then_save(*args, **kwargs):
            lifecycle.set_memory_consent(101, False, datetime.now(timezone.utc), db_path=self.db)
            return original(*args, **kwargs)
        with patch.object(storage, "save_communication_preference", side_effect=revoke_then_save):
            with self.assertRaises(PermissionError):
                self.save()
        self.assertEqual(self.rows(), [])

    def test_another_account_does_not_inherit_style(self):
        self.save()
        lifecycle.set_memory_consent(202, True, datetime.now(timezone.utc), db_path=self.db)
        self.assertEqual(self.bot.read(202, "Расскажи о растениях"), "")
        self.assertIsNone(self.profile(user=202))

    def test_unverified_source_permission_does_not_expose_setting(self):
        self.save()
        self.assertIsNone(self.profile(sources=set()))
        self.assertIsNone(self.profile(sources={"telegram:private:202:5000"}))

    def test_old_api_keeps_personalization_opt_in(self):
        self.save()
        now = datetime.now(timezone.utc)
        permissions = [coordinator.SourcePermission(**row)
                       for row in lifecycle.list_live_sources(101, now, db_path=self.db)]
        policy = coordinator.AccessPolicy(101, True, True, True, (now + timedelta(minutes=1)).isoformat())
        ordinary = coordinator.read_context(101, "Растения", policy, permissions, now.isoformat(), db_path=self.db)
        self.assertIsNone(ordinary["personalization"])
        enabled = coordinator.read_context(101, "Растения", policy, permissions, now.isoformat(), db_path=self.db,
                                           include_communication_preferences=True)
        self.assertEqual(enabled["personalization"]["response_length"], "brief")

    def test_deleting_current_source_never_reactivates_superseded_style(self):
        self.save(message_id=5060)
        self.save("Мне удобнее подробные ответы", 5061)
        lifecycle.delete_message(101, "telegram:private:101:5061", db_path=self.db)
        self.assertIsNone(self.profile())
        self.assertEqual([row["status"] for row in self.rows()], ["superseded"])
        self.assertEqual(self.bot.read(101, "Расскажи о растениях"), "")

    def test_expiry_uses_original_source_date_without_extending_it(self):
        created = datetime.now(timezone.utc) - timedelta(days=364)
        self.save(source_created_at=created)
        with storage._connection(self.db) as db:
            expiry = db.execute("SELECT expires_at FROM private_messages WHERE message_id=?",
                ("telegram:private:101:5000",)).fetchone()[0]
        self.assertEqual(expiry, (created + timedelta(days=365)).isoformat())
        self.assertIsNone(self.profile(now=created + timedelta(days=365)))
        lifecycle.purge_expired(created + timedelta(days=365), db_path=self.db)
        self.assertEqual(self.rows(), [])

    def test_repeated_message_does_not_duplicate_or_extend_setting(self):
        self.save()
        before = lifecycle.list_live_sources(101, datetime.now(timezone.utc), db_path=self.db)
        self.assertEqual(self.save()["saved_communication_preferences"], 0)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(lifecycle.list_live_sources(101, datetime.now(timezone.utc), db_path=self.db), before)

    def test_delayed_old_preference_cannot_replace_newer_setting(self):
        self.save("Мне удобнее подробные ответы", 5080)
        self.save(message_id=5070, source_created_at=datetime.now(timezone.utc) - timedelta(days=1))
        self.assertEqual(self.profile()["response_length"], "detailed")

    def test_storage_refuses_forged_style_even_with_valid_source(self):
        self.save()
        with self.assertRaises(PermissionError):
            storage.save_communication_preference(101, "telegram:private:101:5000", "detailed",
                                                  datetime.now(timezone.utc), db_path=self.db)
        with self.assertRaises(PermissionError):
            storage.save_communication_preference(202, "telegram:private:101:5000", "brief",
                                                  datetime.now(timezone.utc), db_path=self.db)

    def test_preference_can_share_source_with_normal_extracted_facts(self):
        self.bot.model_result = [dict(content="Предпочитает краткие ответы", kind="preference", confidence=0.99)]
        diagnostic = {}
        self.assertEqual(self.bot.save(101, 5090, "Мне удобнее короткие ответы", diagnostics=diagnostic), 1)
        self.assertEqual(diagnostic["saved_communication_preferences"], 1)
        self.assertEqual(len(storage.list_memories(101, db_path=self.db)), 1)
        self.assertEqual(self.profile()["source"], "telegram:private:101:5090")

    def test_groq_path_saves_preference_without_extra_ai_request(self):
        self.save(provider="Groq")
        self.assertEqual((self.bot.calls, self.bot.groq_calls), (0, 1))
        self.assertIn("краткие ответы", self.bot.read(101, "Растения"))

    def test_reading_legacy_database_does_not_create_personalization_tables(self):
        path = Path(self.bot.tmp.name) / "legacy.db"
        storage.initialize_database(path)
        with sqlite3.connect(path) as db:
            before = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        self.assertIsNone(storage.get_communication_preference(101, set(), datetime.now(timezone.utc), db_path=path))
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), before)

    def test_same_second_messages_follow_numeric_telegram_order(self):
        created = datetime.now(timezone.utc) - timedelta(minutes=1)
        self.save(message_id=9, source_created_at=created)
        self.save("Мне удобнее подробные ответы", 10, source_created_at=created)
        self.assertEqual(self.profile()["response_length"], "detailed")

    def test_forgetting_style_fact_removes_only_its_derived_setting(self):
        ids = []
        for mid, text, content in ((5100, "Мне удобнее короткие ответы", "Предпочитает краткие ответы"),
                                   (5101, "Мне удобнее подробные ответы", "Предпочитает подробные ответы")):
            self.bot.model_result = [dict(content=content, kind="preference", confidence=0.99)]
            self.assertEqual(self.bot.save(101, mid, text), 1)
            ids.append(next(row["id"] for row in storage.list_memories(101, db_path=self.db)
                            if row["content"] == content))
        self.bot.command("forget_command", [str(ids[0])], message_id=5102)
        self.assertEqual(self.profile()["response_length"], "detailed")
        self.bot.command("forget_command", [str(ids[1])], message_id=5103)
        self.assertIsNone(self.profile())
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.bot.read(101, "Расскажи о растениях"), "")

    def test_full_seven_file_snapshot_keeps_preferences_and_their_sources(self):
        import test_independent_backup as backup_fixtures
        from memory import cleanup_worker
        fixture = backup_fixtures.Dataset()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        now = backup_fixtures.NOW
        source = "telegram:private:101:preference"
        lifecycle.register_message(101, source, "Мне удобнее короткие ответы", now, db_path=fixture.db)
        storage.save_communication_preference(101, source, "brief", now, db_path=fixture.db)
        stage, result = fixture.capture("personalization-backup")
        self.assertEqual(result["verified_files"], 7)
        restored = storage.get_communication_preference(101, {source}, now, db_path=stage / "memory.db")
        self.assertEqual(restored, {"source": source, "response_length": "brief"})
        self.assertEqual({p.name for p in stage.iterdir()},
                         {*cleanup_worker.FULL_BACKUP_FILES, cleanup_worker.FULL_BACKUP_MANIFEST})


if __name__ == "__main__":
    unittest.main()
