"""End-to-end checks of the actual bot.py semantic extraction function."""
import ast
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memory import coordinator, storage, lifecycle
from memory.coordinator import AccessPolicy
from memory.extraction import has_statement_for_extraction


class SemanticMemoryWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "memory.db"
        self.consented = True
        self.gemini_calls = 0
        storage.initialize_database(self.db)
        lifecycle.initialize_lifecycle(self.db)
        lifecycle.set_memory_consent(101, True, datetime.now(timezone.utc), db_path=self.db)
        source = (Path(__file__).resolve().parents[2] / "bot.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        fn = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "remember_dialog_facts"
        )
        self.ns = {
            "GEMINI_API_KEY": "test-key",
            "OWNER_ID": 101,
            "TEST_USER_IDS": set(),
            "GEMINI_MODEL": "test-model",
            "GEMINI_TIMEOUT": 2,
            "json": json,
            "datetime": datetime,
            "timezone": timezone,
            "timedelta": timedelta,
            "AccessPolicy": AccessPolicy,
            "has_statement_for_extraction": has_statement_for_extraction,
            "save_dialog_candidates": lambda *args, **kw: coordinator.save_dialog_candidates(
                *args, db_path=self.db, **kw),
            "personal_storage": SimpleNamespace(
                initialize_database=lambda: storage.initialize_database(self.db),
                add_memory=lambda u, c, k, s: storage.add_memory(
                    u, c, k, s, db_path=self.db
                ),
            ),
            "personal_lifecycle": SimpleNamespace(
                initialize_lifecycle=lambda: lifecycle.initialize_lifecycle(self.db),
                has_memory_consent=lambda user: user == 101 and self.consented,
                list_live_sources=lambda user, now: lifecycle.list_live_sources(
                    user, now, db_path=self.db
                ),
                delete_message=lambda user, m: lifecycle.delete_message(
                    user, m, db_path=self.db
                ),
                register_message=lambda u, m, c, t: lifecycle.register_message(
                    u, m, c, t, db_path=self.db
                ),
                list_live_memories=lambda u, t: lifecycle.list_live_memories(
                    u, t, db_path=self.db
                ),
                link_memory_sources=lambda u, m, sources, t:
                    lifecycle.link_memory_sources(
                        u, m, sources, t, db_path=self.db
                    ),
            ),
        }
        import urllib.request
        self.ns["urllib"] = SimpleNamespace(request=SimpleNamespace(
            Request=urllib.request.Request,
            urlopen=self.fake_urlopen,
        ))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "bot.py", "exec"),
             self.ns)

    def fake_urlopen(self, request, timeout):
        self.gemini_calls += 1
        self.assertEqual(request.get_header("X-goog-api-key"), "test-key")
        response = {
            "candidates": [{
                "content": {"parts": [{
                    "text": json.dumps(self.model_result, ensure_ascii=False)
                }]}
            }]
        }
        return io.BytesIO(json.dumps(response).encode("utf-8"))

    def test_extract_store_read_and_deduplicate_same_message(self):
        self.model_result = [
            {"content": "Пользователь изучает биологию",
             "kind": "fact", "confidence": 0.98}
        ]
        fn = self.ns["remember_dialog_facts"]
        self.assertEqual(fn(101, 7, "Я изучаю биологию"), 1)
        self.assertEqual(fn(101, 7, "Я изучаю биологию"), 0)
        self.assertEqual(len(lifecycle.list_live_memories(
            101, datetime.now(timezone.utc), db_path=self.db
        )), 1)
        self.assertEqual(lifecycle.list_live_memories(
            202, datetime.now(timezone.utc), db_path=self.db
        ), [])

    def test_consent_missing_skips_gemini_and_storage(self):
        self.consented = False
        self.assertEqual(
            self.ns["remember_dialog_facts"](101, 10, "Я изучаю биологию"), 0
        )
        self.assertEqual(self.gemini_calls, 0)
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])

    def test_revoke_during_gemini_skips_all_writes(self):
        self.model_result = [
            {"content": "Пользователь изучает биологию",
             "kind": "fact", "confidence": 0.98}
        ]
        orig = self.ns["urllib"].request.urlopen

        def revoke_on_response(request, timeout):
            response = orig(request, timeout)
            self.consented = False
            return response

        self.ns["urllib"].request.urlopen = revoke_on_response
        self.assertEqual(
            self.ns["remember_dialog_facts"](101, 11, "Я изучаю биологию"), 0
        )
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])
        self.assertEqual(lifecycle.list_live_sources(
            101, datetime.now(timezone.utc), db_path=self.db
        ), [])

    def test_unverified_source_cannot_be_saved(self):
        self.model_result = [
            {"content": "Пользователь изучает биологию",
             "kind": "fact", "confidence": 0.98}
        ]
        with patch.object(lifecycle, "list_live_sources", return_value=[]), self.assertRaises(PermissionError):
            self.ns["remember_dialog_facts"](101, 12, "Я изучаю биологию")
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])

    def test_existing_source_is_not_reregistered(self):
        self.model_result = [
            {"content": "Пользователь изучает биологию",
             "kind": "fact", "confidence": 0.98}
        ]
        fn = self.ns["remember_dialog_facts"]
        self.assertEqual(fn(101, 13, "Я изучаю биологию"), 1)
        self.assertEqual(fn(101, 13, "Я изучаю биологию"), 0)
        self.assertEqual(len(lifecycle.list_live_sources(
            101, datetime.now(timezone.utc), db_path=self.db
        )), 1)

    def test_no_facts_writes_nothing(self):
        self.model_result = []
        self.assertEqual(
            self.ns["remember_dialog_facts"](101, 8, "Сколько времени?"), 0
        )
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])

    def test_low_confidence_is_not_saved(self):
        self.model_result = [
            {"content": "Пользователь любит чай",
             "kind": "preference", "confidence": 0.5}
        ]
        self.assertEqual(
            self.ns["remember_dialog_facts"](101, 9, "Я люблю чай"), 0
        )
        self.assertEqual(storage.list_memories(101, db_path=self.db), [])


if __name__ == "__main__":
    unittest.main()
