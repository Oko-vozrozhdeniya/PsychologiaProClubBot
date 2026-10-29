"""Verify the real bot save and read functions share the coordinator boundary.

No Telegram credentials and no external Gemini call are needed for this test.
"""
import ast
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from memory import storage, lifecycle
from memory.coordinator import AccessPolicy, SourcePermission, read_context, prepare_candidates
from memory.extraction import extract_candidates
from memory.prompt_context import build_memory_context


def load_function(name, namespace):
    source = (Path(__file__).resolve().parents[2] / "bot.py").read_text(encoding="utf-8")
    node = next(
        item for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef) and item.name == name
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
            add_memory=lambda uid, c, k, src: storage.add_memory(uid, c, k, src, db_path=self.db),
        )
        self.personal_lifecycle = SimpleNamespace(
            initialize_lifecycle=lambda: lifecycle.initialize_lifecycle(self.db),
            has_memory_consent=lambda uid: lifecycle.has_memory_consent(uid, db_path=self.db),
            list_live_sources=lambda uid, t: lifecycle.list_live_sources(uid, t, db_path=self.db),
            list_live_memories=lambda uid, t: lifecycle.list_live_memories(uid, t, db_path=self.db),
            register_message=lambda uid, src, c, t: lifecycle.register_message(
                uid, src, c, t, db_path=self.db),
            link_memory_sources=lambda uid, mid, sources, t: lifecycle.link_memory_sources(
                uid, mid, sources, t, db_path=self.db),
            delete_message=lambda uid, src: lifecycle.delete_message(uid, src, db_path=self.db),
            purge_expired=lambda t: lifecycle.purge_expired(t, db_path=self.db),
        )
        self.calls = 0

        def fake_urlopen(req, timeout):
            self.calls += 1
            content = [
                {"content": "Пользователь изучает биологию",
                 "kind": "fact", "confidence": 0.98}
            ]
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
            "SourcePermission": SourcePermission, "prepare_candidates": prepare_candidates,
            "extract_candidates": extract_candidates,
            "build_memory_context": build_memory_context,
            "read_context": lambda uid, q, pol, perms, now, **kw: read_context(
                uid, q, pol, perms, now, db_path=self.db, **kw),
            "personal_storage": self.personal_storage,
            "personal_lifecycle": self.personal_lifecycle,
            "urllib": SimpleNamespace(request=SimpleNamespace(
                Request=urllib.request.Request, urlopen=fake_urlopen)),
        }
        self.save = load_function("remember_dialog_facts", ns)
        self.read = load_function("read_explicit_memory_prompt", ns)

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


if __name__ == "__main__":
    unittest.main()
