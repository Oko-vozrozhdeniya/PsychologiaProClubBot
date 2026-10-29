"""Confirmed correction and historical changes of existing long-term memory."""
import ast
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory import lifecycle, storage
from memory.coordinator import AccessPolicy, SourcePermission, prepare_correction

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)


class MemoryRevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "memory.db"
        lifecycle.initialize_lifecycle(self.db)
        lifecycle.set_memory_consent(101, True, NOW, db_path=self.db)
        self.source = "telegram:private:101:11"
        lifecycle.register_message(101, self.source, "Я работаю учителем", NOW, db_path=self.db)
        self.mid = storage.add_memory(
            101, "Работает учителем", "fact", self.source, db_path=self.db,
        )
        lifecycle.link_memory_sources(101, self.mid, [self.source], NOW, db_path=self.db)

    def revise(self, new="Сейчас работает психологом", message_id=12,
               kind="change", old="Работает учителем", source=None, user=101):
        old_source = self.source if source is None else source
        new_source = f"telegram:private:{user}:{message_id}"
        decision = prepare_correction(
            101, old, new, old_source,
            AccessPolicy(101, True, True, True, (NOW + timedelta(days=1)).isoformat()),
            [SourcePermission(old_source, (NOW + timedelta(days=364)).isoformat())],
            NOW.isoformat(), explicitly_corrected=True,
        )
        self.assertEqual(decision.action, "update")
        return lifecycle.revise_memory(
            user, self.mid, old, old_source, new, new_source,
            "Раньше работал учителем, теперь психологом", NOW, kind,
            db_path=self.db,
        )

    def test_temporal_change_preserves_history_but_only_current_memory_active(self):
        self.assertTrue(self.revise())
        current = lifecycle.list_live_memories(101, NOW, db_path=self.db)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["content"], "Сейчас работает психологом")
        self.assertEqual(current[0]["source"], "telegram:private:101:12")
        hist = lifecycle.list_memory_revisions(101, self.mid, NOW, db_path=self.db)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["previous_content"], "Работает учителем")
        self.assertEqual(hist[0]["change_kind"], "change")

    def test_explicit_correction_removes_wrong_fact_from_active_memory(self):
        self.assertTrue(self.revise("Работает врачом", kind="correction"))
        active = lifecycle.list_live_memories(101, NOW, db_path=self.db)
        self.assertEqual([m["content"] for m in active], ["Работает врачом"])
        self.assertEqual(lifecycle.list_memory_revisions(
            101, self.mid, NOW, db_path=self.db
        )[0]["change_kind"], "correction")

    def test_chained_changes_keep_order(self):
        self.revise(message_id=12)
        lifecycle.revise_memory(
            101, self.mid, "Сейчас работает психологом", "telegram:private:101:12",
            "Сейчас работает консультантом", "telegram:private:101:13",
            "Сейчас я консультант", NOW + timedelta(seconds=3), "change",
            db_path=self.db,
        )
        h = lifecycle.list_memory_revisions(
            101, self.mid, NOW + timedelta(seconds=3), db_path=self.db
        )
        self.assertEqual(len(h), 2)
        self.assertEqual(h[1]["previous_content"], "Сейчас работает психологом")

    def test_replay_cannot_change_version_or_extend_expiry(self):
        self.revise()
        with self.assertRaises(ValueError):
            self.revise()
        self.assertEqual(len(lifecycle.list_memory_revisions(
            101, self.mid, NOW, db_path=self.db
        )), 1)

    def test_stale_version_rollback(self):
        with self.assertRaises(ValueError):
            self.revise(old="Работает космонавтом")
        self.assertEqual(len(lifecycle.list_memory_revisions(
            101, self.mid, NOW, db_path=self.db
        )), 0)
        self.assertEqual(storage.get_memory(101, self.mid, db_path=self.db)["content"],
                         "Работает учителем")

    def test_revoked_consent_rejects_revision(self):
        lifecycle.set_memory_consent(101, False, NOW, db_path=self.db)
        with self.assertRaises(PermissionError):
            self.revise()
        self.assertEqual(storage.get_memory(101, self.mid, db_path=self.db)["content"],
                         "Работает учителем")

    def test_cross_user_revision_rejected(self):
        with self.assertRaises((LookupError, PermissionError)):
            self.revise(user=202)
        self.assertEqual(lifecycle.list_live_memories(202, NOW, db_path=self.db), [])

    def test_invalid_kind_rejected_before_write(self):
        with self.assertRaises(ValueError):
            self.revise(kind="guess")
        self.assertEqual(len(lifecycle.list_memory_revisions(
            101, self.mid, NOW, db_path=self.db
        )), 0)

    def test_expired_memory_cannot_be_revised(self):
        with self.assertRaises(PermissionError):
            lifecycle.revise_memory(
                101, self.mid, "Работает учителем", self.source,
                "Сейчас психолог", "telegram:private:101:14", "новые сведения",
                NOW + timedelta(days=366), "change", db_path=self.db,
            )

    def test_restart_retains_current_and_history(self):
        self.revise()
        lifecycle.initialize_lifecycle(self.db)
        self.assertEqual(storage.get_memory(
            101, self.mid, db_path=self.db
        )["content"], "Сейчас работает психологом")
        self.assertEqual(len(lifecycle.list_memory_revisions(
            101, self.mid, NOW, db_path=self.db
        )), 1)

    def test_forget_removes_active_and_previous_versions(self):
        self.revise()
        self.assertTrue(storage.delete_memory(101, self.mid, db_path=self.db))
        self.assertEqual(lifecycle.list_memory_revisions(
            101, self.mid, NOW, db_path=self.db
        ), [])
        with storage._connection(self.db) as db:
            self.assertEqual(db.execute(
                "SELECT COUNT(*) FROM memory_revisions WHERE memory_id=?",
                (self.mid,),
            ).fetchone()[0], 0)

    def test_deleting_previous_source_clears_history_not_current(self):
        self.revise()
        lifecycle.delete_message(101, self.source, db_path=self.db)
        self.assertEqual(lifecycle.list_memory_revisions(
            101, self.mid, NOW, db_path=self.db
        ), [])
        self.assertEqual(lifecycle.list_live_memories(
            101, NOW, db_path=self.db
        )[0]["content"], "Сейчас работает психологом")

    def test_deleting_new_source_cascades_active_and_history(self):
        self.revise()
        lifecycle.delete_message(
            101, "telegram:private:101:12", db_path=self.db,
        )
        self.assertEqual(lifecycle.list_live_memories(
            101, NOW, db_path=self.db
        ), [])
        with storage._connection(self.db) as db:
            self.assertEqual(db.execute(
                "SELECT COUNT(*) FROM memory_revisions WHERE memory_id=?",
                (self.mid,),
            ).fetchone()[0], 0)

    def test_expiration_removes_revisions(self):
        self.revise()
        lifecycle.purge_expired(NOW + timedelta(days=366), db_path=self.db)
        with storage._connection(self.db) as db:
            self.assertEqual(db.execute(
                "SELECT COUNT(*) FROM memory_revisions WHERE memory_id=?",
                (self.mid,),
            ).fetchone()[0], 0)

    def test_history_is_user_scoped(self):
        self.revise()
        self.assertEqual(lifecycle.list_memory_revisions(
            202, self.mid, NOW, db_path=self.db
        ), [])

    def test_commands_registered_with_confirmation_and_history(self):
        content = (Path(__file__).resolve().parents[2] / "bot.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(content)
        funcs = {n.name: n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in ("correct_command", "change_command", "history_command", "revise_command"):
            self.assertIn(name, funcs)
        self.assertTrue(any(
            isinstance(n, ast.Name) and n.id == "prepare_correction"
            for n in ast.walk(funcs["revise_command"])
        ))
        commands = {
            n.args[0].value for n in ast.walk(funcs["main"])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "CommandHandler" and n.args
            and isinstance(n.args[0], ast.Constant)
        }
        self.assertTrue({"correct", "change", "history"}.issubset(commands))


if __name__ == "__main__":
    unittest.main()
