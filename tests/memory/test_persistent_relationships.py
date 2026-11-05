"""Source-backed edges through the coordinator; no network or live bot data."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from memory import coordinator, lifecycle, relationships, storage

NOW = datetime(2026, 11, 6, 0, tzinfo=timezone.utc)


def policy(user=101):
    return coordinator.AccessPolicy(user, True, True, True, (NOW + timedelta(days=1)).isoformat())


class PersistentRelationships(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.db = Path(self.temporary.name) / "memory.db"
        lifecycle.initialize_lifecycle(self.db)
        lifecycle.set_memory_consent(101, True, NOW, db_path=self.db)
        self.a = self.fact("Цель: рисование", "a")
        self.b = self.fact("Событие: рисование", "b")
        self.witness = self.message("link", "Эта цель и это событие связаны темой рисования.")
        self.relation = relationships.make_relationship(101, self.a, self.b,
            "same_topic", "confirmed", self.witness, "Тема — рисование")

    def message(self, name, text, user=101, created=None):
        name = f"telegram:private:{user}:{name}"
        lifecycle.register_message(user, name, text,
            created or NOW - timedelta(minutes=5), db_path=self.db)
        return name

    def fact(self, text, name, user=101, created=None, kind="fact"):
        created = created or NOW - timedelta(minutes=10)
        source = self.message(name, text, user, created)
        return lifecycle.add_memory(user, text, kind, source, NOW, db_path=self.db)

    def snapshot(self, memory_id):
        return storage.get_memory(101, memory_id, db_path=self.db)

    def permissions(self, user=101, now=NOW):
        return [coordinator.SourcePermission(**row)
                for row in lifecycle.list_live_sources(user, now, db_path=self.db)]

    def save(self, relation=None, left=None, right=None, **kwargs):
        return coordinator.remember_relationship(101, relation or self.relation,
            left or self.snapshot(self.a), right or self.snapshot(self.b),
            policy(), self.permissions(), NOW.isoformat(), db_path=self.db,
            explicitly_confirmed=kwargs.pop("explicitly_confirmed", True), **kwargs)

    def read(self, sources=None, ids=None, now=NOW):
        return relationships.list_live_relationships(101, ids or [self.a,self.b],
            sources if sources is not None else {p.source for p in self.permissions(now=now)},
            now, db_path=self.db)

    def count(self):
        with storage._connection(self.db) as db:
            return db.execute("SELECT count(*) FROM memory_relationships").fetchone()[0]

    def revise(self, memory_id, content, name="revision"):
        old = self.snapshot(memory_id)
        return coordinator.revise_memory(101, memory_id, old["content"],old["source"],
            content,f"telegram:private:101:{name}",content,policy(),NOW.isoformat(),
            "correction", explicitly_corrected=True,db_path=self.db)

    def test_confirmed_relationship_survives_a_new_database_connection(self):
        identity = self.save()
        rows = self.read()
        self.assertEqual([row["id"] for row in rows], [identity])
        self.assertEqual(rows[0]["source"], self.witness)
        self.assertEqual(rows[0]["evidence"], "confirmed")
        self.assertNotIn("source_version", rows[0])
        with sqlite3.connect(self.db) as db:
            self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_identical_retry_keeps_id_count_and_original_date(self):
        identity = self.save()
        self.assertEqual(self.save(), identity)
        self.assertEqual(self.count(), 1)
        self.assertEqual(self.read()[0]["created_at"], NOW.isoformat())

    def test_confirmed_edge_requires_explicit_confirmation(self):
        with self.assertRaises(PermissionError):
            self.save(explicitly_confirmed=False)
        self.assertEqual(self.count(), 0)

    def test_hypothesis_is_never_implicitly_promoted(self):
        self.save(replace(self.relation, evidence="hypothesis"), explicitly_confirmed=False)
        self.assertEqual(self.read()[0]["evidence"], "hypothesis")

    def test_hypothetical_endpoint_cannot_be_presented_as_confirmed(self):
        with storage._connection(self.db) as db:
            db.execute("UPDATE memories SET kind='hypothesis' WHERE id=?", (self.a,))
        with self.assertRaises(PermissionError):
            self.save()
        self.assertEqual(self.count(), 0)

    def test_stored_consent_is_rechecked_even_when_policy_still_says_yes(self):
        self.save()
        lifecycle.set_memory_consent(101, False, NOW, db_path=self.db)
        self.assertEqual(self.read(), [])
        with self.assertRaises(PermissionError):
            self.save()

    def test_cross_user_relationship_and_endpoint_are_rejected(self):
        lifecycle.set_memory_consent(202, True, NOW, db_path=self.db)
        foreign = self.fact("Чужая цель", "foreign", user=202)
        with self.assertRaises(PermissionError):
            self.save(replace(self.relation, user_id=202))
        with self.assertRaises(PermissionError):
            self.save(replace(self.relation, target_memory_id=foreign),
                right=storage.get_memory(202, foreign, db_path=self.db))
        self.assertEqual(self.count(), 0)

    def test_wrong_policy_cannot_write_an_edge(self):
        with self.assertRaises(PermissionError):
            coordinator.remember_relationship(101, self.relation,self.snapshot(self.a),self.snapshot(self.b),
                policy(202), self.permissions(), NOW.isoformat(), explicitly_confirmed=True,db_path=self.db)

    def test_forged_permission_does_not_replace_registered_provenance(self):
        source = "telegram:private:101:not-registered"
        permissions = self.permissions() + [coordinator.SourcePermission(source, (NOW+timedelta(days=1)).isoformat())]
        with self.assertRaises(PermissionError):
            coordinator.remember_relationship(101, replace(self.relation,source=source),
                self.snapshot(self.a),self.snapshot(self.b),policy(),permissions,NOW.isoformat(),
                explicitly_confirmed=True,db_path=self.db)
        self.assertEqual(self.count(), 0)

    def test_unverified_witness_permission_is_rejected(self):
        permissions = [replace(p,verified=False) if p.source==self.witness else p for p in self.permissions()]
        with self.assertRaises(PermissionError):
            coordinator.remember_relationship(101,self.relation,self.snapshot(self.a),self.snapshot(self.b),
                policy(),permissions,NOW.isoformat(),explicitly_confirmed=True,db_path=self.db)

    def test_stale_endpoint_snapshot_cannot_create_an_edge(self):
        old = self.snapshot(self.a)
        storage.update_memory(101,self.a,"Другая цель",db_path=self.db)
        with self.assertRaises(ValueError):
            self.save(left=old)
        self.assertEqual(self.count(), 0)

    def test_expired_endpoint_cannot_be_used_even_with_fresh_witness(self):
        with storage._connection(self.db) as db:
            db.execute("UPDATE private_messages SET expires_at=? WHERE message_id=?",
                       (NOW.isoformat(),self.snapshot(self.a)["source"]))
        with self.assertRaises(PermissionError):
            self.save()

    def test_expired_witness_is_hidden_before_cleanup(self):
        self.save()
        with storage._connection(self.db) as db:
            db.execute("UPDATE private_messages SET expires_at=? WHERE message_id=?", (NOW.isoformat(),self.witness))
        self.assertEqual(self.read(), [])

    def test_unknown_endpoint_ids_do_not_expand_or_expose_the_context(self):
        self.save()
        self.assertEqual(self.read(ids=[999999,999998]), [])
        self.assertEqual(self.read(ids=[self.a]), [])

    def test_witness_and_both_endpoint_sources_must_be_permitted(self):
        self.save()
        all_sources = {p.source for p in self.permissions()}
        for source in (self.witness,self.snapshot(self.a)["source"],self.snapshot(self.b)["source"]):
            self.assertEqual(self.read(all_sources-{source}), [])

    def test_other_user_cannot_read_edges_by_guessing_owned_ids(self):
        self.save()
        lifecycle.set_memory_consent(202,True,NOW,db_path=self.db)
        self.assertEqual(relationships.list_live_relationships(202,[self.a,self.b],
            {self.witness,self.snapshot(self.a)["source"],self.snapshot(self.b)["source"]},NOW,db_path=self.db), [])

    def test_revision_hides_old_edge_without_changing_original_message(self):
        self.save()
        original = self.snapshot(self.a)["source"]
        self.assertTrue(self.revise(self.a,"Цель: лепка"))
        self.assertEqual(self.read(), [])
        with storage._connection(self.db) as db:
            self.assertEqual(db.execute("SELECT status FROM memory_relationships").fetchone()[0], "review_required")
            self.assertEqual(db.execute("SELECT content FROM private_messages WHERE message_id=?",(original,)).fetchone()[0], "Цель: рисование")

    def test_returning_to_old_wording_does_not_reactivate_old_edge(self):
        self.save()
        storage.update_memory(101,self.a,"Цель: лепка",db_path=self.db)
        storage.update_memory(101,self.a,"Цель: рисование",db_path=self.db)
        self.assertEqual(self.read(), [])
        with self.assertRaises(ValueError):
            self.save()

    def test_repetition_of_unchanged_fact_preserves_valid_edge(self):
        self.save()
        source = self.message("repeat","Цель: рисование",created=NOW)
        self.assertTrue(lifecycle.confirm_memory(101,self.snapshot(self.a),source,NOW,db_path=self.db))
        self.assertEqual(len(self.read()), 1)

    def test_physical_fact_deletion_removes_its_edges_without_touching_other_fact(self):
        self.save()
        self.assertTrue(coordinator.forget_memory(101,self.a,policy(),NOW.isoformat(),db_path=self.db))
        self.assertEqual(self.count(), 0)
        self.assertIsNotNone(self.snapshot(self.b))

    def test_witness_deletion_removes_edge_even_when_both_facts_still_live(self):
        self.save()
        result = lifecycle.delete_message(101,self.witness,db_path=self.db)
        self.assertEqual(result["deleted_memories"], 0)
        self.assertEqual(self.count(), 0)
        self.assertIsNotNone(self.snapshot(self.a))
        self.assertIsNotNone(self.snapshot(self.b))

    def test_endpoint_source_deletion_removes_the_edge(self):
        self.save()
        lifecycle.delete_message(101,self.snapshot(self.a)["source"],db_path=self.db)
        self.assertEqual(self.count(), 0)

    def test_expiry_physically_removes_edge_while_independent_facts_remain(self):
        old = NOW-timedelta(days=364)
        witness = self.message("old-link","Связь была указана давно",created=old)
        self.save(replace(self.relation,source=witness))
        later = NOW+timedelta(days=1)
        counts = lifecycle.purge_expired(later,db_path=self.db)
        self.assertEqual(counts["deleted_memories"],0)
        self.assertEqual(self.count(),0)

    def test_merging_duplicate_endpoint_does_not_invent_a_retargeted_edge(self):
        duplicate = self.fact("Цель: рисование","duplicate")
        self.save(replace(self.relation,source_memory_id=duplicate),left=self.snapshot(duplicate))
        self.assertEqual(lifecycle.merge_memories(101,[self.snapshot(self.a),self.snapshot(duplicate)],NOW,db_path=self.db),1)
        self.assertEqual(self.count(),0)
        self.assertIsNotNone(self.snapshot(self.a))

    def test_description_change_needs_a_new_witness(self):
        self.save()
        with self.assertRaises(ValueError):
            self.save(replace(self.relation,description="Другой смысл связи"))
        self.assertEqual(self.read()[0]["description"],self.relation.description)

    def test_new_explicit_witness_can_confirm_updated_endpoints(self):
        self.save()
        self.revise(self.a,"Цель: лепка")
        source = self.message("new-link","Уточняю связь новой цели и старого события.",created=NOW)
        self.save(replace(self.relation,source=source,description="Уточнённая связь"))
        self.assertEqual([r["source"] for r in self.read()],[source])

    def test_coordinator_opt_in_returns_only_edges_between_ranked_facts(self):
        self.save()
        ordinary = coordinator.read_context(101,"рисование",policy(),self.permissions(),NOW.isoformat(),db_path=self.db)
        self.assertNotIn("relationships",ordinary)
        linked = coordinator.read_context(101,"рисование",policy(),self.permissions(),NOW.isoformat(),
            include_relationships=True,enforce_lifecycle=True,db_path=self.db)
        self.assertEqual(len(linked["relationships"]),1)
        limited = coordinator.read_context(101,"рисование",policy(),self.permissions(),NOW.isoformat(),
            include_relationships=True,limit=1,db_path=self.db)
        self.assertEqual(limited["relationships"],[])

    def test_corrected_edge_does_not_appear_in_coordinator_context(self):
        self.save()
        self.revise(self.a,"Цель: лепка")
        result=coordinator.read_context(101,"рисование лепка",policy(),self.permissions(),NOW.isoformat(),
            include_relationships=True,db_path=self.db)
        self.assertEqual(result["relationships"],[])

    def test_read_does_not_migrate_legacy_database(self):
        old=Path(self.temporary.name)/"legacy.db"
        storage.initialize_database(old)
        self.assertEqual(relationships.list_live_relationships(101,[1,2],{"source"},NOW,db_path=old),[])
        with storage._connection(old) as db:
            self.assertFalse(storage._relationship_table_exists(db))

    def test_oversized_descriptions_and_invalid_flags_are_rejected(self):
        for kwargs in ({"relation":replace(self.relation,description="x"*501)}, {"explicitly_confirmed":1}):
            with self.assertRaises(ValueError):
                self.save(**kwargs)
        self.assertEqual(self.count(),0)

    def test_corrupted_endpoint_version_is_not_exposed(self):
        self.save()
        with storage._connection(self.db) as db:
            db.execute("UPDATE memory_relationships SET source_version='wrong'")
        self.assertEqual(self.read(),[])

    def test_consent_withdrawal_between_coordinator_and_transaction_blocks_write(self):
        original=relationships.save_relationship
        def withdraw(*args,**kwargs):
            lifecycle.set_memory_consent(101,False,NOW,db_path=self.db)
            return original(*args,**kwargs)
        with patch.object(relationships,"save_relationship",side_effect=withdraw):
            with self.assertRaises(PermissionError):
                self.save()
        self.assertEqual(self.count(),0)

    def test_source_filter_is_applied_before_result_limit(self):
        self.save()
        newer=self.message("another-link","Подтверждение связи ещё одним сообщением.",created=NOW)
        identity=self.save(replace(self.relation,source=newer))
        permitted={p.source for p in self.permissions()}-{self.witness}
        rows=relationships.list_live_relationships(101,[self.a,self.b],permitted,NOW,limit=1,db_path=self.db)
        self.assertEqual([r["id"] for r in rows],[identity])


class RelationshipBackup(unittest.TestCase):
    def test_full_backup_keeps_relationship_table_and_current_edges(self):
        import test_independent_backup as fixtures
        from memory import cleanup_worker
        fixture=fixtures.Dataset()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        moment=fixtures.NOW
        text="Событие: второй факт"
        second="telegram:private:101:second"
        lifecycle.register_message(101,second,text,moment-timedelta(hours=1),db_path=fixture.db)
        mid=lifecycle.add_memory(101,text,"fact",second,moment,db_path=fixture.db)
        first=storage.get_memory(101,fixture.mid,db_path=fixture.db)
        target=storage.get_memory(101,mid,db_path=fixture.db)
        relation=relationships.make_relationship(101,fixture.mid,mid,"same_event","confirmed",second)
        identity=relationships.save_relationship(relation,first,target,moment,explicitly_confirmed=True,db_path=fixture.db)
        stage=fixture.root/"graph-backup"
        cleanup_worker.capture_full_dataset_quiescent(fixture.data,stage,moment)
        restored=relationships.list_live_relationships(101,[fixture.mid,mid],{first["source"],second},moment,db_path=stage/"memory.db")
        self.assertEqual([r["id"] for r in restored],[identity])
        self.assertEqual({p.name for p in stage.iterdir()}, {*cleanup_worker.FULL_BACKUP_FILES,cleanup_worker.FULL_BACKUP_MANIFEST})


if __name__ == "__main__":
    unittest.main()
