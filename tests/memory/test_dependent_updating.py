"""Directed conclusions must lose eligibility when any premise changes."""
from datetime import timedelta
import unittest
from test_persistent_relationships import PersistentRelationships, NOW, policy
from memory import coordinator, lifecycle, storage
from memory.updating import blocked_conclusions


class DependentUpdating(unittest.TestCase):
    # Reuse source-backed fixture helpers without inheriting unrelated tests.
    setUp = PersistentRelationships.setUp
    message = PersistentRelationships.message
    fact = PersistentRelationships.fact
    snapshot = PersistentRelationships.snapshot
    permissions = PersistentRelationships.permissions
    save = PersistentRelationships.save
    revise = PersistentRelationships.revise
    def dependency(self, left, right, name='dependency'):
        witness=self.message(name,'Этот вывод основан на указанном факте.')
        return coordinator.remember_dependency(101,self.snapshot(left),self.snapshot(right),
            witness,policy(),self.permissions(),NOW.isoformat(),
            explicitly_confirmed=True,db_path=self.db)

    def blocked(self, now=NOW):
        with storage._connection(self.db) as db:
            return blocked_conclusions(db,101,now.isoformat())

    def conclusion(self, name='conclusion'):
        return self.fact('Учебный вывод '+name,name,kind='hypothesis')

    def test_fact_change_blocks_transitive_conclusions_preserves_history(self):
        b=self.conclusion('b2');c=self.conclusion('c2')
        self.dependency(self.a,b,'ab');self.dependency(b,c,'bc')
        self.assertEqual(self.blocked(),set())
        self.revise(self.a,'Цель отменена')
        self.assertEqual(self.blocked(),{b,c})
        self.assertIsNotNone(self.snapshot(b))
        self.assertEqual(len(lifecycle.list_memory_revisions(101,self.a,NOW,db_path=self.db)),1)
        ids={r['id'] for r in lifecycle.list_live_memories(101,NOW,db_path=self.db)}
        self.assertFalse({b,c}&ids)
        context=coordinator.read_context(101,'Учебный',policy(),self.permissions(),NOW.isoformat(),db_path=self.db)
        self.assertFalse({b,c}&{r['id'] for r in context['memories']})

    def test_same_topic_does_not_create_dependency(self):
        self.save();self.revise(self.a,'Новая цель')
        self.assertNotIn(self.b,self.blocked())

    def test_cycle_and_fact_promotion_are_rejected(self):
        b=self.conclusion('b2');c=self.conclusion('c2')
        self.dependency(b,c,'bc')
        with self.assertRaises(ValueError):self.dependency(c,b,'cb')
        with self.assertRaises(PermissionError):self.dependency(b,self.a,'ba')

    def test_missing_confirmation_and_cross_user_rejected(self):
        b=self.conclusion()
        with self.assertRaises(PermissionError):
            coordinator.remember_dependency(101,self.snapshot(self.a),self.snapshot(b),self.witness,
                policy(),self.permissions(),NOW.isoformat(),db_path=self.db)
        wrong=dict(self.snapshot(b),user_id=202)
        with self.assertRaises((PermissionError,ValueError)):
            coordinator.remember_dependency(101,self.snapshot(self.a),wrong,self.witness,
                policy(),self.permissions(),NOW.isoformat(),db_path=self.db,explicitly_confirmed=True)

    def test_deleted_premise_never_restores_conclusion(self):
        b=self.conclusion();self.dependency(self.a,b)
        storage.delete_memory(101,self.a,db_path=self.db)
        self.assertIn(b,self.blocked())
        self.assertIsNotNone(self.snapshot(b))

    def test_expired_premise_blocks_before_cleanup(self):
        b=self.conclusion();self.dependency(self.a,b)
        later=NOW+timedelta(days=365)-timedelta(minutes=7)
        self.assertIn(b,self.blocked(later))

    def test_unpermitted_premise_blocks_context_even_if_conclusion_allowed(self):
        b=self.conclusion();self.dependency(self.a,b)
        allowed=[p for p in self.permissions() if p.source!=self.snapshot(self.a)['source']]
        result=coordinator.read_context(101,'Учебный',policy(),allowed,NOW.isoformat(),db_path=self.db)
        self.assertNotIn(b,{r['id'] for r in result['memories']})

    def test_replay_and_identical_revision_do_not_revive(self):
        b=self.conclusion();self.dependency(self.a,b)
        self.revise(self.a,'Другая цель')
        with self.assertRaises(ValueError):self.dependency(self.a,b)
        self.assertIn(b,self.blocked())

    def test_noop_update_does_not_block(self):
        b=self.conclusion();self.dependency(self.a,b)
        storage.update_memory(101,self.a,self.snapshot(self.a)['content'],db_path=self.db)
        self.assertEqual(self.blocked(),set())

    def test_expired_witness_cleanup_keeps_review_without_provenance(self):
        b=self.conclusion();self.dependency(self.a,b)
        lifecycle.delete_message(101,'telegram:private:101:dependency',db_path=self.db)
        self.assertIn(b,self.blocked())
        with storage._connection(self.db) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM memory_dependencies').fetchone()[0],0)
        storage.delete_memory(101,b,db_path=self.db)
        with storage._connection(self.db) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM memory_conclusion_reviews').fetchone()[0],0)

    def test_any_changed_premise_invalidates_conclusion(self):
        b=self.conclusion();self.dependency(self.a,b,'first');self.dependency(self.b,b,'second')
        self.revise(self.b,'Другой эпизод')
        self.assertIn(b,self.blocked())

    def test_stale_snapshot_rolls_back_dependency(self):
        b=self.conclusion();old=self.snapshot(self.a)
        self.revise(self.a,'Другая цель')
        with self.assertRaises(ValueError):
            coordinator.remember_dependency(101,old,self.snapshot(b),self.witness,policy(),
                self.permissions(),NOW.isoformat(),db_path=self.db,explicitly_confirmed=True)
        with storage._connection(self.db) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM memory_dependencies').fetchone()[0],0)

    def test_ordinary_dialogue_dependency_and_later_correction(self):
        premise=self.snapshot(self.a)['content']
        conclusion='Мне нужен план занятий'
        message=f'{premise}; мой вывод: {conclusion}.'
        payload={'memories':[dict(content=conclusion,kind='hypothesis',confidence=0.99)],
                 'relationships':[dict(source_content=premise,target_content=conclusion,
                    relation_type='depends_on',confidence=0.99,evidence_quote=message)]}
        saved=coordinator.save_dialog_candidates(101,'telegram:private:101:dialog',message,
            payload,[self.snapshot(self.a)],policy(),NOW.isoformat(),db_path=self.db)
        self.assertEqual(saved,1)
        target=next(row for row in lifecycle.list_live_memories(101,NOW,db_path=self.db)
                    if row['content']==conclusion)
        self.revise(self.a,'Цель отменена')
        self.assertIn(target['id'],self.blocked())

    def test_dependency_quote_cannot_invent_or_promote_conclusions(self):
        from memory.extraction import extract_relationship_candidates
        for quote in ('Я думаю об учёбе.', 'Мой вывод: отдых полезен.'):
            link=dict(source_content='Цель: рисование',target_content='Мне нужен план занятий',
                      relation_type='depends_on',confidence=1,evidence_quote=quote)
            self.assertEqual(extract_relationship_candidates([link],quote),[])

del PersistentRelationships
