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

    def test_dependency_requires_real_statement_quote(self):
        from memory.extraction import extract_relationship_candidates
        for quote, message in (('Я готовлюсь.', 'Я отдыхаю.'),
                               ('Мне нужен план?', 'Мне нужен план?')):
            link=dict(source_memory_id=self.a,target_candidate_index=0,
                      relation_type='depends_on',confidence=1,evidence_quote=quote)
            self.assertEqual(extract_relationship_candidates([link],message),[])

    def save_natural(self, message, *, identity=None, shown=None, index=0):
        self.revise(self.a,'Пользователь готовится к экзамену по биологии',name='exam')
        conclusion='Для подготовки к экзамену нужен план повторения'
        payload={'memories':[dict(content=conclusion,kind='hypothesis',confidence=0.99)],
                 'relationships':[dict(source_memory_id=identity or self.a,
                    target_candidate_index=index,relation_type='depends_on',
                    confidence=0.99,evidence_quote=message)]}
        diag={}
        coordinator.save_dialog_candidates(101,'telegram:private:101:natural',message,
            payload,[self.snapshot(self.a)] if shown is None else shown,policy(),
            NOW.isoformat(),db_path=self.db,relationship_diagnostics=diag)
        return diag

    def test_natural_wording_can_reference_canonical_records_by_id(self):
        # This is an adapter integration test with a supplied model response,
        # not a claim that live Gemini/Groq understood these utterances.
        message='Раз у меня скоро проверка знаний, придётся расписать, что повторять.'
        diag=self.save_natural(message)
        self.assertEqual(diag['saved_relationships'],1)
        self.revise(self.a,'Подготовка больше не нужна')
        self.assertEqual(len(self.blocked()),1)

    def test_unshown_id_is_not_resolved_even_when_owned(self):
        self.assertEqual(self.save_natural('Из-за экзамена нужен график.',identity=self.b)
                         ['saved_relationships'],0)

    def test_candidate_index_is_scoped_to_accepted_current_response(self):
        self.assertEqual(self.save_natural('Из-за экзамена нужен график.',index=4)
                         ['saved_relationships'],0)

    def test_cross_user_id_is_not_resolved(self):
        lifecycle.set_memory_consent(202,True,NOW,db_path=self.db)
        other=self.fact('Другая цель','other',user=202)
        self.assertEqual(self.save_natural('Из-за экзамена нужен график.',identity=other)
                         ['saved_relationships'],0)

    def test_endpoint_reference_validation_rejects_ambiguous_or_bool(self):
        from memory.extraction import extract_relationship_candidates
        quote='Для подготовки мне нужен график.'
        for fields in ({'source_memory_id':True},
                       {'source_memory_id':self.a,'source_content':'Цель: рисование'},
                       {'source_candidate_index':-1}):
            link=dict(fields,target_candidate_index=0,relation_type='depends_on',
                      confidence=0.99,evidence_quote=quote)
            self.assertEqual(extract_relationship_candidates([link],quote),[])

    def test_reference_direction_remains_distinct(self):
        from memory.extraction import extract_relationship_candidates
        quote='Из первого следует второе.'
        links=[dict(source_memory_id=a,target_memory_id=b,relation_type='depends_on',
                    confidence=0.99,evidence_quote=quote) for a,b in ((1,2),(2,1))]
        self.assertEqual(len(extract_relationship_candidates(links,quote)),2)
    def test_paraphrased_quotes_do_not_require_canonical_endpoint_text(self):
        from memory.extraction import extract_relationship_candidates
        for message in ('Из-за предстоящей проверки придётся заниматься каждый вечер.',
                        'Чтобы сдать, надо заново пройти материал.',
                        'Раз я к нему готовлюсь, потребуется расписание.',
                        'Успешная сдача потребует регулярного повторения.'):
            with self.subTest(message=message):
                link=dict(source_memory_id=self.a,target_candidate_index=0,
                          relation_type='depends_on',confidence=0.99,evidence_quote=message)
                self.assertEqual(len(extract_relationship_candidates([link],message)),1)

    def test_natural_revision_requires_real_evidence_or_legacy_change_signal(self):
        row=self.snapshot(self.a)
        for quote in ('Выдуманная цитата', 'Зачем менять цель?', None):
            raw=dict(content='Цель отменена',kind='fact',confidence=0.99,
                     replaces_memory_id=self.a,replaces_content=row['content'],revision_kind='change')
            if quote is not None:raw['revision_evidence_quote']=quote
            saved=coordinator.save_dialog_candidates(101,'telegram:private:101:bad'+str(quote),
                'Передумал заниматься рисованием.',[raw],[row],policy(),NOW.isoformat(),db_path=self.db)
            self.assertEqual(saved,0)
            self.assertEqual(self.snapshot(self.a)['content'],row['content'])


del PersistentRelationships


class NaturalProviderPathTests(unittest.TestCase):
    """Real bot functions, supplied model outputs; no live AI understanding claim."""
    def provider_path(self, provider):
        import test_bot_memory_round_trip as harness
        from datetime import datetime, timezone
        b=harness.ActualMemoryRoundTripTests(methodName='runTest')
        b.setUp()
        self.addCleanup(b.doCleanups)
        premise='Пользователь готовится к экзамену по биологии'
        conclusion='Для подготовки нужен план повторения'
        b.model_result=[dict(content=premise,kind='fact',confidence=0.99)]
        self.assertEqual(b.save(101,9101,'Готовлюсь сдавать биологию.',provider),1)
        row=lifecycle.list_live_memories(101,datetime.now(timezone.utc),db_path=b.db)[0]
        message='Раз скоро проверят знания, придётся распределить повторение по дням.'
        b.model_result={'memories':[dict(content=conclusion,kind='hypothesis',confidence=0.99)],
            'relationships':[dict(source_memory_id=row['id'],target_candidate_index=0,
                relation_type='depends_on',confidence=0.99,evidence_quote=message)]}
        diag={}
        self.assertEqual(b.save(101,9102,message,provider,diagnostics=diag),1)
        self.assertEqual(diag['saved_relationships'],1)
        self.assertIn(conclusion,b.read(101,'план повторения'))
        b.model_result=[dict(content='Пользователь отменил подготовку к экзамену',kind='fact',
            confidence=0.99,replaces_memory_id=row['id'],replaces_content=premise,revision_kind='change',
            revision_evidence_quote='Передумал сдавать экзамен, больше не готовлюсь.')]
        self.assertEqual(b.save(101,9103,'Передумал сдавать экзамен, больше не готовлюсь.',provider),1)
        self.assertNotIn(conclusion,b.read(101,'план повторения'))

    def test_gemini_natural_dependency_uses_scoped_refs(self):
        self.provider_path('Gemini')

    def test_groq_natural_dependency_uses_same_contract(self):
        self.provider_path('Groq')
