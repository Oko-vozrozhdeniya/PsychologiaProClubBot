"""Natural recall, past/current separation, provenance and private archive scope.

AI outputs are supplied in these tests; live language understanding is a
separate Telegram acceptance, never inferred from fixture JSON.
"""
import io
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest
from unittest.mock import patch, Mock
from memory import coordinator, lifecycle, storage, cleanup_worker, ai_providers, ai_contract
from memory.ai_contract import parse_search_plan
import test_persistent_relationships as fixtures
import test_bot_memory_round_trip as bot_fixtures

NOW=fixtures.NOW


def plan(*queries, clarification=False, search=True, scope='historical_reference'):
    return dict(search_history=search, needs_clarification=clarification, queries=list(queries), statement_scope=scope)


class HistoricalRecall(unittest.TestCase):
    setUp=fixtures.PersistentRelationships.setUp
    message=fixtures.PersistentRelationships.message
    fact=fixtures.PersistentRelationships.fact
    snapshot=fixtures.PersistentRelationships.snapshot
    permissions=fixtures.PersistentRelationships.permissions
    revise=fixtures.PersistentRelationships.revise

    def archive(self, records):
        self.path=Path(self.temporary.name)/'private_history.jsonl'
        self.path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        return self.path

    def pair(self, question, answer, *, user=101, age=2, session=0):
        when=(NOW-timedelta(days=age)).isoformat()
        return dict(user_id=user, session_id=session, timestamp=when,
                    question_timestamp=when, answer_timestamp=when,question=question,answer=answer)

    def read(self, queries=None, now=NOW, policy=None, permissions=None, **kwargs):
        return coordinator.read_historical_context(101,queries or ['план тренировок','велопоход'],
            policy or fixtures.policy(),permissions if permissions is not None else self.permissions(now=now),
            now.isoformat(),db_path=self.db,archive_path=getattr(self,'path',None),**kwargs)

    def test_old_session_preserves_generated_plan_and_roles_without_reactivating(self):
        self.archive([self.pair('Готовлюсь к велопоходу, нужен план на вечер.',
            'Вечером: велотренажёр 20 минут, затем растяжка.',session=0)])
        before=self.path.read_bytes()
        r=self.read(['план','велотренажёр'])
        self.assertIn('велотренажёр 20 минут',r['context'])
        self.assertIn('"role": "assistant"',r['context'])
        self.assertIn('"role": "user"',r['context'])
        self.assertIn('past_conversation',r['context'])
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(len(lifecycle.list_live_memories(101,NOW,db_path=self.db)),2)

    def test_reviewed_need_and_previous_goal_are_historical_not_active(self):
        goal=self.fact('Готовлюсь к велопоходу','goal')
        need=self.fact('Нужен вечерний план тренировок','need',kind='preference')
        witness=self.message('because','План нужен ради велопохода.')
        coordinator.remember_dependency(101,self.snapshot(goal),self.snapshot(need),witness,
            fixtures.policy(),self.permissions(),NOW.isoformat(),explicitly_confirmed=True,db_path=self.db)
        self.revise(goal,'Отказался от велопохода')
        current=lifecycle.list_live_memories(101,NOW,db_path=self.db)
        self.assertNotIn(need,{r['id'] for r in current})
        r=self.read()
        self.assertIn('Нужен вечерний план тренировок',r['context'])
        self.assertIn('requires_review',r['context'])
        self.assertIn('superseded',r['context'])
        self.assertIn('Отказался от велопохода',r['context'])
        self.assertEqual(lifecycle.list_live_memories(101,NOW,db_path=self.db),current)
        self.assertEqual(self.snapshot(need)['kind'],'preference')

    def test_other_owner_and_other_owner_permissions_cannot_leak(self):
        self.archive([self.pair('Велопоход','SECRET_OTHER_OWNER',user=202),
                      self.pair('План тренировок','OWN_PLAN')])
        self.assertNotIn('SECRET_OTHER_OWNER',self.read()['context'])
        self.assertEqual(self.read()['scanned_history'],1)
        with self.assertRaises(PermissionError):
            self.read(policy=replace(fixtures.policy(),user_id=202))

    def test_no_stored_consent_denies_even_archive_only(self):
        self.archive([self.pair('Велопоход','PRIVATE_PLAN')])
        lifecycle.set_memory_consent(101,False,NOW,db_path=self.db)
        with self.assertRaises(PermissionError):self.read()

    def test_exact_365_day_boundary_future_and_orphan_are_excluded(self):
        old=self.pair('Велопоход','OLD',age=365)
        future=self.pair('Велопоход','FUTURE',age=-1)
        orphan=self.pair('Велопоход','ORPHAN',age=365)
        orphan['answer_timestamp']=(NOW-timedelta(days=364)).isoformat()
        self.archive([old,future,orphan,self.pair('Велопоход','FRESH',age=364)])
        r=self.read()
        self.assertEqual(r['scanned_history'],1)
        self.assertIn('FRESH',r['context'])
        for value in ('OLD','FUTURE','ORPHAN'):self.assertNotIn(value,r['context'])

    def test_explicit_source_deletion_excludes_original_archive_pair(self):
        text='Готовлюсь к велопоходу'
        record=self.pair(text,'PRIVATE_DELETED_PLAN',age=2)
        sid=self.message('deleted',text,created=NOW-timedelta(days=2))
        self.archive([record])
        self.assertIn('PRIVATE_DELETED_PLAN',self.read()['context'])
        lifecycle.delete_message(101,sid,db_path=self.db)
        self.assertEqual(self.read()['context'],'')
        self.assertIn('PRIVATE_DELETED_PLAN',self.path.read_text())
        lifecycle.purge_expired(NOW+timedelta(days=365),db_path=self.db)
        with storage._connection(self.db) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM private_archive_exclusions').fetchone()[0],0)

    def test_explicit_forget_does_not_resurface_raw_source_pair(self):
        text='Готовлюсь к велопоходу'
        sid=self.message('forgotten',text,created=NOW-timedelta(days=2))
        mid=lifecycle.add_memory(101,text,'fact',sid,NOW,db_path=self.db)
        self.archive([self.pair(text,'FORGOTTEN_TRAINING_PLAN',age=2)])
        self.assertIn('FORGOTTEN_TRAINING_PLAN',self.read()['context'])
        # Wrong-owner deletion cannot hide the owner's history.
        self.assertFalse(storage.delete_memory(202,mid,db_path=self.db))
        self.assertIn('FORGOTTEN_TRAINING_PLAN',self.read()['context'])
        self.assertTrue(coordinator.forget_memory(101,mid,fixtures.policy(),NOW.isoformat(),db_path=self.db))
        self.assertEqual(self.read()['context'],'')

    def test_sql_history_sources_must_be_independently_allowed(self):
        goal=self.fact('Готовлюсь к велопоходу','goal')
        old=self.snapshot(goal)['source']
        self.revise(goal,'Отказался от велопохода')
        self.assertIn('superseded',self.read()['context'])
        allowed=[p for p in self.permissions() if p.source!=old]
        self.assertNotIn('superseded',self.read(permissions=allowed)['context'])

    def test_irrelevant_history_does_not_fall_back_to_recent_records(self):
        self.archive([self.pair('Обсуждали шахматы','Пешка и слон')])
        self.assertEqual(self.read()['context'],'')

    def test_large_plan_returns_bounded_labelled_excerpts(self):
        self.archive([self.pair('План тренировок',('план тренировок вечером '*400))])
        r=self.read(max_chars=1200)
        self.assertLessEqual(len(r['context']),1200)
        self.assertIn('"partial": true',r['context'])

    def test_malformed_archive_does_not_claim_success(self):
        self.archive([]).write_text('{bad json}\n')
        with self.assertRaises(ValueError):self.read()

    def test_read_does_not_refresh_dates_or_write_new_facts(self):
        self.archive([self.pair('Велопоход','План тренировок вечером')])
        before=self.path.read_bytes()
        for _ in range(2):self.read()
        self.assertEqual(before,self.path.read_bytes())
        self.assertEqual(len(lifecycle.list_live_memories(101,NOW,db_path=self.db)),2)


class SearchContract(unittest.TestCase):
    def test_search_envelope_rejects_authority_extra_keys_and_bad_types(self):
        bad=[dict(plan('план'),user_id=202),dict(plan('план'),queries=['x']*6),
             dict(plan('план'),search_history=1),dict(plan('план'),queries=[True]),
             dict(plan('план'),queries=[]),dict(plan('план'),queries=['x'*161]),dict(plan('план'),statement_scope='grant_access')]
        for data in bad:
            with self.subTest(data=data),self.assertRaises(ai_contract.ExtractionFormatError):
                parse_search_plan(json.dumps(data))
        for value in ('{}','[]','{"search_history":true,"search_history":false,"needs_clarification":false,"queries":["x"]}',
                      '{"search_history":true,"needs_clarification":false,"queries":[NaN]}'):
            with self.assertRaises(ai_contract.ExtractionFormatError):parse_search_plan(value)

    def test_ambiguity_cannot_smuggle_guessed_queries(self):
        result=parse_search_plan(json.dumps(plan('случайная тема',clarification=True)))
        self.assertEqual(result['queries'],[])

    def test_truncation_is_not_a_valid_plan(self):
        with self.assertRaises(ai_contract.ExtractionTruncatedError):
            parse_search_plan(json.dumps(plan('план')),'MAX_TOKENS')

    def response(self,provider,text):
        if provider=='Gemini':return {'candidates':[{'content':{'parts':[{'text':text}]}}]}
        return {'choices':[{'message':{'content':text},'finish_reason':'stop'}]}

    def call(self,provider,text):
        with patch.dict('os.environ',{'GROQ_API_KEY':'fake'}),patch('urllib.request.urlopen',
            return_value=io.BytesIO(json.dumps(self.response(provider,text)).encode())) as fetch:
            r=ai_providers.request_memory_search('Вернёмся к вечернему графику.',[],
                'Отказался от велопохода',provider.lower(),gemini_key='fake',gemini_model='gemini-test')
        return r,json.loads(fetch.call_args.args[0].data)

    def test_both_providers_use_same_semantic_task_and_json_budget(self):
        text=json.dumps(plan('велопоход','план тренировок'))
        (g,gprovider),gp=self.call('Gemini',text)
        (q,qprovider),qp=self.call('Groq',text)
        self.assertEqual(g,q)
        self.assertEqual(gprovider,'Gemini');self.assertEqual(qprovider,'Groq')
        self.assertEqual(gp['system_instruction']['parts'][0]['text'],qp['messages'][0]['content'])
        self.assertEqual(gp['generationConfig']['maxOutputTokens'],1024)
        self.assertEqual(qp['max_completion_tokens'],1024)
        self.assertEqual(qp['response_format'],{'type':'json_object'})

    def test_auto_planner_uses_reserve_on_invalid_primary_result(self):
        with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(
                self.response('Gemini','not JSON')).encode())),patch.object(ai_providers,'request_groq',
                return_value=json.dumps(plan('тренировки'))) as groq:
            result,provider=ai_providers.request_memory_search('Вернёмся к тренировкам.',[],
                '', 'auto',gemini_key='fake',gemini_model='test')
        self.assertEqual(provider,'Groq');self.assertTrue(result['search_history'])
        self.assertEqual(groq.call_args.kwargs['response_task'],'search')

    def test_planner_gets_bounded_recent_context_not_full_archive(self):
        h=[{'role':'user','parts':[{'text':'ANCHOR '+'x'*4000}]}]*30
        with patch.object(ai_providers,'request_groq',return_value=json.dumps(plan('план'))) as call:
            ai_providers.request_memory_search('А что я тогда планировал?',h,'x'*20000,'groq',
                gemini_key='',gemini_model='test')
        payload=json.loads(call.call_args.args[0])
        self.assertLessEqual(sum(len(i['text']) for i in payload['recent']),6000)
        self.assertLessEqual(len(payload['active_context']),6000)


class BotHistoricalRecall(unittest.TestCase):
    def setUp(self):
        self.b=bot_fixtures.ActualMemoryRoundTripTests(methodName='runTest')
        self.b.setUp();self.addCleanup(self.b.doCleanups)
        self.path=Path(self.b.tmp.name)/'private_history.jsonl'
        now=datetime.now(timezone.utc)-timedelta(days=1)
        cleanup_worker.append_archive_record(self.path,dict(user_id=101,session_id=0,
            timestamp=now.isoformat(),question='Для велопохода нужен вечерний план тренировок',
            answer='Вечером велотренажёр 20 минут, затем растяжка.'))
        self.b.ns['DIALOG_HISTORY_FILE']=self.path
        self.lookup=bot_fixtures.load_function('read_historical_memory_prompt',self.b.ns)
        self.real=coordinator.read_historical_context

    def local(self,*args,**kwargs):
        return self.real(*args,db_path=self.b.db,**kwargs)

    def lookup_with(self,text,search_plan):
        with patch.object(ai_providers,'request_memory_search',return_value=(search_plan,'Gemini')) as planner,patch.object(
                coordinator,'read_historical_context',side_effect=self.local):
            diag={};r=self.lookup(101,text,[], '', 'auto',diag)
        return r,diag,planner

    def test_indirect_requests_are_not_gated_by_magic_word(self):
        for text in ('Вернёмся к вечерним занятиям.', 'Мы ведь уже обсуждали тренировки.',
                     'А что я тогда собирался делать?', 'Тот план для велопохода снова нужен.'):
            # Semantic interpretations are supplied by the adapter fixture.
            r,diag,planner=self.lookup_with(text,plan('велопоход','план тренировок'))
            self.assertIn('велотренажёр',r)
            self.assertEqual(diag['history_search_status'],'выполнено')
            self.assertEqual(planner.call_args.args[0],text)

    def test_ambiguous_reference_requests_clarification_without_reading_archive(self):
        with patch.object(ai_providers,'request_memory_search',return_value=(plan(clarification=True),'Groq')), \
             patch.object(coordinator,'read_historical_context') as read:
            r=self.lookup(101,'А что тогда?',[], '', 'groq',{})
        self.assertIn('уточни',r);read.assert_not_called()

    def test_unrelated_general_question_does_not_read_archive(self):
        with patch.object(ai_providers,'request_memory_search',return_value=(plan(search=False),'Gemini')), \
             patch.object(coordinator,'read_historical_context') as read:
            self.assertEqual(self.lookup(101,'Сколько будет дважды два?',[], '', 'auto',{}),'')
        read.assert_not_called()

    def test_no_consent_sends_nothing_to_planner(self):
        lifecycle.set_memory_consent(101,False,datetime.now(timezone.utc),db_path=self.b.db)
        with patch.object(ai_providers,'request_memory_search') as planner:
            self.assertEqual(self.lookup(101,'Вернёмся к плану',[], '', 'auto',{}),'')
        planner.assert_not_called()

    def test_revocation_during_semantic_planning_denies_archive_read(self):
        def revoke(*args,**kwargs):
            lifecycle.set_memory_consent(101,False,datetime.now(timezone.utc),db_path=self.b.db)
            return plan('план'),'Gemini'
        with patch.object(ai_providers,'request_memory_search',side_effect=revoke),patch.object(
                coordinator,'read_historical_context') as read:
            with self.assertRaises(PermissionError):self.lookup(101,'Вернёмся к плану',[], '', 'auto',{})
        read.assert_not_called()

    def test_planner_failure_uses_scoped_local_fallback_and_reports_error(self):
        with patch.object(ai_providers,'request_memory_search',side_effect=TimeoutError),patch.object(
                coordinator,'read_historical_context',side_effect=self.local):
            diag={};r=self.lookup(101,'план тренировок',[], '', 'auto',diag)
        self.assertIn('велотренажёр',r)
        self.assertEqual(diag['history_planner_error'],'TimeoutError')


class RealHandlerHistoryRecall(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import test_retention_operation as retention
        self.clock=retention.NOW
        self.f=retention.ActualPrivateRetentionTests(methodName='runTest')
        self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.ns,self.db=self.f.ns,self.f.db
        self.path=self.ns['DIALOG_HISTORY_FILE']
        when=self.clock-timedelta(days=2)
        cleanup_worker.append_archive_record(self.path,dict(user_id=101,session_id=0,
            timestamp=when.isoformat(),question='Я готовлюсь к велопоходу. Нужен вечерний план тренировок.',
            answer='План на вечер: велотренажёр 20 минут и растяжка.'))
        sid='telegram:private:101:cancel'
        lifecycle.register_message(101,sid,'Я отказался от велопохода.',self.clock-timedelta(days=1),db_path=self.db)
        self.mid=lifecycle.add_memory(101,'Пользователь отказался от велопохода','fact',sid,self.clock,db_path=self.db)
        self.ns['read_dialog_state']=lambda:{'101':1}
        self.f.fixture.model_result=[dict(content='Снова готовлюсь к велопоходу',kind='fact',confidence=0.99)]
        bot_fixtures.load_function('read_historical_memory_prompt',self.ns)
        self.f.message.text='Вернёмся к прежнему вечернему плану тренировок.'
        self.f.message.date=self.clock
        self.calls=[]
        def answer(question,history,memory,mode,gemini):
            self.calls.append((question,history,memory,mode))
            return 'Ответ по найденному прошлому плану','Groq' if mode=='groq' else 'Gemini'
        self.ns['request_personal_ai']=answer
        self.real=coordinator.read_historical_context

    def local(self,*a,**kw):
        return self.real(*a,db_path=self.db,**kw)

    async def run_provider(self,mode):
        self.f.app.bot_data['test_ai_mode']=mode
        with patch.object(ai_providers,'request_memory_search',return_value=(plan('велопоход','план тренировок'),
                'Groq' if mode=='groq' else 'Gemini')),patch.object(coordinator,'read_historical_context',side_effect=self.local):
            await self.f.invoke()
        self.ns['send_error_response'].assert_not_awaited()
        self.assertEqual(len(self.calls),1)
        _,short,prompt,_=self.calls[0]
        self.assertEqual(short,[])
        self.assertIn('велотренажёр 20 минут',prompt)
        self.assertIn('отказался от велопохода',prompt)
        self.assertIn('past_conversation',prompt)
        self.assertEqual(len(lifecycle.list_live_memories(101,self.clock,db_path=self.db)),1)
        self.assertEqual(self.f.records[-1]['history_search_status'],'выполнено')
        self.assertEqual(self.f.records[-1]['write_status'],'пропущено: обращение к прошлому без нового утверждения')
        self.assertGreater(self.f.records[-1]['history_matched'],0)

    async def test_gemini_handler_sees_prior_session_plan_and_current_cancelled_state(self):
        await self.run_provider('gemini')

    async def test_groq_handler_sees_same_past_current_separation(self):
        await self.run_provider('groq')

    async def test_real_new_decision_is_still_extracted_and_saved(self):
        self.f.message.text='Я всё-таки решил снова готовиться к велопоходу.'
        self.f.fixture.model_result=[dict(content='Готовлюсь к велопоходу',kind='fact',confidence=0.99)]
        with patch.object(ai_providers,'request_memory_search',return_value=(
                plan('велопоход',scope='current_assertion'),'Gemini')), \
             patch.object(coordinator,'read_historical_context',side_effect=self.local):
            await self.f.invoke()
        self.ns['send_error_response'].assert_not_awaited()
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.f.records[-1]['write_status'],'выполнено')
        self.assertEqual(self.f.records[-1]['saved_facts'],1)
        self.assertTrue(any(m['content']=='Готовлюсь к велопоходу'
            for m in lifecycle.list_live_memories(101,self.clock,db_path=self.db)))

    async def test_opt_out_during_planning_stops_main_answer_and_archive_append(self):
        before=self.path.read_bytes()
        def revoke(*a,**kw):
            lifecycle.set_memory_consent(101,False,self.clock,db_path=self.db)
            return plan('план тренировок'),'Gemini'
        with patch.object(ai_providers,'request_memory_search',side_effect=revoke):
            await self.f.invoke()
        self.assertEqual(self.calls,[])
        self.assertEqual(self.path.read_bytes(),before)

    async def test_archive_failure_keeps_active_answer_and_does_not_claim_history_found(self):
        self.path.write_text('{malformed}\n')
        # Short-term read is isolated here so this specifically exercises the
        # separate historical retrieval failure boundary.
        self.ns['read_recent_dialog']=lambda *a:[]
        with patch.object(ai_providers,'request_memory_search',return_value=(plan('велопоход'),'Gemini')), \
             patch.object(coordinator,'read_historical_context',side_effect=self.local):
            await self.f.invoke()
        self.assertEqual(len(self.calls),1)
        self.assertIn('не выполнен',self.calls[0][2])
        self.assertNotIn('велотренажёр',self.calls[0][2])
        self.assertEqual(self.f.records[-1]['history_search_status'],'ошибка')
