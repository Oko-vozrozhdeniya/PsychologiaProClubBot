"""Offline contracts only: these never assert that a live model understood a phrase."""
import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch
from tools import dialogue_lab as lab
from memory.ai_contract import ExtractionFormatError


class DialogueLabTests(unittest.TestCase):
    def observation(self, **changes):
        d = dict(unresolved_reference=False, candidates=['conversation'], closure_act='expressed',
                 new_request=False, evidence=['До свидания'])
        d.update(changes)
        return d

    def test_multiple_readings_or_unresolved_reference_block_closure(self):
        for d in (self.observation(candidates=['conversation','topic']),
                  self.observation(unresolved_reference=True),
                  self.observation(closure_act='inferred'), self.observation(evidence=[])):
            self.assertEqual(lab.choose_action(d), 'clarify')
        self.assertEqual(lab.choose_action(self.observation()), 'close')

    def test_new_request_and_known_topic_are_not_forced_into_closure(self):
        self.assertEqual(lab.choose_action(self.observation(new_request=True)), 'answer')
        self.assertEqual(lab.choose_action(self.observation(candidates=['topic'])), 'answer')
        self.assertEqual(lab.choose_action(self.observation(candidates=['conversation','topic','pause','other'],
            closure_act='absent')), 'answer')
        self.assertEqual(lab.choose_action(self.observation(closure_act='absent',
            unresolved_reference=True)), 'clarify')

    def test_forged_quotes_duplicate_keys_bad_types_are_rejected(self):
        good = self.observation()
        self.assertEqual(lab.parse_observations(json.dumps(good), messages=['До свидания']), good)
        for value in (dict(good, evidence=['Ненаписанная цитата']), dict(good, candidates=['topic','topic']),
                      dict(good, unresolved_reference=1), dict(good, candidates=[{}])):
            with self.assertRaises(ExtractionFormatError):
                lab.parse_observations(json.dumps(value), messages=['До свидания'])

    def test_no_live_flag_or_missing_key_sends_no_request(self):
        with patch.object(lab, 'run_case') as run, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), patch.dict('os.environ', {}, clear=True):
            self.assertEqual(lab.main([]), 0)
            self.assertEqual(lab.main(['--live']), 2)
            run.assert_not_called()

    def test_candidate_schema_reaches_native_groq_and_gemini_clients(self):
        observation=self.observation(unresolved_reference=True, candidates=['topic','conversation'],
            closure_act='inferred', evidence=[])
        value=json.dumps(observation)
        for provider in ('groq','gemini'):
            body=({'choices':[{'message':{'content':value},'finish_reason':'stop'}]}
                if provider=='groq' else {'candidates':[{'content':{'parts':[{'text':value}]}}]})
            details={}
            with self.subTest(provider=provider), patch.dict(os.environ,{'GROQ_API_KEY':'FAKE'}), \
                    patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(body).encode())) as fetch:
                action=lab.run_case('candidate',lab.CASES[0],provider,'FAKE','test',details=details)
            self.assertEqual(action,'clarify')
            self.assertEqual(fetch.call_count,1)
            self.assertTrue(details['unresolved_reference'])
            self.assertEqual(details['candidates'],['topic','conversation'])
            self.assertNotIn('evidence',details)

    def test_native_groq_default_search_still_rejects_other_contract(self):
        from memory.ai_providers import request_groq
        body={'choices':[{'message':{'content':json.dumps(self.observation())},'finish_reason':'stop'}]}
        with patch.dict(os.environ,{'GROQ_API_KEY':'FAKE'}),patch('urllib.request.urlopen',
                return_value=io.BytesIO(json.dumps(body).encode())):
            with self.assertRaises(ExtractionFormatError):
                request_groq('synthetic',system_instruction='JSON',response_task='search')

    def test_report_keeps_baseline_candidate_and_errors_distinct(self):
        def fake(policy, case, *args):
            if policy == 'current':
                raise TimeoutError('DO NOT PRINT RAW ERROR')
            return case[3]
        out=io.StringIO()
        with patch.object(lab,'run_case',side_effect=fake), patch.dict('os.environ',
                {'GEMINI_API_KEY':'FAKE SECRET'}), contextlib.redirect_stdout(out):
            self.assertEqual(lab.main(['--live','--provider','gemini']), 1)
        self.assertNotIn('FAKE SECRET',out.getvalue())
        self.assertNotIn('DO NOT PRINT RAW ERROR',out.getvalue())
        rows=[json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(sum(r.get('actual')=='error' for r in rows),4)
        self.assertEqual(sum(r.get('policy')=='candidate' and r['ok'] for r in rows),4)


class DialogueStrictTransportTests(unittest.TestCase):
    def test_each_profile_sends_its_model_schema_and_budget_without_changing_env(self):
        observation = dict(unresolved_reference=True, candidates=['conversation','topic'],
                           closure_act='inferred', new_request=False, evidence=[])
        body = {'choices':[{'message':{'content':json.dumps(observation)}, 'finish_reason':'stop'}]}
        for profile, model in lab.STRICT_PROFILES.items():
            with self.subTest(profile=profile), patch.dict(os.environ, {'GROQ_API_KEY':'FAKE',
                    'GROQ_MODEL':'unchanged-model'}), patch('urllib.request.urlopen',
                    return_value=io.BytesIO(json.dumps(body).encode())) as fetch:
                self.assertEqual(lab.run_case(profile, lab.CASES[0], 'groq', '', ''), 'clarify')
                payload = json.loads(fetch.call_args.args[0].data)
                self.assertEqual(payload['model'], model)
                self.assertEqual(payload['response_format']['type'], 'json_schema')
                self.assertIs(payload['response_format']['json_schema']['strict'], True)
                self.assertEqual(payload['response_format']['json_schema']['schema'], lab.OBSERVATION_SCHEMA)
                self.assertEqual(payload['max_completion_tokens'],4096)
                self.assertEqual(payload['reasoning_effort'],'medium')
                self.assertIs(payload['include_reasoning'],False)
                self.assertEqual(os.environ['GROQ_MODEL'],'unchanged-model')
                self.assertEqual(fetch.call_count,1)

    def test_http_failure_preserves_machine_code_and_hides_remote_message_and_failed_generation(self):
        import urllib.error
        from memory.ai_providers import GroqRequestError
        body = {'error':{'type':'invalid_request_error','code':'json_validate_failed',
                         'message':'RAW_SECRET_KEY', 'failed_generation':'PRIVATE_TEXT'*700}}
        error = urllib.error.HTTPError('https://api.groq.com/',400,'bad',{},
                                      io.BytesIO(json.dumps(body).encode()))
        with patch.dict(os.environ,{'GROQ_API_KEY':'FAKE'}), \
                patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaises(GroqRequestError) as caught:
                lab.run_case('strict_20b',lab.CASES[0],'groq','','')
        safe = lab.safe_error(caught.exception)
        self.assertEqual(safe,dict(error='GroqRequestError',http_code=400,
            provider_type='invalid_request_error',provider_code='json_validate_failed'))
        self.assertNotIn('RAW_SECRET',repr(safe))
        self.assertNotIn('PRIVATE_TEXT',repr(safe))
        self.assertNotIn('RAW_SECRET',caught.exception.reason)
        forged=GroqRequestError(reason='SECRET',provider_type='SECRET + injected',provider_code='KEY\nTOKEN')
        self.assertEqual(lab.safe_error(forged),{'error':'GroqRequestError'})

    def test_structured_options_cannot_leak_into_ordinary_reply(self):
        from memory.ai_providers import request_groq
        with patch.dict(os.environ,{'GROQ_API_KEY':'FAKE'}), patch('urllib.request.urlopen') as fetch:
            with self.assertRaises(ValueError):
                request_groq('normal',model_override='openai/gpt-oss-120b')
            fetch.assert_not_called()


class DialogueVerificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_pass_runs_two_smoke_rounds_then_held_out_cases_per_model(self):
        from unittest.mock import AsyncMock
        emit=AsyncMock()
        with patch.object(lab,'run_case',side_effect=lambda p,c,*a,**kw:c[3]) as run, \
                patch.object(lab.asyncio,'sleep',new_callable=AsyncMock):
            results=await lab.verify_profiles(emit)
        self.assertEqual(run.call_count,lab.VERIFY_MAX_CALLS)
        self.assertEqual(len(results),2)
        self.assertTrue(all(r['passed'] and r['correct']==14 and r['tested']==14 for r in results))
        calls=run.call_args_list[:14]
        self.assertEqual([c.args[1][0] for c in calls],
                         [c[0] for c in lab.CASES[:4]]*2+[c[0] for c in lab.CASES[4:]])
        self.assertTrue(all(c.args[2]=='groq' for c in run.call_args_list))

    async def test_second_round_regression_disqualifies_profile_and_skips_extra_cases(self):
        from unittest.mock import AsyncMock
        counts={}
        def fake(profile,case,*args,**kw):
            counts[profile]=counts.get(profile,0)+1
            return 'close' if counts[profile]==5 else case[3]
        with patch.object(lab,'run_case',side_effect=fake) as run, \
                patch.object(lab.asyncio,'sleep',new_callable=AsyncMock):
            results=await lab.verify_profiles(AsyncMock())
        self.assertEqual(run.call_count,16)
        self.assertTrue(all(not r['passed'] and r['tested']==8 for r in results))

    async def test_bad_request_stops_profile_without_retry_but_quota_stops_entire_run(self):
        from unittest.mock import AsyncMock
        from memory.ai_providers import GroqRequestError
        for code, expected_calls in ((400,2),(429,1),(503,1)):
            with self.subTest(code=code), patch.object(lab,'run_case',
                    side_effect=GroqRequestError(http_code=code)) as run, \
                    patch.object(lab.asyncio,'sleep',new_callable=AsyncMock):
                results=await lab.verify_profiles(AsyncMock())
            self.assertEqual(run.call_count,expected_calls)
            self.assertTrue(all(not r['passed'] and r['errors']==1 for r in results))
