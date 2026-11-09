"""Native payload and policy tests; no claims about live semantic understanding."""
import io
import json
import os
import unittest
from unittest.mock import patch, AsyncMock
from tools import dialogue_lab as lab
from tools import dialogue_context as ctx
from memory.ai_contract import ExtractionFormatError

class ContextTests(unittest.TestCase):
    def test_native_roles_current_message_and_independent_stage_contracts(self):
        case=lab.CASES[3]
        binding=dict(status='resolved',referent='бухгалтерия',
                     evidence=['Ты устал от обсуждения бухгалтерии?'])
        intent=dict(action='answer',evidence=['Да, я устал от этого.'])
        def response(d):
            return io.BytesIO(json.dumps({'choices':[{'message':{'content':json.dumps(d)},
                                                      'finish_reason':'stop'}]}).encode())
        details={}
        with patch.dict(os.environ,{'GROQ_API_KEY':'FAKE','GROQ_MODEL':'unchanged'}), \
                patch('urllib.request.urlopen',side_effect=[response(binding),response(intent)]) as fetch, \
                patch.object(ctx.time,'sleep'):
            self.assertEqual(lab.run_case('context_120b',case,'groq','','',details=details),'answer')
            self.assertEqual(os.environ['GROQ_MODEL'],'unchanged')
        self.assertEqual(details,dict(http_calls=2,reference_status='resolved',intent_action='answer'))
        payloads=[json.loads(c.args[0].data) for c in fetch.call_args_list]
        for p in payloads:
            self.assertEqual([m['role'] for m in p['messages']],['system','user','assistant','user'])
            self.assertEqual(p['messages'][-1]['content'],case[1])
            self.assertEqual(p['messages'][-2]['content'],case[2][-1]['parts'][0]['text'])
            self.assertEqual(p['model'],ctx.MODEL)
            self.assertTrue(p['response_format']['json_schema']['strict'])
        self.assertEqual(payloads[0]['messages'][0]['content'],ctx.REFERENCE_INSTRUCTION)
        self.assertIn(ctx.INTENT_INSTRUCTION,payloads[1]['messages'][0]['content'])
        self.assertIn('бухгалтерия',payloads[1]['messages'][0]['content'])

    def test_unresolved_reference_blocks_closure_and_first_failure_does_not_call_second_stage(self):
        details={}
        with patch.object(ctx,'request_groq',side_effect=[
                json.dumps(dict(status='unclear',referent='',evidence=[])),
                json.dumps(dict(action='close',evidence=['На сегодня мне достаточно, спасибо.']))]) as call, \
                patch.object(ctx.time,'sleep'):
            self.assertEqual(ctx.run_context_case(lab.CASES[1],details=details),'clarify')
            self.assertEqual(call.call_count,2)
        details={}
        with patch.object(ctx,'request_groq',side_effect=TimeoutError) as call:
            with self.assertRaises(TimeoutError):
                ctx.run_context_case(lab.CASES[0],details=details)
            self.assertEqual(call.call_count,1)
            self.assertEqual(details['http_calls'],1)

    def test_forged_quotes_and_unjustified_resolution_cannot_reach_policy(self):
        values=[dict(status='resolved',referent='разговор',evidence=[]),
                dict(status='none',referent='разговор',evidence=[]),
                dict(status='resolved',referent='разговор',evidence=['forged'])]
        for value in values:
            with self.assertRaises(ExtractionFormatError):
                ctx.parse(json.dumps(value),None,reference=True,source=['real'])
        with self.assertRaises(ExtractionFormatError):
            ctx.parse('{"action":"close","action":"answer","evidence":[]}',None,
                      reference=False,source=[])
        with self.assertRaises(ExtractionFormatError):
            ctx.parse(json.dumps(dict(action='close',evidence=['assistant bye'])),None,
                      reference=False,source=['user text'])

class ContextBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_model_four_repeated_plus_six_cases_is_at_most_28_native_calls(self):
        def fake(p,c,*args,details,**kwargs):
            self.assertEqual(p,'context_120b')
            details['http_calls']=2
            return c[3]
        emit=AsyncMock()
        with patch.object(lab,'run_case',side_effect=fake) as call, \
                patch.object(lab.asyncio,'sleep',new_callable=AsyncMock):
            result=await lab.verify_profiles(emit,decomposed=True)
        self.assertEqual(call.call_count,14)
        self.assertEqual(len(result),1)
        self.assertTrue(result[0]['passed'])
        self.assertIn('28/28',str(emit.await_args_list))
