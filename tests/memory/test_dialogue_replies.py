"""Natural-response harness mechanics only, no semantic claims from mocks."""
import io
import json
import os
import asyncio
import unittest
from unittest.mock import patch, AsyncMock
from tools import dialogue_replies as r
from memory.ai_contract import ExtractionFormatError, ExtractionTruncatedError, instructions_for

class ReplyTests(unittest.TestCase):
    def test_reply_contract_rejects_action_labels_duplicates_empty_and_truncated(self):
        self.assertEqual(r.parse_reply('{"reply":"Текст"}'),'Текст')
        for value in ('{"action":"close"}','{"reply":"a","reply":"b"}',
                      '{"reply":12}','{"reply":""}'):
            with self.assertRaises(ExtractionFormatError):
                r.parse_reply(value)
        with self.assertRaises(ExtractionTruncatedError):
            r.parse_reply('{"reply":"Текст"}','length')

    def test_native_direct_reply_has_no_assigned_dialogue_mode_or_expected_label(self):
        body={'choices':[{'message':{'content':'{"reply":"Живой ответ"}'},'finish_reason':'stop'}]}
        case=r.CASES[3]
        with patch.dict(os.environ,{'GROQ_API_KEY':'FAKE','GROQ_MODEL':'unchanged'}), \
                patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(body).encode())) as fetch:
            self.assertEqual(r.generate_reply(case[1],case[2],'answer'),'Живой ответ')
            self.assertEqual(os.environ['GROQ_MODEL'],'unchanged')
        payload=json.loads(fetch.call_args.args[0].data)
        self.assertEqual(payload['model'],r.MODEL)
        self.assertEqual([m['role'] for m in payload['messages']],['system','user','assistant','user'])
        self.assertEqual(payload['messages'][-1]['content'],case[1])
        self.assertEqual(payload['messages'][0]['content'],instructions_for('private',reply_mode='answer')+r.RULE+r.FORMAT)
        self.assertEqual(payload['response_format']['json_schema']['schema'],r.SCHEMA)
        self.assertNotIn(r.EXPECTATIONS[case[3]],payload['messages'][0]['content'])

class ReplyHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_maximum_requests_and_followups_use_actual_answer_not_assumed_clarification(self):
        histories=[]
        def generate(q,h,mode):
            histories.append((q,h,mode))
            return 'ACTUAL GENERATED RESPONSE'
        with patch.object(r,'request_memory_search',return_value=({'reply_mode':'brief_close'},'Groq')) as plan, \
                patch.object(r,'request_closure_review',return_value=('conversation','Groq')) as review, \
                patch.object(r,'generate_reply',side_effect=generate) as gen, \
                patch.object(r.asyncio,'sleep',new_callable=AsyncMock):
            report=await r.compare_replies(AsyncMock())
        self.assertEqual(plan.call_count,4)
        self.assertEqual(review.call_count,4)
        self.assertEqual(gen.call_count,20)
        self.assertIn('28/28',report)
        self.assertEqual(sum(m=='brief_close' for _,_,m in histories),4)
        self.assertEqual(sum(m=='answer' for _,_,m in histories),16)
        for _,h,_ in histories[-2:]:
            self.assertEqual(h[-1]['role'],'model')
            self.assertEqual(h[-1]['parts'][0]['text'],'ACTUAL GENERATED RESPONSE')
        self.assertEqual(report.count('Ответ модели: ACTUAL GENERATED RESPONSE'),20)
        self.assertNotIn('4/4',report)

    async def test_first_error_stops_and_preserves_partial_report_without_raw_exception(self):
        with patch.object(r,'request_memory_search',side_effect=RuntimeError('SECRET')), \
                patch.object(r,'generate_reply') as generate:
            report=await r.compare_replies(AsyncMock())
        self.assertIn('1/28',report)
        self.assertIn('fatigue_without_context / baseline',report)
        self.assertIn('RuntimeError',report)
        self.assertNotIn('SECRET',report)
        generate.assert_not_called()

    async def test_cancellation_is_not_swallowed_into_a_success_report(self):
        with patch.object(r.asyncio,'to_thread',side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await r.compare_replies(AsyncMock())
