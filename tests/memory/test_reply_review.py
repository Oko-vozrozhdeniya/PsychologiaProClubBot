"""Unsent-draft review contracts. Mocks do not establish semantic success."""
import asyncio
import copy
import io
import json
import os
import unittest
from unittest.mock import patch,AsyncMock
from memory.reply_review import review_reply,REVIEW_INSTRUCTION,REVIEW_CHECK
from memory.ai_providers import GroqRequestError
from tools import dialogue_review as lab

class ReviewTransportTests(unittest.TestCase):
    def test_native_history_contains_original_message_draft_then_marked_service_check(self):
        case=lab.CASES[3]
        before=copy.deepcopy(case[2])
        for provider in ('gemini','groq'):
            draft=lab.DRAFTS[provider][3]
            body=({'choices':[{'message':{'content':'Итоговая реплика'},'finish_reason':'stop'}]}
                if provider=='groq' else {'candidates':[{'content':{'parts':[{'text':'Итоговая реплика'}]},'finishReason':'STOP'}]})
            with self.subTest(provider=provider),patch.dict(os.environ,{'GROQ_API_KEY':'FAKE'}), \
                    patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(body).encode())) as fetch:
                result=review_reply(case[1],case[2],draft,provider=provider,
                    gemini_key='KEY',gemini_model='test',groq_model=lab.MODEL)
            self.assertEqual(result,'Итоговая реплика')
            request=fetch.call_args.args[0]
            payload=json.loads(request.data)
            self.assertNotIn('KEY',request.full_url)
            if provider=='groq':
                messages=payload['messages']
                self.assertEqual(messages[0]['content'],REVIEW_INSTRUCTION)
                self.assertEqual([m['role'] for m in messages],['system','user','assistant','user','assistant','user'])
                self.assertEqual([m['content'] for m in messages[-3:]], [case[1],draft,REVIEW_CHECK])
                self.assertNotIn('response_format',payload)
            else:
                messages=payload['contents']
                self.assertEqual(payload['system_instruction']['parts'][0]['text'],REVIEW_INSTRUCTION)
                self.assertEqual([m['role'] for m in messages],['user','model','user','model','user'])
                self.assertEqual([m['parts'][0]['text'] for m in messages[-3:]],[case[1],draft,REVIEW_CHECK])
                self.assertNotIn('responseMimeType',payload['generationConfig'])
            self.assertEqual(case[2],before)
            self.assertNotIn(case[1],REVIEW_INSTRUCTION)
            self.assertNotIn(draft,REVIEW_CHECK)

    def test_invalid_inputs_fail_before_any_provider_request(self):
        variants=[('',[],'draft'),('q',[],'x'*6001),('q',[{}],'draft'),
                  ('q',[dict(role='user',parts=[dict(text='x')])]*61,'draft'),
                  ('q',[dict(role='user',parts=[dict(text=12)])],'draft')]
        with patch('memory.reply_review.request_plain_text') as request:
            for q,h,d in variants:
                with self.assertRaises(ValueError):
                    review_reply(q,h,d,provider='gemini',gemini_key='KEY',gemini_model='test')
            request.assert_not_called()

class ReviewHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_existing_drafts_and_controls_require_four_review_calls_only(self):
        for provider in ('gemini','groq'):
            with self.subTest(provider=provider),patch.object(lab,'review_reply',return_value='Проверенный текст') as review, \
                    patch.object(lab.asyncio,'sleep',new_callable=AsyncMock):
                report=await lab.compare_reviews(AsyncMock(),provider=provider,gemini_key='SECRET',gemini_model='test')
            self.assertEqual(review.call_count,4)
            self.assertEqual([c.args[2] for c in review.call_args_list],list(lab.DRAFTS[provider]))
            self.assertEqual([c.args[0] for c in review.call_args_list],[c[1] for c in lab.CASES[:4]])
            self.assertEqual(report.count('Ответ после проверки: Проверенный текст'),4)
            self.assertIn('explicit_farewell',report)
            self.assertIn('fatigue_grounded_in_topic',report)
            self.assertIn('4/4 запросов',report)  # transport count, explicitly not a score
            self.assertNotIn('SECRET',report)

    async def test_failure_stops_once_and_keeps_partial_answers_without_raw_error(self):
        with patch.object(lab,'review_reply',side_effect=['Сохранённый ответ',
                GroqRequestError(http_code=429,provider_code='rate_limit_exceeded',reason='SECRET')]) as review, \
                patch.object(lab.asyncio,'sleep',new_callable=AsyncMock):
            report=await lab.compare_reviews(AsyncMock(),provider='groq',gemini_key='',gemini_model='test')
        self.assertEqual(review.call_count,2)
        self.assertIn('Сохранённый ответ',report)
        self.assertIn('rate_limit_exceeded',report)
        self.assertNotIn('SECRET',report)
        self.assertIn('2/4 запросов',report)

    async def test_cancellation_propagates_without_starting_more_reviews(self):
        with patch.object(lab.asyncio,'to_thread',side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await lab.compare_reviews(AsyncMock(),provider='gemini',gemini_key='KEY',gemini_model='test')
