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

class GroundsThenReplyTests(unittest.TestCase):
    def test_assessment_is_a_separate_assistant_turn_before_final_request(self):
        from memory.reply_review import audit_reply,reply_after_audit,AUDIT_CHECK,AFTER_AUDIT_CHECK
        q='Исходная реплика';draft='Неотправленный ответ';summary='Краткое обоснование'
        with patch('memory.reply_review.request_plain_text',side_effect=[summary,'Итоговая реплика']) as request:
            got=audit_reply(q,[],draft,provider='gemini',gemini_key='KEY',gemini_model='test')
            final=reply_after_audit(q,[],draft,got,provider='gemini',gemini_key='KEY',gemini_model='test')
        self.assertEqual(final,'Итоговая реплика')
        first,second=request.call_args_list
        self.assertEqual(first.args[1],AUDIT_CHECK)
        self.assertEqual([m['parts'][0]['text'] for m in first.args[2]],[q])
        self.assertEqual(second.args[1],AFTER_AUDIT_CHECK)
        self.assertEqual([m['role'] for m in second.args[2]],['user','user','model','user'])
        self.assertEqual([m['parts'][0]['text'] for m in second.args[2]],[q,AUDIT_CHECK,summary,'ОБЪЕКТ ПРОВЕРКИ: неотправленный черновик помощника, не слова пользователя.\n'+draft])
        self.assertNotIn(summary,second.kwargs['instruction'])
        self.assertEqual(first.kwargs['gemini_key'],second.kwargs['gemini_key'])

    def test_empty_assessment_cannot_trigger_revision(self):
        from memory.reply_review import reply_after_audit
        with patch('memory.reply_review.request_plain_text') as request:
            with self.assertRaises(ValueError):
                reply_after_audit('q',[],'draft','',provider='gemini',gemini_key='KEY',gemini_model='test')
            request.assert_not_called()

class ChallengeHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_eight_calls_keep_every_assessment_before_its_final_reply(self):
        with patch.object(lab,'audit_reply',return_value='Обоснование') as audit, \
                patch.object(lab,'reply_after_audit',return_value='Итог') as revise, \
                patch.object(lab,'review_reply') as old, \
                patch.object(lab.asyncio,'sleep',new_callable=AsyncMock):
            report=await lab.compare_reviews(AsyncMock(),provider='gemini',gemini_key='SECRET',gemini_model='test',challenge=True)
        self.assertEqual(audit.call_count,4)
        self.assertEqual(revise.call_count,4)
        self.assertEqual([c.args[3] for c in revise.call_args_list],['Обоснование']*4)
        old.assert_not_called()
        self.assertEqual(report.count('Краткое обоснование агента: Обоснование'),4)
        self.assertEqual(report.count('Итоговая реплика после обоснования: Итог'),4)
        self.assertIn('8/8 запросов',report)
        self.assertNotIn('SECRET',report)

    async def test_second_stage_failure_preserves_the_assessment_and_stops_without_retry(self):
        with patch.object(lab,'audit_reply',return_value='Полученное обоснование') as audit, \
                patch.object(lab,'reply_after_audit',side_effect=GroqRequestError(http_code=429,reason='SECRET')) as revise, \
                patch.object(lab.asyncio,'sleep',new_callable=AsyncMock):
            report=await lab.compare_reviews(AsyncMock(),provider='groq',gemini_key='',gemini_model='test',challenge=True)
        self.assertEqual(audit.call_count,1)
        self.assertEqual(revise.call_count,1)
        self.assertIn('Полученное обоснование',report)
        self.assertIn('2/8 запросов',report)
        self.assertNotIn('SECRET',report)

class RulesBeforeAssessmentTests(unittest.TestCase):
    def test_same_policy_is_sent_before_each_review_task_separate_from_user_data(self):
        from memory.reply_review import (review_reply,audit_reply,reply_after_audit,
                                        REVIEW_POLICY,POLICY_PREFIX)
        case=lab.CASES[0]
        with patch('memory.reply_review.request_plain_text',return_value='Ответ') as request:
            options=dict(provider='gemini',gemini_key='KEY',gemini_model='test')
            review_reply(case[1],case[2],lab.DRAFTS['gemini'][0],**options)
            audit_reply(case[1],case[2],lab.DRAFTS['gemini'][0],**options)
            reply_after_audit(case[1],case[2],lab.DRAFTS['gemini'][0],'Краткая оценка',**options)
        self.assertEqual(request.call_count,3)
        for call in request.call_args_list:
            instruction=call.kwargs['instruction']
            self.assertTrue(instruction.startswith(POLICY_PREFIX))
            self.assertEqual(instruction.count(REVIEW_POLICY),1)
            self.assertLess(instruction.index(REVIEW_POLICY),instruction.index('ЗАДАЧА ПРОВЕРКИ.'))
            self.assertFalse(any(REVIEW_POLICY in p['text'] for m in call.args[2] for p in m['parts']))
        for evaluated_case in lab.CASES:
            self.assertNotIn(evaluated_case[1],REVIEW_POLICY)
