"""Verify actual plaintext request contracts, not model understanding."""
import io
import json
import os
import unittest
from unittest.mock import patch,AsyncMock
from tools import dialogue_plain as p
from memory.ai_providers import request_groq,GroqRequestError
from memory.ai_contract import ExtractionFormatError,ExtractionTruncatedError

class PlainTransportTests(unittest.TestCase):
    def test_native_providers_omit_json_constraints_preserve_roles_and_exclude_thoughts(self):
        case=p.CASES[3]
        bodies={
            'groq':{'choices':[{'message':{'content':'Текстовый ответ'},'finish_reason':'stop'}]},
            'gemini':{'candidates':[{'content':{'parts':[{'text':'HIDDEN','thought':True},
                {'text':'Текстовый ответ'}]},'finishReason':'STOP'}]}}
        prompts=[]
        for provider in bodies:
            with self.subTest(provider=provider),patch.dict(os.environ,{'GROQ_API_KEY':'FAKE','GROQ_MODEL':'unchanged'}), \
                    patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(bodies[provider]).encode())) as fetch:
                reply=p.generate_plain(provider,case[1],case[2],gemini_key='FAKE GEMINI',gemini_model='test')
                self.assertEqual(reply,'Текстовый ответ')
                self.assertEqual(os.environ['GROQ_MODEL'],'unchanged')
            req=fetch.call_args.args[0]
            body=json.loads(req.data)
            self.assertNotIn('FAKE GEMINI',req.full_url)
            self.assertNotIn('response_format',body)
            if provider=='groq':
                self.assertEqual([m['role'] for m in body['messages']],['system','user','assistant','user'])
                self.assertEqual(body['messages'][-1]['content'],case[1])
                self.assertEqual(body['temperature'],0)
                self.assertEqual(body['max_completion_tokens'],4096)
                self.assertEqual(body['reasoning_effort'],'medium')
                prompts.append(body['messages'][0]['content'])
            else:
                self.assertEqual(body['contents'],[*case[2],{'role':'user','parts':[{'text':case[1]}]}])
                self.assertEqual(body['generationConfig'],dict(temperature=0,maxOutputTokens=4096))
                prompts.append(body['system_instruction']['parts'][0]['text'])
        self.assertEqual(prompts,[p.INSTRUCTION]*2)
        self.assertNotIn('Верни JSON',p.INSTRUCTION)

    def test_plain_mode_requires_explicit_instruction_and_text_parser_without_schema(self):
        with patch.dict(os.environ,{'GROQ_API_KEY':'FAKE'}),patch('urllib.request.urlopen') as fetch:
            for options in ({}, {'system_instruction':'text'},
                    {'system_instruction':'text','structured_parser':p.parse_text,'response_schema':{}}):
                with self.assertRaises(ValueError):
                    request_groq('question',response_task='plain',**options)
            fetch.assert_not_called()

    def test_text_parser_preserves_plain_text_and_rejects_truncation(self):
        self.assertEqual(p.parse_text(' обычная реплика '),'обычная реплика')
        for text in ('',None,'x'*6001):
            with self.assertRaises(ExtractionFormatError):
                p.parse_text(text)
        with self.assertRaises(ExtractionTruncatedError):
            p.parse_text('unfinished','length')

class PlainHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_eight_calls_and_reports_without_automatic_semantic_scores(self):
        with patch.object(p,'generate_plain',return_value='Реальная реплика') as generate, \
                patch.object(p.asyncio,'sleep',new_callable=AsyncMock):
            report=await p.compare_plain(AsyncMock(),gemini_key='SECRET',gemini_model='test')
        self.assertEqual(generate.call_count,8)
        self.assertEqual([c.args[0] for c in generate.call_args_list],['gemini']*4+['groq']*4)
        self.assertEqual(report.count('Ответ модели: Реальная реплика'),8)
        self.assertIn('8/8',report)
        self.assertNotIn('SECRET',report)
        self.assertNotIn('4/4',report)

    async def test_failure_stops_provider_without_retries_or_substituting_models(self):
        with patch.object(p,'generate_plain',side_effect=GroqRequestError(http_code=429,
                provider_code='rate_limit_exceeded',reason='SECRET')) as generate, \
                patch.object(p.asyncio,'sleep',new_callable=AsyncMock):
            report=await p.compare_plain(AsyncMock(),gemini_key='FAKE',gemini_model='test')
        self.assertEqual(generate.call_count,2)
        self.assertEqual([c.args[0] for c in generate.call_args_list],['gemini','groq'])
        self.assertIn('2/8',report)
        self.assertIn('rate_limit_exceeded',report)
        self.assertNotIn('SECRET',report)
