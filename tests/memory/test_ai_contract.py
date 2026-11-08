"""Common tasks exercised through actual provider payloads; no live API keys."""
import io
import json
import os
import unittest
from unittest.mock import patch
import test_bot_memory_round_trip as harness
from memory import ai_contract as contract
from memory.ai_adapters import extraction_settings
from memory.ai_providers import request_groq
from memory.gemini_prompt_adapter import append_private_memory


class AIContractTests(unittest.TestCase):
    def setUp(self):
        self.bot = harness.ActualMemoryRoundTripTests(methodName='runTest')
        self.bot.setUp()
        self.addCleanup(self.bot.doCleanups)
        import logging
        self.bot.ns.update(logger=logging.getLogger(__name__), append_private_memory=append_private_memory)
        self.gemini = harness.load_function('request_gemini', self.bot.ns)

    def groq(self, content='OK', reason='stop', instruction=None, reply_mode='answer'):
        response = {'choices': [{'message': {'content': content}, 'finish_reason': reason}]}
        with patch.dict(os.environ, {'GROQ_API_KEY': 'fake', 'GROQ_MODEL': 'openai/gpt-oss-20b'}), patch(
                'urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as fetch:
            result = request_groq('JSON task', system_instruction=instruction, reply_mode=reply_mode)
        return result, json.loads(fetch.call_args.args[0].data)

    def test_both_chat_providers_read_the_same_live_instruction(self):
        with patch.object(contract, 'PRIVATE_INSTRUCTIONS', 'SHARED CHANGE MARKER'):
            self.gemini('Привет')
            gemini = self.bot.last_payload['system_instruction']['parts'][0]['text']
            _, groq = self.groq()
        self.assertEqual(gemini, groq['messages'][0]['content'])
        self.assertIn('SHARED CHANGE MARKER', gemini)

    def test_both_providers_generate_own_words_under_identical_contextual_rules(self):
        generated='Выберем, что сейчас важнее. Что хочется обсудить?'
        for mode in ('brief_continue','brief_ack','brief_close','brief_clarify'):
            body={'candidates':[{'content':{'parts':[{'text':generated}]}}]}
            with self.subTest(mode=mode), patch.object(self.bot.ns['urllib'].request,'urlopen',
                    return_value=io.BytesIO(json.dumps(body).encode())) as fetch:
                answer=self.gemini('Сообщение',reply_mode=mode)
                payload=json.loads(fetch.call_args.args[0].data)
            groq_answer,groq_payload=self.groq(generated,reply_mode=mode)
            instruction=payload['system_instruction']['parts'][0]['text']
            self.assertEqual(instruction,groq_payload['messages'][0]['content'])
            self.assertEqual(instruction,contract.instructions_for('private',reply_mode=mode))
            self.assertEqual(answer,generated)
            self.assertEqual(groq_answer,generated)
            self.assertEqual(payload['generationConfig']['maxOutputTokens'],1200)

    def test_fallback_preserves_guidance_and_actual_generated_provider_text(self):
        from memory.ai_providers import request_personal_ai
        from unittest.mock import Mock
        error=RuntimeError('temporary');error.error_type='timeout'
        gemini=Mock(side_effect=error)
        with patch('memory.ai_providers.request_groq',return_value='Собственный ответ резервной модели') as reserve:
            answer,provider=request_personal_ai('Сообщение',[], '', 'auto',gemini,reply_mode='brief_continue')
        self.assertEqual(answer,'Собственный ответ резервной модели')
        self.assertEqual(provider,'Groq')
        self.assertEqual(gemini.call_args.kwargs,{'reply_mode':'brief_continue'})
        self.assertEqual(reserve.call_args.kwargs,{'reply_mode':'brief_continue'})

    def test_unknown_guidance_cannot_call_provider_or_select_a_template(self):
        from memory.ai_providers import request_personal_ai
        from unittest.mock import Mock
        gemini=Mock()
        with self.assertRaises(ValueError):
            request_personal_ai('Сообщение',[], '', 'gemini',gemini,reply_mode='unknown')
        gemini.assert_not_called()

    def test_both_extractors_share_instruction_and_token_budget(self):
        with patch.object(contract, 'EXTRACTION_INSTRUCTIONS', 'JSON SHARED EXTRACTION'):
            self.bot.save(101, 8001, 'Я изучаю биологию')
            gemini = self.bot.last_payload
            _, groq = self.groq('{"memories":[],"relationships":[]}', instruction=contract.instructions_for('extraction'))
        self.assertEqual(gemini['system_instruction']['parts'][0]['text'], groq['messages'][0]['content'])
        self.assertEqual(gemini['generationConfig']['maxOutputTokens'], groq['max_completion_tokens'])
        self.assertEqual(gemini['generationConfig']['responseMimeType'], 'application/json')
        self.assertEqual(groq['response_format'], {'type': 'json_object'})
        self.assertNotIn('response_format', self.groq()[1])

    def test_archive_keeps_separate_task_and_never_receives_private_memory(self):
        self.gemini('Вопрос', archive_context='PUBLIC ARCHIVE', personal_memory_context='PRIVATE SECRET', reply_mode='brief_close')
        payload = json.dumps(self.bot.last_payload, ensure_ascii=False)
        self.assertNotIn('PRIVATE SECRET', payload)
        self.assertIn('PUBLIC ARCHIVE', payload)
        self.assertEqual(self.bot.last_payload['system_instruction']['parts'][0]['text'], contract.instructions_for('archive'))

    def test_groq_truncation_is_distinct_from_invalid_json(self):
        with self.assertRaises(contract.ExtractionTruncatedError):
            self.groq('{"memories":', reason='length', instruction=contract.instructions_for('extraction'))
        with self.assertRaises(contract.ExtractionFormatError) as raised:
            self.groq('not JSON', instruction=contract.instructions_for('extraction'))
        self.assertNotIsInstance(raised.exception, contract.ExtractionTruncatedError)

    def test_gemini_truncation_cannot_save_even_valid_json(self):
        response = {'candidates': [{'finishReason': 'MAX_TOKENS', 'content': {'parts': [{'text': '[]'}]}}]}
        self.bot.ns['urllib'].request.urlopen = lambda *a, **kw: io.BytesIO(json.dumps(response).encode())
        with self.assertRaises(contract.ExtractionTruncatedError):
            self.bot.save(101, 8002, 'Я изучаю биологию')
        self.assertEqual(self.bot.personal_lifecycle.list_live_memories(101, __import__('datetime').datetime.now(__import__('datetime').timezone.utc)), [])

    def test_parser_rejects_malformed_ambiguous_and_oversized_results(self):
        for value in ('', '```json\n[]\n```', 'text []', '{}', '{"memories":[],"relationships":[],"extra":1}',
                      '{"memories":[],"memories":[],"relationships":[]}', '[NaN]', ' '*65537):
            with self.subTest(value=value[:50]), self.assertRaises(contract.ExtractionFormatError):
                contract.parse_extraction(value)
        self.assertEqual(contract.parse_extraction('[]'), [])
        self.assertEqual(contract.parse_extraction('{"memories":[],"relationships":[]}'), dict(memories=[], relationships=[]))

    def test_failure_never_delivers_generated_persistence_claim(self):
        result = contract.finalize_memory_answer('Я обновил информацию', {'write_status': 'ошибка'})
        self.assertNotIn('обновил', result)
        for text in ('Я сохранил сведения', 'Информация обновлена', 'Запомнено', 'Я записал изменение'):
            self.assertIn('не удалось', contract.finalize_memory_answer(text, {'write_status': 'ошибка'}))
        self.assertEqual(contract.finalize_memory_answer('Полезный ответ', {'write_status': 'ошибка'}), 'Полезный ответ')
        self.assertIn('не удалось', result)
        self.assertEqual(contract.finalize_memory_answer('Ответ', {'write_status': 'выполнено'}), 'Ответ')

    def test_unknown_task_or_adapter_is_not_silently_downgraded(self):
        for f in (contract.instructions_for, extraction_settings):
            with self.assertRaises(ValueError):
                f('unsupported')

    def test_real_handler_delivers_and_archives_failure_instead_of_false_confirmation(self):
        import asyncio
        import test_retention_operation as retention
        from unittest.mock import Mock
        fixture = retention.ActualPrivateRetentionTests(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.fixture.doCleanups)
        fixture.ns['request_personal_ai'] = lambda *args: ('Я обновил информацию', 'Gemini')
        fixture.ns['remember_dialog_facts'] = Mock(side_effect=contract.ExtractionFormatError('Invalid extraction output'))
        asyncio.run(fixture.invoke())
        delivered = fixture.ns['send_long_message'].await_args.args[1]
        self.assertIn('не удалось', delivered)
        self.assertNotIn('обновил', delivered)
        history = fixture.ns['DIALOG_HISTORY_FILE'].read_text()
        self.assertNotIn('обновил', history)
        self.assertIn(delivered, history)
        self.assertEqual(fixture.records[-1]['write_error'], 'ExtractionFormatError')
        fixture.ns['send_error_response'].assert_not_awaited()
