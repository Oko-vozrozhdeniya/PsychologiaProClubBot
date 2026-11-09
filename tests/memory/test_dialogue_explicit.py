import json
import unittest
from unittest.mock import patch, AsyncMock
from tools import dialogue_explicit as gate


class ExplicitGateTests(unittest.TestCase):
    def test_direct_requires_current_quote_but_does_not_claim_semantic_proof(self):
        for evidence in ('', 'цитата из другого сообщения'):
            with self.assertRaises(ValueError):
                gate.parse_decision(json.dumps(dict(kind='direct_end',evidence=evidence)), 'До свидания')
        self.assertEqual(gate.parse_decision('{"kind":"direct_end","evidence":"До свидания"}', 'До свидания')['kind'], 'direct_end')

    def test_ambiguous_or_malformed_decisions_do_not_silently_become_farewells(self):
        for raw in ('{}','{"kind":"close","evidence":""}',
                    '{"kind":"continue","kind":"direct_end","evidence":"bye"}',
                    '{"kind":"indirect_stop","evidence":"text"}'):
            with self.assertRaises((ValueError, TypeError)):
                gate.parse_decision(raw, 'bye')

    def test_composer_gets_original_conversation_not_a_reviewer_dialogue(self):
        question='Мне достаточно';recent=[dict(role='user',parts=[dict(text='История')])]
        options=dict(provider='gemini',gemini_key='KEY',gemini_model='test')
        with patch.object(gate,'request_plain_text',return_value='Вопрос') as request:
            gate.compose(question,recent,dict(kind='indirect_stop',evidence=''),**options)
        self.assertEqual(request.call_args.args,('gemini',question,recent))
        self.assertIn(gate.REPLY['indirect_stop'],request.call_args.kwargs['instruction'])
        self.assertNotIn('direct_end',request.call_args.kwargs['instruction'])


class ExplicitHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_bad_classifier_stops_before_answer_generation(self):
        with patch.object(gate,'classify',side_effect=ValueError('bad')) as classify, \
             patch.object(gate,'compose') as compose:
            report=await gate.compare_explicit(AsyncMock(),provider='gemini',gemini_key='KEY',gemini_model='test')
        self.assertEqual(classify.call_count,1)
        compose.assert_not_called()
        self.assertIn('1/8',report)
        self.assertNotIn('KEY',report)
