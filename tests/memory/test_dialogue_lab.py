"""Offline contracts only: these never assert that a live model understood a phrase."""
import contextlib
import io
import json
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
