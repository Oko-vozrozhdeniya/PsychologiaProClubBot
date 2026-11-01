"""The real diagnostic handler must obey private-dialog consent, too."""
from datetime import datetime, timezone
import json
import unittest
from unittest.mock import Mock

from memory import lifecycle, storage
import test_bot_memory_round_trip as helpers
import test_retention_operation as fixtures


class TestAiConsentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = fixtures.ActualPrivateRetentionTests(
            'test_real_handler_excludes_expired_pairs_before_provider_and_keeps_own_scope')
        self.fixture.setUp()
        self.addCleanup(self.fixture.fixture.doCleanups)
        self.ns = self.fixture.ns
        for name in ('is_owner', 'testai'):
            helpers.load_function(name, self.ns)
        self.ns['request_gemini'] = Mock(return_value='CONTROL_DIAGNOSTIC_REPLY')
        self.ns['read_dialog_state'] = Mock(return_value={'101': 2})
        self.history = [{'role': 'user', 'parts': [{'text': 'CONTROL_OWN_HISTORY'}]}]
        self.ns['read_recent_dialog'] = Mock(return_value=self.history)
        self.ns['create_error_record'] = Mock(return_value={'reason': 'CONTROL_ERROR'})
        self.update, self.context = self.fixture.update, self.fixture.context

    def consent(self, enabled):
        lifecycle.set_memory_consent(101, enabled, datetime.now(timezone.utc), db_path=self.fixture.db)

    async def invoke(self):
        await self.ns['testai'](self.update, self.context)

    async def test_missing_consent_blocks_history_read_and_both_provider_calls(self):
        with storage._connection(self.fixture.db) as db:
            db.execute('DELETE FROM memory_consents')
        await self.invoke()
        self.ns['read_dialog_state'].assert_not_called()
        self.ns['read_recent_dialog'].assert_not_called()
        self.ns['request_gemini'].assert_not_called()
        self.assertIn('/memory_on', self.update.effective_message.reply_text.await_args.args[0])

    async def test_revoked_consent_blocks_diagnostic_despite_owner_role(self):
        self.consent(False)
        await self.invoke()
        self.ns['read_recent_dialog'].assert_not_called()
        self.ns['request_gemini'].assert_not_called()

    async def test_opt_in_runs_comparison_without_changing_archive_or_consent(self):
        self.consent(True)
        archive = self.fixture.root / 'private_history.jsonl'
        archive.write_text(json.dumps({'control': 'PRESERVE'}) + '\n')
        before = archive.read_bytes()
        await self.invoke()
        self.ns['read_recent_dialog'].assert_called_once_with(101, 2)
        calls = self.ns['request_gemini'].call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[2], [])
        self.assertEqual(calls[1].args[2], self.history)
        self.assertEqual(archive.read_bytes(), before)
        self.assertTrue(lifecycle.has_memory_consent(101, db_path=self.fixture.db))

    async def test_revocation_during_first_request_blocks_second_history_request(self):
        self.consent(True)
        def first_request(*args):
            self.consent(False)
            return 'CONTROL_FIRST_REPLY'
        self.ns['request_gemini'].side_effect = first_request
        await self.invoke()
        self.assertEqual(self.ns['request_gemini'].call_count, 1)
        self.assertEqual(self.ns['request_gemini'].call_args.args[2], [])
        self.ns['send_long_message'].assert_not_awaited()

    async def test_consent_read_failure_blocks_history_and_provider(self):
        self.ns['personal_lifecycle'].has_memory_consent = Mock(side_effect=OSError('CONTROL_LOOKUP_FAILURE'))
        await self.invoke()
        self.ns['read_recent_dialog'].assert_not_called()
        self.ns['request_gemini'].assert_not_called()
        self.ns['send_error_response'].assert_awaited_once()

    async def test_second_consent_read_failure_does_not_send_any_request(self):
        self.ns['personal_lifecycle'].has_memory_consent = Mock(side_effect=[True, OSError('CONTROL_LOOKUP_FAILURE')])
        await self.invoke()
        self.ns['request_gemini'].assert_not_called()
        self.ns['send_error_response'].assert_awaited_once()

    async def test_nonowner_or_group_cannot_run_diagnostic(self):
        for user, chat_type in ((202, 'private'), (101, 'group')):
            with self.subTest(user=user, chat_type=chat_type):
                self.update.effective_user.id = user
                self.update.effective_chat.type = chat_type
                await self.invoke()
        self.ns['read_dialog_state'].assert_not_called()
        self.ns['request_gemini'].assert_not_called()


if __name__ == '__main__':
    unittest.main()
