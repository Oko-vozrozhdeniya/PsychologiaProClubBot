import ast
import asyncio
import os
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from tools import dialogue_lab as lab


class DialogueLabCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root/'bot.py').read_text())
        selected = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                    and n.name in ('dialogue_lab_command','run_dialogue_lab')]
        self.sent = AsyncMock()
        self.ns = dict(asyncio=asyncio, os=os, json=json, Update=object,
            ContextTypes=SimpleNamespace(DEFAULT_TYPE=object), GEMINI_API_KEY='FAKE GEMINI',
            GEMINI_MODEL='test', is_owner=lambda u:u.effective_user.id==1,
            send_long_message=self.sent)
        exec(compile(ast.Module(body=selected,type_ignores=[]),'bot.py','exec'),self.ns)
        self.message=SimpleNamespace(reply_text=AsyncMock())
        self.update=SimpleNamespace(effective_message=self.message,
            effective_chat=SimpleNamespace(type='private'),effective_user=SimpleNamespace(id=1))
        self.jobs=[]
        self.app=SimpleNamespace(bot_data={'test_ai_mode':'auto'},create_task=self.jobs.append)
        self.context=SimpleNamespace(application=self.app)
        self.env=patch.dict(os.environ,{'GROQ_API_KEY':'FAKE GROQ'})
        self.env.start();self.addCleanup(self.env.stop)

    async def test_other_users_and_groups_cannot_launch_paid_requests(self):
        for user,chat in ((2,'private'),(1,'group')):
            self.update.effective_user.id=user;self.update.effective_chat.type=chat
            await self.ns['dialogue_lab_command'](self.update,self.context)
        self.assertEqual(self.jobs,[])
        self.message.reply_text.assert_not_awaited()

    async def test_missing_keys_cannot_launch(self):
        self.ns['GEMINI_API_KEY']=''
        await self.ns['dialogue_lab_command'](self.update,self.context)
        self.assertEqual(self.jobs,[])

    async def test_background_run_is_bounded_rejects_duplicate_and_preserves_mode(self):
        with patch.object(lab,'run_case',side_effect=lambda p,c,*a,**kw:c[3]) as run:
            await self.ns['dialogue_lab_command'](self.update,self.context)
            await self.ns['dialogue_lab_command'](self.update,self.context)
            self.assertEqual(len(self.jobs),1)
            self.assertTrue(self.app.bot_data['dialogue_lab_running'])
            await self.jobs.pop()
        self.assertEqual(run.call_count,16)
        self.assertNotIn('dialogue_lab_running',self.app.bot_data)
        self.assertEqual(self.app.bot_data['test_ai_mode'],'auto')
        text=str(self.sent.await_args_list)
        self.assertIn('4/4',text)
        self.assertNotIn('FAKE GEMINI',text)
        self.assertNotIn('FAKE GROQ',text)

    async def test_provider_failures_report_only_class_and_clear_running_flag(self):
        with patch.object(lab,'run_case',side_effect=RuntimeError('SECRET RAW ERROR')):
            await self.ns['dialogue_lab_command'](self.update,self.context)
            await self.jobs.pop()
        text=str(self.sent.await_args_list)
        self.assertIn('RuntimeError',text)
        self.assertNotIn('SECRET RAW ERROR',text)
        self.assertNotIn('dialogue_lab_running',self.app.bot_data)

    async def test_groq_short_run_makes_only_four_candidate_calls(self):
        self.context.args=['groq']
        with patch.object(lab,'run_case',side_effect=lambda p,c,*a,**kw:c[3]) as run:
            await self.ns['dialogue_lab_command'](self.update,self.context)
            await self.jobs.pop()
        self.assertEqual(run.call_count,4)
        self.assertTrue(all(c.args[0]=='candidate' and c.args[2]=='groq' for c in run.call_args_list))

    async def test_cancellation_clears_flag(self):
        async def cancelled(*a,**k):
            raise asyncio.CancelledError
        await self.ns['dialogue_lab_command'](self.update,self.context)
        with patch.object(asyncio,'to_thread',side_effect=cancelled):
            with self.assertRaises(asyncio.CancelledError):
                await self.jobs.pop()
        self.assertNotIn('dialogue_lab_running',self.app.bot_data)


    async def test_verification_runs_in_background_without_gemini_and_always_clears_flag(self):
        self.context.args=['verify']
        self.ns['GEMINI_API_KEY']=''
        with patch.object(lab,'verify_profiles',new_callable=AsyncMock) as verify:
            await self.ns['dialogue_lab_command'](self.update,self.context)
            self.assertEqual(len(self.jobs),1)
            await self.jobs.pop()
        verify.assert_awaited_once()
        self.assertEqual(self.app.bot_data,{'test_ai_mode':'auto'})
        self.assertIn('28',str(self.message.reply_text.await_args_list))

    async def test_context_command_is_owner_background_task_with_only_groq_key(self):
        self.context.args=['context']
        self.ns['GEMINI_API_KEY']=''
        with patch.object(lab,'verify_profiles',new_callable=AsyncMock) as verify:
            await self.ns['dialogue_lab_command'](self.update,self.context)
            await self.jobs.pop()
        self.assertTrue(verify.await_args.kwargs['decomposed'])
        self.assertEqual(self.app.bot_data,{'test_ai_mode':'auto'})

    async def test_replies_command_returns_complete_file_and_clears_flag_on_upload_failure(self):
        from tools import dialogue_replies as replies
        self.context.args=['replies']
        self.ns['GEMINI_API_KEY']=''
        self.message.reply_document=AsyncMock(side_effect=RuntimeError('upload failure'))
        with patch.object(replies,'compare_replies',new_callable=AsyncMock,return_value='Полный отчёт') as compare:
            await self.ns['dialogue_lab_command'](self.update,self.context)
            with self.assertRaises(RuntimeError):
                await self.jobs.pop()
        compare.assert_awaited_once()
        self.assertEqual(self.app.bot_data,{'test_ai_mode':'auto'})
        kwargs=self.message.reply_document.await_args.kwargs
        self.assertEqual(kwargs['filename'],'dialogue-replies.txt')
        self.assertEqual(kwargs['document'].getvalue().decode(),'Полный отчёт')

    async def test_minimal_control_is_owner_background_task_with_separate_report(self):
        from tools import dialogue_replies as replies
        self.context.args=['minimal']
        self.ns['GEMINI_API_KEY']=''
        self.message.reply_document=AsyncMock()
        with patch.object(replies,'compare_replies',new_callable=AsyncMock,return_value='Короткий отчёт') as compare:
            await self.ns['dialogue_lab_command'](self.update,self.context)
            await self.jobs.pop()
        self.assertTrue(compare.await_args.kwargs['minimal'])
        self.assertEqual(self.message.reply_document.await_args.kwargs['filename'],'dialogue-minimal.txt')
        self.assertEqual(self.app.bot_data,{'test_ai_mode':'auto'})
