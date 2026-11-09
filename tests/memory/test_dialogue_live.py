import asyncio
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from tools import dialogue_live as live

class LiveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.message=SimpleNamespace(reply_text=AsyncMock())
        self.app=SimpleNamespace(bot_data={})
        self.consent=AsyncMock(return_value=True)
        self.options=dict(consent=self.consent,gemini_key='KEY',gemini_model='test')
    async def enable(self):
        await live.command(self.message,self.app,['gemini'],1,**self.options)
    async def test_two_calls_preserve_multi_turn_history_and_do_not_change_ordinary_mode(self):
        await self.enable()
        self.app.bot_data['test_ai_mode']='auto'
        with patch.object(live,'classify',return_value=dict(kind='continue',evidence='')) as classify, \
             patch.object(live,'compose',return_value='Ответ'),patch.object(live.asyncio,'sleep',new_callable=AsyncMock):
            await live.handle(self.message,self.app,1,'Вопрос',**self.options)
            self.assertEqual(self.app.bot_data[live.KEY]['calls'],2)
            self.assertEqual(len(self.app.bot_data[live.KEY]['history']),2)
            await live.handle(self.message,self.app,1,'Продолжение',**self.options)
        self.assertEqual(self.app.bot_data['test_ai_mode'],'auto')
        self.assertEqual(self.app.bot_data[live.KEY]['calls'],4)
    async def test_other_user_cannot_use_owner_session(self):
        await self.enable()
        with patch.object(live,'classify') as classify:
            self.assertFalse(await live.handle(self.message,self.app,2,'Вопрос',**self.options))
            classify.assert_not_called()
    async def test_revocation_between_calls_prevents_composition_and_erases_history(self):
        await self.enable()
        self.consent.side_effect=[True,True,False]
        with patch.object(live,'classify',return_value=dict(kind='continue',evidence='')),patch.object(live,'compose') as compose:
            await live.handle(self.message,self.app,1,'Вопрос',**self.options)
        compose.assert_not_called()
        self.assertNotIn(live.KEY,self.app.bot_data)
    async def test_reset_keeps_budget_and_off_removes_session(self):
        await self.enable()
        self.app.bot_data[live.KEY].update(calls=12,history=[{'text':'private'}])
        await live.command(self.message,self.app,['reset'],1,**self.options)
        self.assertEqual(self.app.bot_data[live.KEY]['calls'],12)
        self.assertEqual(self.app.bot_data[live.KEY]['history'],[])
        await live.command(self.message,self.app,['off'],1,**self.options)
        self.assertNotIn(live.KEY,self.app.bot_data)
    async def test_budget_prevents_further_calls_and_failure_does_not_keep_turn(self):
        await self.enable()
        self.app.bot_data[live.KEY]['calls']=40
        with patch.object(live,'classify') as classify:
            await live.handle(self.message,self.app,1,'Вопрос',**self.options)
            classify.assert_not_called()
        self.app.bot_data[live.KEY]['calls']=0
        with patch.object(live,'classify',side_effect=ValueError('SECRET')):
            await live.handle(self.message,self.app,1,'Вопрос',**self.options)
        self.assertEqual(self.app.bot_data[live.KEY]['history'],[])
        self.assertNotIn('SECRET',self.message.reply_text.await_args.args[0])
    async def test_command_rejects_nonowner_and_groups_before_module_call(self):
        tree=ast.parse((Path(__file__).resolve().parents[2]/'bot.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='dialogue_live_command')
        ns=dict(Update=object,ContextTypes=SimpleNamespace(DEFAULT_TYPE=object),is_owner=lambda u:u.effective_user.id==1)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'bot.py','exec'),ns)
        with patch.object(live,'command',new_callable=AsyncMock) as command:
            for uid,kind in ((2,'private'),(1,'group')):
                await ns['dialogue_live_command'](SimpleNamespace(effective_user=SimpleNamespace(id=uid),effective_chat=SimpleNamespace(type=kind)),None)
            command.assert_not_awaited()
