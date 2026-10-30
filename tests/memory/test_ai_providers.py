"""Provider routing tests: no live keys, network, Telegram or Gemini."""
import ast
import io
import json
import os
import socket
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from memory.ai_providers import (
    GroqRequestError, request_groq, request_personal_ai
)

ROOT = Path(__file__).resolve().parents[2]
KEY = "FAKE_KEY_MUST_NEVER_LEAK_87654321"


class GroqProviderTests(unittest.TestCase):
    def env(self, **extra):
        data = {"GROQ_API_KEY": KEY, "GROQ_MODEL": "openai/gpt-oss-20b"}
        data.update(extra)
        return patch.dict(os.environ, data, clear=True)

    def test_successful_request_uses_documented_endpoint_and_model(self):
        payload = {"choices": [{"message": {"content": "Привет из Groq!"}}]}
        with self.env(), patch(
            "urllib.request.urlopen",
            return_value=io.BytesIO(json.dumps(payload).encode())
        ) as fetch:
            result = request_groq(
                "Привет!", history=[
                    {"role": "user", "parts": [{"text": "Что такое память?"}]},
                    {"role": "model", "parts": [{"text": "Она хранит факты."}]}
                ]
            )
            self.assertEqual(result, "Привет из Groq!")
            self.assertEqual(fetch.call_args.kwargs["timeout"], 30)
            req = fetch.call_args.args[0]
            self.assertEqual(req.get_method(), "POST")
            self.assertEqual(req.full_url,
                             "https://api.groq.com/openai/v1/chat/completions")
            self.assertEqual(req.get_header("Authorization"), "Bearer " + KEY)
            self.assertEqual(req.get_header("User-agent"), "PsychologiaProClubBot/1.0")
            body = json.loads(req.data.decode("utf-8"))
            self.assertEqual(body["model"], "openai/gpt-oss-20b")
            self.assertEqual([m["role"] for m in body["messages"]],
                             ["system", "user", "assistant", "user"])
            self.assertEqual(body["messages"][-1]["content"], "Привет!")
            self.assertNotIn(KEY, req.data.decode())

    def test_groq_receives_authorized_memory_only_if_supplied(self):
        ok = {"choices": [{"message": {"content": "OK"}}]}
        with self.env(), patch("urllib.request.urlopen",
                               return_value=io.BytesIO(json.dumps(ok).encode())) as fetch:
            request_groq("Что я люблю?", history=[], personal_memory_context="")
            sent = json.loads(fetch.call_args.args[0].data)["messages"]
            self.assertNotIn("красные кактусы", json.dumps(sent, ensure_ascii=False))
        with self.env(), patch("urllib.request.urlopen",
                               return_value=io.BytesIO(json.dumps(ok).encode())) as fetch:
            request_groq("Что я люблю?", history=[],
                         personal_memory_context="Я люблю красные кактусы")
            sent = json.loads(fetch.call_args.args[0].data)["messages"]
            self.assertIn("красные кактусы", json.dumps(sent, ensure_ascii=False))

    def test_explicit_extraction_instruction_uses_existing_client_without_history(self):
        response = {"choices": [{"message": {"content": "[]"}}]}
        with self.env(), patch("urllib.request.urlopen",
                               return_value=io.BytesIO(json.dumps(response).encode())) as fetch:
            self.assertEqual(request_groq("Сообщение пользователя",
                             system_instruction="Верни только JSON-массив фактов"), "[]")
            sent = json.loads(fetch.call_args.args[0].data)["messages"]
        self.assertEqual(sent, [
            {"role": "system", "content": "Верни только JSON-массив фактов"},
            {"role": "user", "content": "Сообщение пользователя"},
        ])

    def test_invalid_extraction_instruction_fails_before_network(self):
        for instruction in ("", "  ", False, 123, []):
            with self.subTest(instruction=instruction), self.env(), patch("urllib.request.urlopen") as fetch:
                with self.assertRaises(GroqRequestError):
                    request_groq("Сообщение", system_instruction=instruction)
                fetch.assert_not_called()

    def test_key_missing_fails_without_network(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "urllib.request.urlopen"
        ) as fetch:
            with self.assertRaises(GroqRequestError):
                request_groq("Hi")
            fetch.assert_not_called()

    def test_http_503_categorized_and_error_body_not_exposed(self):
        failure = urllib.error.HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            503, "service down", {},
            io.BytesIO(b'{"secret": "FAKE_KEY_MUST_NEVER_LEAK_87654321"}'),
        )
        with self.env(), patch("urllib.request.urlopen", side_effect=failure):
            with self.assertRaises(GroqRequestError) as ctx:
                request_groq("Привет")
        self.assertEqual(ctx.exception.error_type, "unavailable")
        self.assertEqual(ctx.exception.http_code, 503)
        self.assertNotIn(KEY, ctx.exception.reason)

    def test_http_403_org_and_project_blocks_report_only_safe_codes(self):
        from memory.ai_providers import _describe_groq_http_error

        for code, expected in (
            ("model_permission_blocked_org", "организации"),
            ("model_permission_blocked_project", "проекта"),
        ):
            with self.subTest(code=code):
                payload = json.dumps({"error": {
                    "type": "permissions_error", "code": code,
                    "message": "Sensitive user token: " + KEY
                }}).encode("utf-8")
                err = urllib.error.HTTPError(
                    "https://api.groq.com/openai/v1/chat/completions",
                    403, "Forbidden", {}, io.BytesIO(payload)
                )
                with self.env(), patch("urllib.request.urlopen",
                                       side_effect=err):
                    with self.assertRaises(GroqRequestError) as raised:
                        request_groq("Test request")
                self.assertIn(code, raised.exception.reason)
                self.assertIn(expected, raised.exception.reason)
                self.assertNotIn(KEY, raised.exception.reason)

    def test_http_403_region_or_account_restriction_sanitized(self):
        scenarios = [
            ("Your location is not supported", "регион"),
            ("Your organization is restricted", "организации"),
            ("Code signup_blocked: reference " + KEY, "аккаунта"),
        ]
        for message, expected in scenarios:
            with self.subTest(message=message.split()[0]):
                payload = json.dumps({"error": {
                    "type": "permissions_error", "message": message
                }}).encode()
                err = urllib.error.HTTPError(
                    "https://api.groq.com/openai/v1/chat/completions",
                    403, "Forbidden", {}, io.BytesIO(payload)
                )
                with self.env(), patch("urllib.request.urlopen",
                                       side_effect=err):
                    with self.assertRaises(GroqRequestError) as raised:
                        request_groq("Test")
                self.assertIn(expected, raised.exception.reason)
                self.assertNotIn(KEY, raised.exception.reason)

    def test_html_403_and_malicious_code_do_not_echo_untrusted_text(self):
        cases = [
            (b"<html>Private " + KEY.encode() + b"</html>", "не является JSON"),
            (json.dumps({"error": {
                "type": "permissions_error", "code": KEY,
                "message": "Unrecognized " + KEY
            }}).encode(), "код=не указан"),
        ]
        for payload, expected in cases:
            with self.subTest(expected=expected):
                err = urllib.error.HTTPError(
                    "https://api.groq.com/openai/v1/chat/completions",
                    403, "Forbidden", {}, io.BytesIO(payload)
                )
                with self.env(), patch("urllib.request.urlopen",
                                       side_effect=err):
                    with self.assertRaises(GroqRequestError) as raised:
                        request_groq("Test")
                self.assertIn(expected, raised.exception.reason)
                self.assertNotIn(KEY, raised.exception.reason)

    def test_timeout_and_rate_limit_keep_provider_error(self):
        with self.env(), patch("urllib.request.urlopen",
                               side_effect=socket.timeout()):
            with self.assertRaises(GroqRequestError) as ctx:
                request_groq("Привет")
            self.assertEqual(ctx.exception.error_type, "timeout")
        err = urllib.error.HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            429, "slow down", {}, io.BytesIO(b"secret"),
        )
        with self.env(), patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(GroqRequestError) as ctx:
                request_groq("Привет")
            self.assertEqual(ctx.exception.error_type, "rate_limit")

    def test_direct_groq_skips_gemini_and_auto_falls_back_only_on_outage(self):
        success = {"choices": [{"message": {"content": "Ответ Groq"}}]}
        def fresh_response(*args, **kwargs):
            return io.BytesIO(json.dumps(success).encode())
        with self.env(), patch("urllib.request.urlopen",
                               side_effect=fresh_response):
            calls = []
            def gemini(*args):
                calls.append(args)
                raise AssertionError("Gemini must not be called in Groq mode")
            self.assertEqual(
                request_personal_ai("Привет", [], "", "groq", gemini),
                ("Ответ Groq", "Groq"),
            )
            self.assertFalse(calls)
            class FailingGemini(Exception):
                error_type = "unavailable"
            def unavailable(*args):
                calls.append(args)
                raise FailingGemini()
            self.assertEqual(
                request_personal_ai("Привет", [], "", "auto", unavailable),
                ("Ответ Groq", "Groq"),
            )
            self.assertEqual(calls[-1][-1], 15)

    def test_auto_no_fallback_on_unexpected_configuration_error(self):
        class ConfigurationError(Exception):
            error_type = "general"
        with self.env(), patch("urllib.request.urlopen") as fetch:
            with self.assertRaises(ConfigurationError):
                request_personal_ai(
                    "Hi", [], "", "auto",
                    lambda *args: (_ for _ in ()).throw(ConfigurationError())
                )
            fetch.assert_not_called()

    def test_auto_prefers_gemini_when_available(self):
        with self.env(), patch("urllib.request.urlopen") as fetch:
            outcome = request_personal_ai(
                "Привет", [], "", "auto",
                lambda *args: "Ответ Gemini"
            )
            self.assertEqual(outcome, ("Ответ Gemini", "Gemini"))
            fetch.assert_not_called()

    def test_private_test_commands_are_owner_only_and_no_group_handlers(self):
        text = (ROOT / "test_bot.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        build = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
                     and x.name == "create_test_application")
        contents = ast.unparse(build)
        for name in ("ai_groq", "ai_gemini", "ai_auto", "ai_mode"):
            self.assertIn(f"'{name}':", contents)
        self.assertIn("filters.ChatType.PRIVATE", contents)
        self.assertIn("filters.User", contents)
        self.assertIn("bot.private_message", contents)
        self.assertNotIn("filters.ChatType.GROUPS", contents)
        bot = (ROOT / "bot.py").read_text(encoding="utf-8")
        self.assertIn('bot_data.get("test_ai_mode", "gemini")', bot)
        self.assertIn("if provider == \"Gemini\"", bot)


class AiModeCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        code = (ROOT / "bot.py").read_text(encoding="utf-8")
        self.functions = {node.name: node for node in ast.parse(code).body
                          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.ns = {"OWNER_ID": 101, "os": os, "Update": object,
                   "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object)}
        for name in ("is_owner", "choose_ai_mode", "show_ai_mode"):
            exec(compile(ast.Module(body=[self.functions[name]], type_ignores=[]),
                         "bot.py", "exec"), self.ns)
        self.context = SimpleNamespace(application=SimpleNamespace(bot_data={}))

    async def invoke(self, command, *, user=101, chat_type="private"):
        reply = AsyncMock()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=user),
                                 effective_chat=SimpleNamespace(type=chat_type),
                                 effective_message=SimpleNamespace(text="/" + command, reply_text=reply))
        callback = "show_ai_mode" if command == "ai_mode" else "choose_ai_mode"
        await self.ns[callback](update, self.context)
        return reply

    async def test_owner_can_choose_auto_and_groq_in_actual_private_callbacks(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": KEY}, clear=True):
            await self.invoke("ai_auto")
            self.assertEqual(self.context.application.bot_data["test_ai_mode"], "auto")
            self.assertTrue(self.context.application.bot_data["test_show_provider"])
            reply = await self.invoke("ai_mode")
            self.assertIn("auto", reply.await_args.args[0])
            await self.invoke("ai_groq")
            self.assertEqual(self.context.application.bot_data["test_ai_mode"], "groq")

    async def test_participants_and_group_chats_cannot_change_or_read_owner_mode(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": KEY}, clear=True):
            for user, chat in ((202, "private"), (101, "group")):
                for command in ("ai_auto", "ai_groq", "ai_gemini", "ai_mode"):
                    with self.subTest(user=user, chat=chat, command=command):
                        reply = await self.invoke(command, user=user, chat_type=chat)
                        reply.assert_not_awaited()
        self.assertEqual(self.context.application.bot_data, {})

    async def test_backup_key_missing_does_not_change_the_selected_mode(self):
        self.context.application.bot_data["test_ai_mode"] = "gemini"
        with patch.dict(os.environ, {}, clear=True):
            for command in ("ai_auto", "ai_groq"):
                reply = await self.invoke(command)
                self.assertIn("GROQ_API_KEY", reply.await_args.args[0])
                self.assertEqual(self.context.application.bot_data["test_ai_mode"], "gemini")

    async def test_fresh_runtime_defaults_to_gemini_without_new_settings(self):
        reply = await self.invoke("ai_mode")
        self.assertIn("gemini", reply.await_args.args[0])

    async def test_main_entrypoint_registers_the_owner_mode_callbacks(self):
        registered = {}

        def add_handler(handler):
            if isinstance(handler, tuple):
                registered[handler[0]] = handler

        app = SimpleNamespace(add_handler=add_handler, run_polling=Mock())
        builder = MagicMock()
        builder.token.return_value = builder
        builder.build.return_value = app
        self.ns.update({
            "Application": SimpleNamespace(builder=lambda: builder), "TOKEN": "FAKE",
            "CommandHandler": lambda name, callback, **kw: (name, callback, kw.get("filters")),
            "MessageHandler": lambda *args: None, "filters": MagicMock(),
        })
        for name in ("start", "analyze", "reset_dialog", "remember_command", "memories_command",
                     "forget_command", "memory_on_command", "memory_off_command", "correct_command",
                     "change_command", "history_command", "diagnostics", "testai", "groq_ping",
                     "private_message", "save_message", "track_topic_changes"):
            self.ns[name] = Mock()
        exec(compile(ast.Module(body=[self.functions["main"]], type_ignores=[]), "bot.py", "exec"), self.ns)
        self.ns["main"]()
        for command in ("ai_gemini", "ai_groq", "ai_auto"):
            self.assertIs(registered[command][1], self.ns["choose_ai_mode"])
            self.assertIsNotNone(registered[command][2])
        self.assertIs(registered["ai_mode"][1], self.ns["show_ai_mode"])
        self.assertIsNotNone(registered["ai_mode"][2])


if __name__ == "__main__":
    unittest.main()
