"""Safety checks of the isolated Telegram test entrypoint without credentials."""
import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load_config():
    script = (ROOT / "test_bot.py").read_text(encoding="utf-8")
    tree = ast.parse(script)
    names = {"configure_test_runtime", "verify_test_bot_identity"}
    pieces = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
              and node.name in names]
    assert len(pieces) == 2
    ns = {"os": os, "Path": Path,
          "TEST_DATA_DIR": Path("/app/data/psychologia_live_test"),
          "EXPECTED_PRODUCTION_USERNAME": "psychologiaproclub_bot"}
    exec(compile(ast.Module(body=pieces, type_ignores=[]), "test_bot.py", "exec"), ns)
    return ns, tree


class IsolatedTelegramTestModeTests(unittest.TestCase):
    def setUp(self):
        self.config, self.tree = load_config()

    def base(self):
        # Only secrets from the preexisting Bothost service are required.
        return {
            "BOT_TOKEN": "TEST_TOKEN_NOT_FOR_NETWORK",
            "GEMINI_API_KEY": "TEST_GEMINI_KEY",
        }

    def test_missing_optional_variables_uses_safe_defaults(self):
        with patch.dict(os.environ, self.base(), clear=True):
            username, testers = self.config["configure_test_runtime"]()
            self.assertEqual(username, "psychologiaproclub_bot")
            self.assertEqual(testers, {885005330})

    def test_explicitly_disabling_test_mode_is_rejected(self):
        opts = self.base()
        opts["BOT_TEST_MODE"] = "0"
        with patch.dict(os.environ, opts, clear=True):
            with self.assertRaises(RuntimeError):
                self.config["configure_test_runtime"]()

    def test_existing_bot_identity_override_must_match(self):
        opts = self.base()
        opts["TEST_BOT_USERNAME"] = "AnotherTest_bot"
        with patch.dict(os.environ, opts, clear=True):
            with self.assertRaises(RuntimeError):
                self.config["configure_test_runtime"]()
        opts["TEST_BOT_USERNAME"] = "@PsychologiaProClub_bot"
        with patch.dict(os.environ, opts, clear=True):
            username, testers = self.config["configure_test_runtime"]()
            self.assertEqual(username, "psychologiaproclub_bot")
            self.assertEqual(testers, {885005330})

    def test_wrong_flag_or_testers_are_rejected(self):
        for name, value in (
            ("TEST_BOT_USE_EXISTING_IDENTITY", "0"),
            ("TEST_BOT_ALLOWED_USER_IDS", "7563789785"),
            ("TEST_BOT_ALLOWED_USER_IDS", "885005330,7563789785"),
            ("TEST_BOT_ALLOWED_USER_IDS", "bad-id"),
        ):
            with self.subTest(name=name, value=value):
                opts = self.base()
                opts[name] = value
                with patch.dict(os.environ, opts, clear=True):
                    with self.assertRaises(RuntimeError):
                        self.config["configure_test_runtime"]()

    def test_bot_identity_mismatch_fails_before_polling(self):
        import asyncio
        from types import SimpleNamespace

        class FakeBot:
            def __init__(self, username):
                self.username = username
            async def get_me(self):
                return SimpleNamespace(username=self.username)

        with patch.dict(os.environ, self.base(), clear=True):
            identity = self.config["verify_test_bot_identity"]
            asyncio.run(identity(
                SimpleNamespace(bot=FakeBot("PsychologiaProClub_bot")),
                "psychologiaproclub_bot"
            ))
            with self.assertRaises(RuntimeError):
                asyncio.run(identity(
                    SimpleNamespace(bot=FakeBot("SomeOther_bot")),
                    "psychologiaproclub_bot"
                ))

    def test_data_folders_are_separate_from_production(self):
        with patch.dict(os.environ, self.base(), clear=True):
            name, ids = self.config["configure_test_runtime"]()
            self.assertEqual(name, "psychologiaproclub_bot")
            self.assertEqual(ids, {885005330})
            self.assertEqual(os.environ["BOT_DATA_DIR"],
                             "/app/data/psychologia_live_test")
            self.assertEqual(os.environ["MEMORY_DB_PATH"],
                             "/app/data/psychologia_live_test/memory.db")
            self.assertNotEqual(os.environ["MEMORY_DB_PATH"], "/app/data/memory.db")

    def test_explicit_data_override_cannot_point_to_production(self):
        opts = self.base()
        opts["MEMORY_DB_PATH"] = "/app/data/memory.db"
        with patch.dict(os.environ, opts, clear=True):
            with self.assertRaises(RuntimeError):
                self.config["configure_test_runtime"]()

    def test_missing_testers_and_gemini_key_fail_closed(self):
        for name in ("BOT_TOKEN", "GEMINI_API_KEY"):
            with self.subTest(name=name):
                opts = self.base()
                opts.pop(name)
                with patch.dict(os.environ, opts, clear=True):
                    with self.assertRaises(RuntimeError):
                        self.config["configure_test_runtime"]()

    def test_no_shared_group_handlers_or_normal_bot_main(self):
        funcs = {node.name: node for node in self.tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        build = ast.unparse(funcs["create_test_application"])
        self.assertIn("filters.ChatType.PRIVATE", build)
        self.assertIn("filters.User", build)
        self.assertIn("bot.private_message", build)
        self.assertNotIn("filters.ChatType.GROUPS", build)
        self.assertNotIn("bot.main(", build)
        self.assertNotIn("bot.save_message", build)

    def test_network_check_requires_owner_private_filter(self):
        import ast
        code = (ROOT / "test_bot.py").read_text(encoding="utf-8")
        tree = ast.parse(code)
        builder = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
                       and x.name == "create_test_application")
        body = ast.unparse(builder)
        self.assertIn("'network_check': network_check", body)
        self.assertIn("filters.ChatType.PRIVATE", body)
        self.assertIn("filters.User", body)

    def test_country_diagnostic_success_and_untrusted_response(self):
        import io
        import urllib.request

        code = (ROOT / "test_bot.py").read_text(encoding="utf-8")
        tree = ast.parse(code)
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
                  and x.name == "detect_server_country")
        namespace = {"urllib": urllib}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "test_bot.py", "exec"),
             namespace)
        detect = namespace["detect_server_country"]

        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"NL\n")) as api:
            self.assertEqual(detect(), "NL")
            self.assertEqual(api.call_args.kwargs["timeout"], 8)
            self.assertEqual(api.call_args.args[0].full_url,
                             "https://ipinfo.io/country")
        for invalid in (b"", b"US,NL", b"Unknown", b"NL\nGB", b"nl"):
            with self.subTest(invalid=invalid):
                with patch("urllib.request.urlopen",
                           return_value=io.BytesIO(invalid)):
                    with self.assertRaises(ValueError):
                        detect()

    def test_gemini_check_only_in_private_tester_command_set(self):
        code = (ROOT / "test_bot.py").read_text(encoding="utf-8")
        funcs = {node.name: node for node in ast.parse(code).body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertIn("probe_gemini_models", funcs)
        builder = ast.unparse(funcs["create_test_application"])
        self.assertIn("'gemini_check': gemini_check", builder)
        self.assertIn("filters.ChatType.PRIVATE", builder)
        self.assertIn("filters.User", builder)

    def test_gemini_probe_is_metadata_only_and_keeps_key_out_of_result(self):
        import io
        import json
        import urllib
        import urllib.error
        import urllib.request
        code = (ROOT / "test_bot.py").read_text(encoding="utf-8")
        fn = next(x for x in ast.parse(code).body
                  if isinstance(x, ast.FunctionDef)
                  and x.name == "probe_gemini_models")
        ns = {"os": os, "urllib": urllib, "json": json}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "test_bot.py", "exec"), ns)
        probe = ns["probe_gemini_models"]
        key = "NO_REAL_SECRET_UNIT_TEST_KEY"
        with patch.dict(os.environ, {"GEMINI_API_KEY": key}, clear=True):
            with patch("urllib.request.urlopen",
                       return_value=io.BytesIO(b'{"models":[]}')) as http:
                result = probe()
                self.assertIn("200 OK", result)
                self.assertNotIn(key, result)
                req = http.call_args.args[0]
                self.assertEqual(req.get_method(), "GET")
                self.assertEqual(
                    req.full_url,
                    "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1"
                )
                self.assertEqual(req.get_header("X-goog-api-key"), key)

            payload = json.dumps({"error": {
                "status": "FAILED_PRECONDITION",
                "message": "User location is not supported for the API use"
            }}).encode("utf-8")
            err = urllib.error.HTTPError(
                "https://generativelanguage.googleapis.com/v1beta/models",
                400, "Bad Request", {}, io.BytesIO(payload)
            )
            with patch("urllib.request.urlopen", side_effect=err):
                result = probe()
                self.assertIn("HTTP 400", result)
                self.assertIn("FAILED_PRECONDITION", result)
                self.assertIn("расположение", result)
                self.assertNotIn(key, result)

    def test_gemini_ping_minimal_post_and_safe_status(self):
        import io
        import json
        import socket
        import time
        import urllib
        import urllib.error
        import urllib.request

        tree = ast.parse((ROOT / "test_bot.py").read_text(encoding="utf-8"))
        funcs = {node.name: node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertIn("gemini_ping", funcs)
        build = ast.unparse(funcs["create_test_application"])
        self.assertIn("'gemini_ping': gemini_ping", build)
        self.assertIn("filters.ChatType.PRIVATE", build)
        self.assertIn("filters.User", build)
        ns = {"os": os, "urllib": urllib, "socket": socket,
              "json": json, "time": time}
        exec(compile(ast.Module(
            body=[funcs["probe_gemini_generation"]], type_ignores=[]
        ), "test_bot.py", "exec"), ns)
        probe = ns["probe_gemini_generation"]
        fake_key = "DO_NOT_EXPOSE_THIS_FAKE_KEY"
        with patch.dict(os.environ, {
                "GEMINI_MODEL": "gemini-3.8-flash",
                "GEMINI_API_KEY": fake_key,
        }, clear=True):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(
                b'{"candidates":[{"content":{"parts":[{"text":"OK"}]}}]}'
            )) as http:
                result = probe()
                self.assertIn("HTTP 200", result)
                self.assertNotIn(fake_key, result)
                req = http.call_args.args[0]
                self.assertEqual(req.get_method(), "POST")
                self.assertIn("gemini-3.8-flash:generateContent", req.full_url)
                self.assertEqual(http.call_args.kwargs["timeout"], 25)
                self.assertEqual(req.get_header("X-goog-api-key"), fake_key)
                body = json.loads(req.data.decode("utf-8"))
                self.assertEqual(body["generationConfig"]["maxOutputTokens"], 64)
                self.assertNotIn("синие кактусы", req.data.decode("utf-8"))

            with patch("urllib.request.urlopen", side_effect=socket.timeout()):
                result = probe()
                self.assertIn("Тайм-аут POST", result)
                self.assertNotIn(fake_key, result)

            err = urllib.error.HTTPError(
                "https://generativelanguage.googleapis.com/",
                404, "Not Found", {},
                io.BytesIO(b'{"error":{"status":"NOT_FOUND","message":"bad"}}')
            )
            with patch("urllib.request.urlopen", side_effect=err):
                result = probe()
                self.assertIn("HTTP 404", result)
                self.assertIn("NOT_FOUND", result)
                self.assertNotIn(fake_key, result)

    def test_groq_probe_owner_only_and_no_private_data(self):
        import io
        import json
        import socket
        import urllib
        import urllib.error
        import urllib.request
        code = (ROOT / "test_bot.py").read_text(encoding="utf-8")
        funcs = {node.name: node for node in ast.parse(code).body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertIn("probe_groq_metadata", funcs)
        build = ast.unparse(funcs["create_test_application"])
        self.assertIn("'groq_check': groq_check", build)
        self.assertIn("filters.ChatType.PRIVATE", build)
        self.assertIn("filters.User", build)
        namespace = {"os": os, "json": json, "socket": socket, "urllib": urllib}
        exec(compile(ast.Module(body=[funcs["probe_groq_metadata"]],
                                type_ignores=[]), "test_bot.py", "exec"), namespace)
        check = namespace["probe_groq_metadata"]
        key = "FAKE_TEST_KEY_DO_NOT_EXPOSE"
        env = {"GROQ_API_KEY": key}
        payload = b'{"data":[{"id":"openai/gpt-oss-20b"}]}'
        with patch.dict(os.environ, env, clear=True):
            with patch("urllib.request.urlopen",
                       return_value=io.BytesIO(payload)) as api:
                result = check()
                self.assertIn("HTTP 200", result)
                self.assertIn("модель openai/gpt-oss-20b: есть", result)
                request = api.call_args.args[0]
                self.assertEqual(request.get_method(), "GET")
                self.assertEqual(request.full_url,
                                 "https://api.groq.com/openai/v1/models")
                self.assertEqual(api.call_args.kwargs["timeout"], 12)
                self.assertEqual(request.get_header("Authorization"),
                                 "Bearer " + key)
                self.assertIsNone(request.data)
                self.assertNotIn(key, result)

            errors = [
                ({"Content-Type": "text/html", "Server": "cloudflare",
                  "cf-mitigated": "challenge"},
                 b"<html>Private " + key.encode() + b"</html>",
                 ("HTTP 403", "HTML", "Cloudflare", "защитная проверка")),
                ({"Content-Type": "application/json"},
                 b'{"error":{"code":"model_permission_blocked_org"}}',
                 ("HTTP 403", "model_permission_blocked_org")),
            ]
            for headers, body, expected in errors:
                with self.subTest(headers=headers):
                    err = urllib.error.HTTPError(
                        "https://api.groq.com/openai/v1/models",
                        403, "Forbidden", headers, io.BytesIO(body)
                    )
                    with patch("urllib.request.urlopen", side_effect=err):
                        result = check()
                    for piece in expected:
                        self.assertIn(piece, result)
                    self.assertNotIn(key, result)

            with patch("urllib.request.urlopen", side_effect=socket.timeout()):
                self.assertIn("12 секунд", check())

    def test_bot_identity_checked_before_polling(self):
        build = ast.unparse(next(node for node in self.tree.body
                                 if isinstance(node, ast.FunctionDef)
                                 and node.name == "create_test_application"))
        self.assertIn("post_init", build)
        self.assertIn("verify_test_bot_identity", build)

    def test_bot_database_path_reads_environment_with_legacy_default(self):
        bot_code = (ROOT / "bot.py").read_text(encoding="utf-8")
        storage_code = (ROOT / "memory/storage.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("BOT_DATA_DIR", "/app/data")', bot_code)
        self.assertIn('os.environ.get("MEMORY_DB_PATH", "/app/data/memory.db")',
                      storage_code)


if __name__ == "__main__":
    unittest.main()
