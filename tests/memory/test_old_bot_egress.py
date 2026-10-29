"""One-off, credential-free validation of legacy bot IP command."""
import ast
import io
import ipaddress
import urllib.request
import unittest
from pathlib import Path
from unittest.mock import patch


class OutboundIpTests(unittest.TestCase):
    def test_diagnostic_safe_and_owner_only(self):
        source = (Path(__file__).resolve().parents[2] / "bot.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        handler = ast.unparse(functions["main"])
        self.assertIn("'ip_check'", handler)
        self.assertIn("filters.ChatType.PRIVATE", handler)
        self.assertIn("filters.User(user_id=OWNER_ID)", handler)
        code = ast.unparse(next(node for node in tree.body
                              if isinstance(node, ast.AsyncFunctionDef) and node.name == "ip_check"))
        self.assertIn("is_owner(update)", code)
        ns = {"urllib": urllib, "ipaddress": ipaddress}
        node = functions["read_public_ip"]
        exec(compile(ast.Module(body=[node], type_ignores=[]), "bot.py", "exec"), ns)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"8.8.8.8")) as mock:
            self.assertEqual(ns["read_public_ip"](), "8.8.8.8")
            self.assertEqual(mock.call_args.args[0], "https://api.ipify.org")
            self.assertEqual(mock.call_args.kwargs["timeout"], 10)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"127.0.0.1")):
            with self.assertRaises(ValueError):
                ns["read_public_ip"]()


if __name__ == "__main__":
    unittest.main()
