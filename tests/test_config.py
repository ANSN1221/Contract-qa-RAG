"""config 的 .env 解析与配置校验测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from contract_qa.config import ConfigError, _parse_env_line, get_settings, load_env


class ParseLineTest(unittest.TestCase):
    def test_export_prefix(self):
        self.assertEqual(_parse_env_line("export FOO=bar"), ("FOO", "bar"))

    def test_inline_comment_stripped(self):
        self.assertEqual(_parse_env_line("FOO=bar  # comment"), ("FOO", "bar"))

    def test_hash_inside_value_kept(self):
        self.assertEqual(_parse_env_line("FOO=#fff"), ("FOO", "#fff"))

    def test_quoted_hash_kept(self):
        self.assertEqual(_parse_env_line('FOO="a # b"'), ("FOO", "a # b"))

    def test_blank_and_comment_skipped(self):
        self.assertIsNone(_parse_env_line(""))
        self.assertIsNone(_parse_env_line("   "))
        self.assertIsNone(_parse_env_line("# comment"))

    def test_bad_key_rejected(self):
        with self.assertRaises(ConfigError):
            _parse_env_line("BAD KEY=value")

    def test_unclosed_quote_rejected(self):
        with self.assertRaises(ConfigError):
            _parse_env_line('FOO="abc')


class LoadEnvFileTest(unittest.TestCase):
    def test_full_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text(
                "export A=1\nB=\"x y\"\nC=v  # note\nBAD LINE\n",
                encoding="utf-8",
            )
            values = load_env(env)
        self.assertEqual(values, {"A": "1", "B": "x y", "C": "v"})

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_env(Path(tmp) / "nope.env"), {})


class EnvFloatTest(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("BM25_WEIGHT")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("BM25_WEIGHT", None)
        else:
            os.environ["BM25_WEIGHT"] = self._old

    def test_weight_read_from_env(self):
        os.environ["BM25_WEIGHT"] = "0.7"
        self.assertEqual(get_settings().bm25_weight, 0.7)

    def test_bad_weight_raises_config_error(self):
        os.environ["BM25_WEIGHT"] = "abc"
        with self.assertRaises(ConfigError):
            get_settings()

    def test_negative_weight_raises(self):
        os.environ["BM25_WEIGHT"] = "-1"
        with self.assertRaises(ConfigError):
            get_settings()


if __name__ == "__main__":
    unittest.main()
