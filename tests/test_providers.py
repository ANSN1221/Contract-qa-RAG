"""多供应商切换的单元测试：不联网，全部 mock post_json。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from contract_qa.config import LLMProvider, _read_llm_providers
from contract_qa.llm import ApiError, chat, chat_with_tools


def _settings(providers) -> mock.Mock:
    settings = mock.Mock()
    settings.llm_providers = tuple(providers)
    settings.has_llm = any(provider.ready for provider in providers)
    settings.llm_base_url = "https://primary.example/v1"
    settings.llm_api_key = "k"
    settings.llm_model = "m"
    return settings


def _provider(name: str, url: str, key: str = "k", model: str = "m") -> LLMProvider:
    return LLMProvider(name=name, base_url=url, api_key=key, model=model)


def _ok_response(content: str = "done") -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


class FailoverTest(unittest.TestCase):
    def test_primary_balance_error_falls_back(self):
        primary = _provider("primary", "https://a.example/v1")
        backup = _provider("backup", "https://b.example/v1", model="m2")

        def fake_post(url, payload, api_key):
            if "a.example" in url:
                raise ApiError("HTTP 402 Insufficient Balance", retryable=False)
            assert payload["model"] == "m2"  # 备用要用自己的模型名
            return _ok_response()

        with mock.patch("contract_qa.llm.post_json", side_effect=fake_post):
            result = chat([{"role": "user", "content": "hi"}], _settings([primary, backup]))
        self.assertEqual(result.content, "done")
        self.assertEqual(result.provider, "backup")

    def test_all_providers_fail_aggregates_reasons(self):
        providers = [
            _provider("primary", "https://a.example/v1"),
            _provider("backup", "https://b.example/v1"),
        ]
        with mock.patch(
            "contract_qa.llm.post_json", side_effect=ApiError("HTTP 402", retryable=False)
        ):
            with self.assertRaises(ApiError) as ctx:
                chat([{"role": "user", "content": "hi"}], _settings(providers))
        message = str(ctx.exception)
        self.assertIn("primary", message)
        self.assertIn("backup", message)
        self.assertFalse(ctx.exception.retryable)

    def test_legacy_settings_without_providers(self):
        settings = _settings([])
        del settings.llm_providers  # 模拟旧配置对象
        settings.has_llm = True
        with mock.patch("contract_qa.llm.post_json", return_value=_ok_response()):
            result = chat([{"role": "user", "content": "hi"}], settings)
        self.assertEqual(result.provider, "primary")

    def test_tool_call_uses_backup_and_keeps_calls(self):
        primary = _provider("primary", "https://a.example/v1")
        backup = _provider("backup", "https://b.example/v1")

        def fake_post(url, payload, api_key):
            if "a.example" in url:
                raise ApiError("HTTP 401", retryable=False)
            self.assertIn("tools", payload)
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {"name": "search_clauses", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        with mock.patch("contract_qa.llm.post_json", side_effect=fake_post):
            result = chat_with_tools(
                [{"role": "user", "content": "hi"}], [{}], _settings([primary, backup])
            )
        self.assertEqual(result.provider, "backup")
        self.assertEqual(result.tool_calls[0].name, "search_clauses")


def _clean_env(**extra) -> mock._patch:
    """把配置读取会碰到的键全部清空，避免读到本机 .env 或其他测试的残留。"""
    base = {"LLM_NAME": "", "LLM_BASE_URL": "", "LLM_API_KEY": "", "LLM_MODEL": ""}
    for index in range(1, 6):
        prefix = "LLM_FALLBACK" if index == 1 else f"LLM_FALLBACK_{index}"
        for suffix in ("NAME", "BASE_URL", "API_KEY", "MODEL"):
            base[f"{prefix}_{suffix}"] = ""
    base.update(extra)
    return mock.patch.dict(os.environ, base, clear=False)


class ProviderConfigTest(unittest.TestCase):
    def test_reads_primary_and_two_backups(self):
        env = {
            "LLM_API_KEY": "k1",
            "LLM_BASE_URL": "https://a.example/v1",
            "LLM_MODEL": "m1",
            "LLM_FALLBACK_API_KEY": "k2",
            "LLM_FALLBACK_BASE_URL": "https://b.example/v1",
            "LLM_FALLBACK_MODEL": "m2",
            "LLM_FALLBACK_2_API_KEY": "k3",
            "LLM_FALLBACK_2_BASE_URL": "https://c.example/v1",
            "LLM_FALLBACK_2_NAME": "third",
        }
        with _clean_env(**env):
            providers = _read_llm_providers()
        self.assertEqual(len(providers), 3)
        self.assertEqual(providers[0].model, "m1")
        self.assertEqual(providers[1].base_url, "https://b.example/v1")
        self.assertEqual(providers[2].name, "third")

    def test_skips_backup_without_key(self):
        env = {
            "LLM_API_KEY": "k1",
            "LLM_FALLBACK_BASE_URL": "https://b.example/v1",  # 只有地址没密钥
        }
        with _clean_env(**env):
            providers = _read_llm_providers()
        self.assertEqual(len(providers), 1)

    def test_no_key_at_all_returns_empty(self):
        with _clean_env():
            providers = _read_llm_providers()
        self.assertEqual(providers, ())


if __name__ == "__main__":
    unittest.main()
