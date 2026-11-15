"""llm 客户端的重试与响应校验测试：不联网。"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from contract_qa.config import Settings
from contract_qa.llm import ApiError, chat, embed_texts, post_json


def make_settings() -> Settings:
    return Settings(
        llm_api_key="k",
        llm_base_url="https://api.test/v1",
        llm_model="m",
        embedding_api_key="e",
        embedding_base_url="https://vec.test/v1",
        embedding_model="m",
        chroma_path=Path("chroma"),
        processed_dir=Path("data/processed"),
    )


class RetryTest(unittest.TestCase):
    """429 必须重试，400 必须立刻失败，网络错误重试到上限。"""

    @mock.patch("contract_qa.llm.time.sleep")  # 跳过真实退避
    @mock.patch("contract_qa.llm._post")
    def test_429_is_retried(self, mpost, _sleep):
        mpost.side_effect = [ApiError("HTTP 429", status=429, retryable=True), {"ok": 1}]
        self.assertEqual(post_json("http://x", {}, "k"), {"ok": 1})
        self.assertEqual(mpost.call_count, 2)

    @mock.patch("contract_qa.llm.time.sleep")
    @mock.patch("contract_qa.llm._post")
    def test_400_fails_immediately(self, mpost, _sleep):
        mpost.side_effect = ApiError("HTTP 400", status=400, retryable=False)
        with self.assertRaises(ApiError):
            post_json("http://x", {}, "k")
        self.assertEqual(mpost.call_count, 1)

    @mock.patch("contract_qa.llm.time.sleep")
    @mock.patch("contract_qa.llm._post")
    def test_network_error_retried_then_exhausted(self, mpost, _sleep):
        mpost.side_effect = [ApiError("连接失败")] * 3
        with self.assertRaises(ApiError):
            post_json("http://x", {}, "k")
        self.assertEqual(mpost.call_count, 3)


class ChatResponseTest(unittest.TestCase):
    @mock.patch("contract_qa.llm.post_json")
    def test_content_none_becomes_empty(self, mpost):
        mpost.return_value = {"choices": [{"message": {"content": None}}], "usage": {}}
        result = chat([{"role": "user", "content": "x"}], make_settings())
        self.assertEqual(result.content, "")

    @mock.patch("contract_qa.llm.post_json")
    def test_missing_choices_raises(self, mpost):
        mpost.return_value = {"usage": {}}
        with self.assertRaises(ApiError):
            chat([{"role": "user", "content": "x"}], make_settings())

    def test_missing_api_key(self):
        s = make_settings()
        no_key = Settings(
            llm_api_key="",
            llm_base_url=s.llm_base_url,
            llm_model=s.llm_model,
            embedding_api_key=s.embedding_api_key,
            embedding_base_url=s.embedding_base_url,
            embedding_model=s.embedding_model,
            chroma_path=s.chroma_path,
            processed_dir=s.processed_dir,
        )
        with self.assertRaises(ApiError) as ctx:
            chat([{"role": "user", "content": "x"}], no_key)
        self.assertFalse(ctx.exception.retryable)


class EmbedTest(unittest.TestCase):
    @mock.patch("contract_qa.llm.post_json")
    def test_batch_sorted_by_index(self, mpost):
        mpost.return_value = {
            "data": [
                {"index": 1, "embedding": [0.2]},
                {"index": 0, "embedding": [0.1]},
            ]
        }
        self.assertEqual(embed_texts(["a", "b"], make_settings()), [[0.1], [0.2]])

    @mock.patch("contract_qa.llm.post_json")
    def test_missing_index_field_raises(self, mpost):
        mpost.return_value = {"data": [{"embedding": [0.1]}]}
        with self.assertRaises(ApiError):
            embed_texts(["a"], make_settings())

    @mock.patch("contract_qa.llm.post_json")
    def test_count_mismatch_raises(self, mpost):
        mpost.return_value = {"data": [{"index": 0, "embedding": [0.1]}]}
        with self.assertRaises(ApiError):
            embed_texts(["a", "b"], make_settings())

    def test_empty_texts(self):
        self.assertEqual(embed_texts([], make_settings()), [])


if __name__ == "__main__":
    unittest.main()
