"""rerank 的解析、校验与降级测试：不联网。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from contract_qa.llm import ApiError
from contract_qa.rerank import RerankResult, rerank
from contract_qa.retrieval import Retriever


class FakeSettings:
    def __init__(self, api_key="key", base_url="https://example.com/v1", model="m"):
        self.rerank_api_key = api_key
        self.rerank_base_url = base_url
        self.rerank_model = model
        self.rerank_candidates = 16
        self.rerank_enabled = True
        self.bm25_weight = 1.0
        self.dense_weight = 0.5

    @property
    def has_rerank(self) -> bool:
        return bool(self.rerank_api_key and self.rerank_base_url)


def candidates() -> list[tuple[str, str]]:
    return [("c0", "文本零"), ("c1", "文本一"), ("c2", "文本二")]


class RerankParseTest(unittest.TestCase):
    def test_sorted_by_score(self):
        payload = {
            "results": [
                {"index": 0, "relevance_score": 0.1},
                {"index": 2, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.5},
            ]
        }
        with patch("contract_qa.rerank.post_json", return_value=payload):
            result = rerank("问题", candidates(), settings=FakeSettings())
        self.assertEqual([cid for cid, _ in result.ranking], ["c2", "c1", "c0"])
        self.assertEqual(result.ranked_count, 3)
        self.assertEqual(result.missing, [])

    def test_out_of_range_index_dropped(self):
        payload = {"results": [{"index": 5, "relevance_score": 0.9}]}
        with patch("contract_qa.rerank.post_json", return_value=payload):
            with self.assertRaises(ApiError):
                rerank("问题", candidates(), settings=FakeSettings())

    def test_missing_candidates_reported(self):
        payload = {"results": [{"index": 1, "relevance_score": 0.8}]}
        with patch("contract_qa.rerank.post_json", return_value=payload):
            result = rerank("问题", candidates(), settings=FakeSettings())
        self.assertEqual([cid for cid, _ in result.ranking], ["c1"])
        self.assertEqual(result.missing, ["c0", "c2"])

    def test_duplicate_index_kept_once(self):
        payload = {
            "results": [
                {"index": 1, "relevance_score": 0.8},
                {"index": 1, "relevance_score": 0.2},
            ]
        }
        with patch("contract_qa.rerank.post_json", return_value=payload):
            result = rerank("问题", candidates(), settings=FakeSettings())
        self.assertEqual(len(result.ranking), 1)
        self.assertAlmostEqual(result.ranking[0][1], 0.8)

    def test_bool_index_rejected(self):
        # bool 是 int 的子类，True 不能被当成下标 1
        payload = {"results": [{"index": True, "relevance_score": 0.8}]}
        with patch("contract_qa.rerank.post_json", return_value=payload):
            with self.assertRaises(ApiError):
                rerank("问题", candidates(), settings=FakeSettings())

    def test_non_numeric_score_rejected(self):
        payload = {"results": [{"index": 0, "relevance_score": "high"}]}
        with patch("contract_qa.rerank.post_json", return_value=payload):
            with self.assertRaises(ApiError):
                rerank("问题", candidates(), settings=FakeSettings())

    def test_missing_results_field(self):
        with patch("contract_qa.rerank.post_json", return_value={"id": "x"}):
            with self.assertRaises(ApiError):
                rerank("问题", candidates(), settings=FakeSettings())

    def test_empty_candidates_skips_request(self):
        with patch("contract_qa.rerank.post_json") as post:
            result = rerank("问题", [], settings=FakeSettings())
        post.assert_not_called()
        self.assertEqual(result, RerankResult(ranking=[], ranked_count=0))

    def test_missing_key_raises_without_retry(self):
        settings = FakeSettings(api_key="")
        with patch("contract_qa.rerank.post_json") as post:
            with self.assertRaises(ApiError) as ctx:
                rerank("问题", candidates(), settings=settings)
        post.assert_not_called()
        self.assertFalse(ctx.exception.retryable)

    def test_top_n_clamped_to_candidates(self):
        payload = {"results": [{"index": 0, "relevance_score": 0.9}]}
        with patch("contract_qa.rerank.post_json", return_value=payload) as post:
            rerank("问题", candidates(), top_n=99, settings=FakeSettings())
        self.assertEqual(post.call_args[0][1]["top_n"], 3)

    def test_payload_shape(self):
        payload = {"results": [{"index": 0, "relevance_score": 0.9}]}
        with patch("contract_qa.rerank.post_json", return_value=payload) as post:
            rerank("问题", candidates(), top_n=2, settings=FakeSettings())
        url, body, key = post.call_args[0]
        self.assertTrue(url.endswith("/rerank"))
        self.assertEqual(body["query"], "问题")
        self.assertEqual(body["documents"], ["文本零", "文本一", "文本二"])
        self.assertEqual(body["top_n"], 2)
        self.assertEqual(key, "key")


class SearchFallbackTest(unittest.TestCase):
    """重排故障时必须退回融合顺序，且不能吞掉原因。"""

    def _retriever(self) -> Retriever:
        retriever = Retriever.__new__(Retriever)
        retriever.settings = FakeSettings()
        retriever.bm25_weight = 1.0
        retriever.dense_weight = 0.5
        retriever.rerank_error = ""
        retriever.index_warning = ""
        retriever.records = {
            "a": type("R", (), {"text": "甲"})(),
            "b": type("R", (), {"text": "乙"})(),
        }
        retriever.bm25_search = lambda query, top_k: [("a", 1.0), ("b", 0.5)]
        retriever.dense_search = lambda query, top_k: [("b", 0.9)]
        return retriever

    def test_rerank_error_falls_back(self):
        retriever = self._retriever()
        with patch("contract_qa.retrieval.rerank", side_effect=ApiError("401 鉴权失败")):
            hits = retriever.search("问题", top_k=2, use_rerank=True)
        self.assertEqual([hit.clause_id for hit in hits], ["b", "a"])
        self.assertIn("401", retriever.rerank_error)
        self.assertTrue(all(hit.rerank_score is None for hit in hits))

    def test_rerank_reorders(self):
        retriever = self._retriever()
        result = RerankResult(ranking=[("a", 0.95), ("b", 0.2)], ranked_count=2)
        with patch("contract_qa.retrieval.rerank", return_value=result):
            hits = retriever.search("问题", top_k=2, use_rerank=True)
        self.assertEqual([hit.clause_id for hit in hits], ["a", "b"])
        self.assertAlmostEqual(hits[0].rerank_score, 0.95)
        self.assertEqual(retriever.rerank_error, "")

    def test_rerank_disabled_keeps_fused_order(self):
        retriever = self._retriever()
        with patch("contract_qa.retrieval.rerank") as fake:
            hits = retriever.search("问题", top_k=2, use_rerank=False)
        fake.assert_not_called()
        self.assertEqual([hit.clause_id for hit in hits], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
