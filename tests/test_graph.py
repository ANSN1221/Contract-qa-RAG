"""图链路与 JSON 容错的单元测试：不联网。"""

from __future__ import annotations

import unittest
from unittest import mock

from contract_qa.graph import GraphResult, QAGraph, _parse_json, map_citations
from contract_qa.index import ClauseRecord
from contract_qa.llm import ApiError
from contract_qa.retrieval import Hit


def fake_hit(clause_id: str) -> Hit:
    record = ClauseRecord(
        clause_id=clause_id,
        doc_id="DOC001",
        doc_title="测试合同",
        clause_no="第一条",
        chapter_path=["第一章 总则"],
        page_start=3,
        page_end=4,
        text="测试条款正文",
    )
    return Hit(clause_id=clause_id, score=0.5, dense_rank=1, bm25_rank=2, record=record)


class ParseJsonTest(unittest.TestCase):
    def test_clean_json(self):
        self.assertEqual(_parse_json('{"a": 1}'), {"a": 1})

    def test_json_wrapped_in_prose(self):
        content = '好的，结果如下：\n{"answer": "是", "citations": []}\n希望有帮助。'
        self.assertEqual(_parse_json(content)["answer"], "是")

    def test_broken_json_returns_none(self):
        self.assertIsNone(_parse_json('{"answer": "是", "citations": [}'))

    def test_empty_returns_none(self):
        self.assertIsNone(_parse_json(""))

    def test_non_object_json_returns_none(self):
        # 数组不是我们要的结构，取不到 answer 字段，按失败处理
        self.assertIsNone(_parse_json("[1, 2, 3]"))

    def test_none_and_blank_return_none(self):
        # 模型偶尔返回空 content，不能因为 TypeError 崩溃
        self.assertIsNone(_parse_json(None))
        self.assertIsNone(_parse_json(""))
        self.assertIsNone(_parse_json("   "))


class MapCitationsTest(unittest.TestCase):
    def test_known_citation_mapped(self):
        citations, violations = map_citations(["DOC001-C0001"], [fake_hit("DOC001-C0001")])
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["label"], "DOC001 · 第一章 总则 · 第一条")
        self.assertEqual(citations[0]["page_start"], 3)
        self.assertEqual(violations, [])

    def test_unknown_citation_is_a_violation_not_dropped(self):
        citations, violations = map_citations(
            ["DOC001-C0001", "DOC999-C9999"], [fake_hit("DOC001-C0001")]
        )
        self.assertEqual(len(citations), 1)
        self.assertEqual(violations, ["DOC999-C9999"])

    def test_empty_citations(self):
        citations, violations = map_citations([], [fake_hit("DOC001-C0001")])
        self.assertEqual((citations, violations), ([], []))

    def test_duplicate_citation_is_deduplicated(self):
        citations, violations = map_citations(
            ["DOC001-C0001", "DOC001-C0001"], [fake_hit("DOC001-C0001")]
        )
        self.assertEqual(len(citations), 1)
        self.assertEqual(violations, [])


class NodeFallbackTest(unittest.TestCase):
    """接口调用失败时节点必须降级继续，并记录 node_errors。"""

    class _FakeRetriever:
        records: dict = {}

        def search(self, query: str, top_k: int = 8, *, use_rerank: bool | None = None):
            return []

    def test_route_fails_open_and_marks_error(self):
        with mock.patch("contract_qa.graph.chat", side_effect=ApiError("timeout")):
            graph = QAGraph(retriever=self._FakeRetriever())
            result = graph.ask("合同里付款方式是什么？")
        # 放行按合同域处理，链路继续走完，错误被记录而不是抛栈
        self.assertEqual(result.route, "contract_qa")
        self.assertTrue(result.node_errors)
        self.assertFalse(result.answerable)

    def test_rewrite_falls_back_to_original_question(self):
        calls = []

        def fake_chat(messages, settings=None, **kwargs):
            calls.append(messages)
            if len(calls) == 1:  # 第一次（route）成功
                class R:
                    content = '{"route": "contract_qa", "reason": "t"}'
                    prompt_tokens = 1
                    completion_tokens = 1
                return R()
            raise ApiError("timeout")  # rewrite 失败

        with mock.patch("contract_qa.graph.chat", side_effect=fake_chat):
            graph = QAGraph(retriever=self._FakeRetriever())
            result = graph.ask("合同里付款方式是什么？")
        self.assertEqual(result.queries, ["合同里付款方式是什么？"])
        self.assertTrue(result.node_errors)

    def test_generate_degrades_without_hits(self):
        def fake_chat(messages, settings=None, **kwargs):
            class R:
                content = '{"route": "contract_qa", "reason": "t"}'
                prompt_tokens = 1
                completion_tokens = 1
            return R()

        with mock.patch("contract_qa.graph.chat", side_effect=fake_chat):
            graph = QAGraph(retriever=self._FakeRetriever())
            result = graph.ask("合同里付款方式是什么？")
        self.assertIn("未检索到相关条款", result.answer)
        self.assertFalse(result.answerable)


class ReviewTest(unittest.TestCase):
    def test_answerable_without_citations_needs_review(self):
        result = GraphResult(question="q", answer="a", answerable=True, citations=[])
        needs, reasons = QAGraph._review(result)
        self.assertTrue(needs)
        self.assertTrue(any("没有任何可核对" in r for r in reasons))

    def test_violation_needs_review(self):
        result = GraphResult(
            question="q", answer="a", answerable=True,
            citations=[], citation_violations=["DOC999-C1"],
        )
        needs, _ = QAGraph._review(result)
        self.assertTrue(needs)

    def test_consistent_answer_passes(self):
        result = GraphResult(
            question="q", answer="a", answerable=True, citations=[{"clause_id": "C1"}]
        )
        needs, reasons = QAGraph._review(result)
        self.assertFalse(needs)
        self.assertEqual(reasons, [])

    def test_unanswerable_with_citations_needs_review(self):
        result = GraphResult(
            question="q", answer="a", answerable=False, citations=[{"clause_id": "C1"}]
        )
        needs, _ = QAGraph._review(result)
        self.assertTrue(needs)


if __name__ == "__main__":
    unittest.main()
