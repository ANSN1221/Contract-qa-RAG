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


class _FixedRetriever:
    """固定返回一条候选，让 verify 能按 id 找到条款。"""

    def __init__(self) -> None:
        hit = fake_hit("DOC001-C0001")
        self.records = {"DOC001-C0001": hit.record}

    def search(self, query: str, top_k: int = 8, *, use_rerank: bool | None = None):
        return [fake_hit("DOC001-C0001")]


class _Reply:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompt_tokens = 10
        self.completion_tokens = 5


class VerifyNodeTest(unittest.TestCase):
    """verify 节点：规则校验、模型校验、最多 2 轮回退、接口失败降级。"""

    def _ask(self, generate_seq, verify_seq=None, verify_llm=True):
        generate_seq = list(generate_seq)
        verify_seq = list(verify_seq or [])

        def fake_chat(messages, settings=None, **kwargs):
            system = messages[0]["content"]
            if system.startswith("判断用户问题"):
                return _Reply('{"route": "contract_qa", "reason": "t"}')
            if system.startswith("把用户问题改写"):
                return _Reply('{"queries": ["付款方式"]}')
            if system.startswith("你是合同审查助手"):
                return _Reply(generate_seq.pop(0))
            if system.startswith("核对答案"):
                return _Reply(verify_seq.pop(0))
            raise AssertionError(f"未预期的 system prompt：{system[:30]}")

        settings = mock.Mock()
        settings.has_llm = True
        settings.verify_llm = verify_llm
        with mock.patch("contract_qa.graph.chat", side_effect=fake_chat):
            graph = QAGraph(settings=settings, retriever=_FixedRetriever())
            return graph.ask("付款方式是什么？")

    def test_answer_without_citation_retries_once(self):
        result = self._ask(
            [
                '{"answer": "分三次付款", "citations": [], "answerable": true}',
                '{"answer": "分三次付款", "citations": ["DOC001-C0001"], "answerable": true}',
            ],
            ['{"supported": true, "problems": [], "fix_hint": ""}'],
        )
        # 第一轮被规则层拦下（没有引用），第二轮通过
        self.assertEqual(result.verify_rounds, 2)
        self.assertEqual([c["clause_id"] for c in result.citations], ["DOC001-C0001"])
        self.assertFalse(result.needs_review)

    def test_citation_outside_candidates_retries(self):
        result = self._ask(
            [
                '{"answer": "分三次", "citations": ["DOC999-C0001"], "answerable": true}',
                '{"answer": "分三次", "citations": ["DOC001-C0001"], "answerable": true}',
            ],
            ['{"supported": true, "problems": [], "fix_hint": ""}'],
        )
        self.assertEqual(result.verify_rounds, 2)
        self.assertEqual(result.citation_violations, [])
        self.assertFalse(result.needs_review)

    def test_refusal_needs_no_verify_call(self):
        # 拒答不调模型校验：verify_seq 为空，一旦被调用就会 IndexError
        result = self._ask(
            ['{"answer": "未在所给条款中找到依据", "citations": [], "answerable": false}']
        )
        self.assertEqual(result.verify_rounds, 1)
        self.assertFalse(result.answerable)
        self.assertFalse(result.needs_review)

    def test_llm_rejects_then_passes(self):
        result = self._ask(
            [
                '{"answer": "分三次付款", "citations": ["DOC001-C0001"], "answerable": true}',
                '{"answer": "分三次付款，比例 30/50/20", '
                '"citations": ["DOC001-C0001"], "answerable": true}',
            ],
            [
                '{"supported": false, "problems": ["比例在条款里找不到"], "fix_hint": "删掉比例"}',
                '{"supported": true, "problems": [], "fix_hint": ""}',
            ],
        )
        self.assertEqual(result.verify_rounds, 2)
        self.assertEqual(result.verify_problems, [])
        self.assertFalse(result.needs_review)

    def test_gives_up_after_two_rounds(self):
        bad = '{"answer": "分三次付清", "citations": ["DOC001-C0001"], "answerable": true}'
        reject = '{"supported": false, "problems": ["条款未约定次数"], "fix_hint": "改为不可答"}'
        result = self._ask([bad, bad, bad], [reject, reject, reject])
        # 第 3 次校验后达到上限，结束并标记人工复核
        self.assertEqual(result.verify_rounds, 3)
        self.assertTrue(result.needs_review)
        self.assertTrue(any("引用校验未通过" in reason for reason in result.review_reasons))

    def test_verify_api_error_degrades_without_blocking(self):
        def fake_chat(messages, settings=None, **kwargs):
            system = messages[0]["content"]
            if system.startswith("判断用户问题"):
                return _Reply('{"route": "contract_qa", "reason": "t"}')
            if system.startswith("把用户问题改写"):
                return _Reply('{"queries": ["付款方式"]}')
            if system.startswith("你是合同审查助手"):
                return _Reply(
                    '{"answer": "分三次付款", "citations": ["DOC001-C0001"], "answerable": true}'
                )
            raise ApiError("timeout")

        settings = mock.Mock()
        settings.has_llm = True
        settings.verify_llm = True
        with mock.patch("contract_qa.graph.chat", side_effect=fake_chat):
            graph = QAGraph(settings=settings, retriever=_FixedRetriever())
            result = graph.ask("付款方式是什么？")
        # 校验挂了要放行，但必须留下痕迹
        self.assertTrue(result.answerable)
        self.assertEqual(result.verify_rounds, 1)
        self.assertTrue(any("verify" in error for error in result.node_errors))


if __name__ == "__main__":
    unittest.main()
