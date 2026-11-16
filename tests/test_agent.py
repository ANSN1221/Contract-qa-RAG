"""工具调用 Agent 的单元测试：不联网，全部 mock 模型返回。"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from contract_qa.agent import ContractAgent
from contract_qa.index import ClauseRecord
from contract_qa.llm import ApiError, ToolCall, ToolChatResult
from contract_qa.retrieval import Hit


def _record(clause_id: str = "DOC001-C0001", text: str = "付款方式：分三次支付") -> ClauseRecord:
    return ClauseRecord(
        clause_id=clause_id,
        doc_id="DOC001",
        doc_title="测试合同",
        clause_no="第一条",
        chapter_path=["第一章 总则"],
        page_start=1,
        page_end=1,
        text=text,
    )


class _FakeRetriever:
    def __init__(self) -> None:
        record = _record()
        self.records = {record.clause_id: record}

    def search(self, query: str, top_k: int = 8, *, use_rerank: bool | None = None):
        record = next(iter(self.records.values()))
        return [Hit(record.clause_id, 0.5, 1, 1, record)]


def _reply(content: str = "", calls: list[ToolCall] | None = None) -> ToolChatResult:
    return ToolChatResult(
        content=content, tool_calls=calls or [], prompt_tokens=10, completion_tokens=5
    )


def _call(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(args, ensure_ascii=False))


def _settings(verify_llm: bool = True) -> mock.Mock:
    settings = mock.Mock()
    settings.has_llm = True
    settings.verify_llm = verify_llm
    return settings


class _ChatStub:
    """替身：verify 里调用的普通 chat，返回固定判定。"""

    def __init__(self, content: str = '{"supported": true, "problems": []}') -> None:
        self.content = content

    def __call__(self, messages, settings=None, **kwargs):
        from contract_qa.llm import ChatResult

        return ChatResult(content=self.content, prompt_tokens=8, completion_tokens=4)


class AgentFlowTest(unittest.TestCase):
    def _run(self, seq, chat_stub=None, max_rounds: int = 6, verify_llm: bool = True):
        with (
            mock.patch("contract_qa.agent.chat_with_tools", side_effect=seq),
            mock.patch("contract_qa.agent.chat", side_effect=chat_stub or _ChatStub()),
        ):
            agent = ContractAgent(
                settings=_settings(verify_llm), retriever=_FakeRetriever(), max_rounds=max_rounds
            )
            return agent.ask("怎么付款？")

    def test_search_read_then_answer(self):
        seq = [
            _reply(calls=[_call("search_clauses", {"query": "付款方式"})]),
            _reply(calls=[_call("read_clause", {"clause_id": "DOC001-C0001"}, "c2")]),
            _reply(
                calls=[
                    _call(
                        "answer",
                        {
                            "answer": "分三次支付",
                            "answerable": True,
                            "citations": ["DOC001-C0001"],
                        },
                        "c3",
                    )
                ]
            ),
        ]
        result = self._run(seq)
        self.assertTrue(result.answerable)
        self.assertEqual([c["clause_id"] for c in result.citations], ["DOC001-C0001"])
        self.assertEqual(result.queries, ["付款方式"])
        self.assertEqual(result.route, "agent")
        self.assertFalse(result.needs_review)

    def test_rounds_exhausted_degrades(self):
        seq = [_reply(calls=[_call("search_clauses", {"query": "付款"})])] * 2
        result = self._run(seq, max_rounds=2)
        self.assertFalse(result.answerable)
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.error)

    def test_citation_never_read_is_flagged(self):
        seq = [
            _reply(
                calls=[
                    _call(
                        "answer",
                        {"answer": "x", "answerable": True, "citations": ["DOC999-C0001"]},
                    )
                ]
            )
        ]
        result = self._run(seq)
        self.assertEqual(result.citation_violations, ["DOC999-C0001"])
        self.assertTrue(result.needs_review)

    def test_malformed_tool_arguments_do_not_crash(self):
        bad = ToolCall(id="c1", name="read_clause", arguments="{not json")
        seq = [
            _reply(calls=[bad]),
            _reply(
                calls=[
                    _call(
                        "answer",
                        {"answer": "未找到", "answerable": False, "citations": []},
                        "c2",
                    )
                ]
            ),
        ]
        result = self._run(seq)
        self.assertFalse(result.answerable)
        self.assertEqual(result.attempts, 2)

    def test_api_error_returns_degraded_result(self):
        with (
            mock.patch("contract_qa.agent.chat_with_tools", side_effect=ApiError("boom")),
            mock.patch("contract_qa.agent.chat", side_effect=_ChatStub()),
        ):
            agent = ContractAgent(settings=_settings(), retriever=_FakeRetriever())
            result = agent.ask("怎么付款？")
        self.assertFalse(result.answerable)
        self.assertIn("工具对话失败", result.error)

    def test_answerable_without_citation_is_flagged(self):
        seq = [
            _reply(
                calls=[_call("answer", {"answer": "分三次", "answerable": True, "citations": []})]
            )
        ]
        result = self._run(seq)
        self.assertTrue(result.needs_review)
        self.assertTrue(any("没有任何引用" in item for item in result.verify_problems))


if __name__ == "__main__":
    unittest.main()
