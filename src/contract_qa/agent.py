"""工具调用版问答：模型自己决定查什么、读哪几条，然后交卷。

与 graph.py 的固定链路相比，这里没有「先改写、再检索、再生成」的硬顺序：
模型可以反复 search、按 clause_id 读原文、读相邻条款补上下文，
自己判断「够了」再调用 answer 交卷。检索次数与顺序由模型决定，
不再由代码写死 top_k 与单次召回。

边界：最多 MAX_TOOL_ROUNDS 轮工具调用；到上限仍没交卷就强制用已读条款作答。
交卷后仍走引用校验（规则层 + 模型核对），与固定链路同一套标准。
"""

from __future__ import annotations

import json
import time

import structlog

from contract_qa.config import Settings, get_settings
from contract_qa.graph import (
    GENERATE_PROMPT,
    VERIFY_PROMPT,
    GraphResult,
    _parse_json,
    map_citations,
)
from contract_qa.llm import ApiError, chat, chat_with_tools
from contract_qa.retrieval import Hit, Retriever, get_retriever

log = structlog.get_logger("contract_qa.agent")

MAX_TOOL_ROUNDS = 6

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_clauses",
            "description": (
                "按关键词或语义检索合同条款，返回条款 ID、出处与摘要。"
                "可以多次调用，换用不同说法再搜。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索词，保留原文术语效果最好"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 8"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_clause",
            "description": "读取某条条款的完整原文与页码。摘要不够判断时必须调用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_id": {"type": "string", "description": "search_clauses 返回的条款 ID"},
                },
                "required": ["clause_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_neighbors",
            "description": "读取某条条款前后相邻的条款。条款被切分切断、上下文不完整时用它补全。",
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_id": {"type": "string"},
                    "before": {"type": "integer", "description": "向前取几条，默认 1"},
                    "after": {"type": "integer", "description": "向后取几条，默认 1"},
                },
                "required": ["clause_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": (
                "提交最终答案。每个事实都必须能在 citations 指向的条款里逐字找到；"
                "检索不到依据时 answerable 传 false。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "给用户看的回答，不要出现条款 ID"},
                    "answerable": {"type": "boolean"},
                    "citations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "支撑答案的条款 ID 列表",
                    },
                },
                "required": ["answer", "answerable", "citations"],
            },
        },
    },
]

SYSTEM_PROMPT = """你能查阅合同条款，通过调用工具回答问题。

工作方式：
1. 先想清楚问题在问什么，用 search_clauses 检索；一次不准就换说法再搜。
2. 摘要不够判断时，用 read_clause 读原文；条款被切断、上下文不完整时用 read_neighbors。
3. 只依据读到的条款回答，不许用常识、行业惯例或推测补充。
4. 答案里每个事实都要能在 citations 指向的条款里逐字找到。
5. 条款明确说明「不适用 / 不收取 / 无需提供 / 已免除」时，直接说明结论并引用，
   不要因为「问金额而没给金额」就判为不可回答。
6. 多份文件说法不同时，分别列出再说明差异，不要只挑一个。
7. 检索后确认所有相关条款都读不到依据时，才用 answerable=false 交卷；
   此时 citations 必须为空数组——引用只用于支撑答案，不能用来表示「我查过这些」。
8. 答案正文不要出现条款 ID（如 DOC002-C0006），用文件名或项目名指代。
9. 最多 6 轮工具调用，信息够了就尽早交卷。
"""


class ContractAgent:
    """工具调用驱动的最小 Agent：检索、读原文、读邻居、交卷。"""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
        max_rounds: int = MAX_TOOL_ROUNDS,
    ):
        self.settings = settings or get_settings()
        self.retriever = retriever or get_retriever()
        self.max_rounds = max_rounds
        self._order: list[str] = []
        self._pos: dict[str, int] = {}
        self._seen: dict[str, Hit] = {}
        self._tool_log: list[dict] = []
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._timings: dict[str, float] = {}

    # ---------- 工具实现 ----------

    def _tool_search(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return "缺少 query 参数"
        top_k = int(args.get("top_k") or 8)
        hits = self.retriever.search(query, top_k=max(1, min(top_k, 20)))
        for hit in hits:
            self._seen[hit.clause_id] = hit
        rows = [
            {
                "clause_id": hit.clause_id,
                "出处": hit.record.citation_label,
                "页码": f"{hit.record.page_start}-{hit.record.page_end}",
                "摘要": hit.record.text[:140].replace("\n", " "),
            }
            for hit in hits
        ]
        return json.dumps(rows, ensure_ascii=False)

    def _tool_read(self, args: dict) -> str:
        clause_id = str(args.get("clause_id") or "").strip()
        record = self.retriever.records.get(clause_id)
        if record is None:
            return f"条款 {clause_id} 不存在，请用 search_clauses 返回的 ID"
        self._seen[clause_id] = Hit(
            clause_id=clause_id, score=0.0, dense_rank=None, bm25_rank=None, record=record
        )
        return (
            f"条款 ID：{clause_id}\n出处：{record.citation_label}\n"
            f"页码：第 {record.page_start}-{record.page_end} 页\n原文：\n{record.text}"
        )

    def _tool_neighbors(self, args: dict) -> str:
        clause_id = str(args.get("clause_id") or "").strip()
        if clause_id not in self._pos:
            self._prepare_order()
        index = self._pos.get(clause_id)
        if index is None:
            return f"条款 {clause_id} 不存在"
        before = max(0, min(int(args.get("before") or 1), 3))
        after = max(0, min(int(args.get("after") or 1), 3))
        base = self.retriever.records[clause_id]
        chunks: list[str] = []
        for offset in range(-before, after + 1):
            pos = index + offset
            if pos < 0 or pos >= len(self._order):
                continue
            cid = self._order[pos]
            record = self.retriever.records[cid]
            if record.doc_id != base.doc_id:  # 不跨文件读邻居
                continue
            marker = "（本条）" if cid == clause_id else ""
            chunks.append(f"[{cid}]{marker}\n{record.text}")
        return "\n\n".join(chunks) or "没有相邻条款"

    def _prepare_order(self) -> None:
        self._order = list(self.retriever.records.keys())
        self._pos = {cid: i for i, cid in enumerate(self._order)}

    def _dispatch(self, name: str, args: dict) -> str:
        if name == "search_clauses":
            return self._tool_search(args)
        if name == "read_clause":
            return self._tool_read(args)
        if name == "read_neighbors":
            self._prepare_order()
            return self._tool_neighbors(args)
        return f"未知工具：{name}"

    # ---------- 主循环 ----------

    def ask(
        self, question: str, top_k: int = 8, history: list[dict] | None = None
    ) -> GraphResult:
        started = time.perf_counter()
        self._seen = {}
        self._tool_log = []
        self._prompt_tokens = 0
        self._completion_tokens = 0

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # 带上最近两轮：让「那履约保证金呢」这类追问能落到具体对象上
        for turn in (history or [])[-2:]:
            past_question = str(turn.get("question", "")).strip()
            past_answer = str(turn.get("answer", "")).strip().replace("\n", " ")
            if not past_question:
                continue
            messages.append({"role": "user", "content": f"问题：{past_question}"})
            if past_answer:
                messages.append({"role": "assistant", "content": past_answer[:200]})
        messages.append({"role": "user", "content": f"问题：{question}"})
        queries: list[str] = []
        error = ""
        rounds = 0

        for round_index in range(1, self.max_rounds + 1):
            rounds = round_index
            try:
                result = chat_with_tools(messages, TOOLS, self.settings)
            except ApiError as exc:
                error = f"工具对话失败：{exc}"
                log.warning("agent_api_error", round=round_index, error=str(exc))
                break
            self._prompt_tokens += result.prompt_tokens
            self._completion_tokens += result.completion_tokens

            if not result.tool_calls:
                # 模型没调工具，直接回了一段话：当作交卷，但不带引用
                self._timings["agent"] = time.perf_counter() - started
                return self._build_result(
                    question, result.content, False, [], queries, rounds, error
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": result.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in result.tool_calls
                    ],
                }
            )

            for call in result.tool_calls:
                args = _parse_json(call.arguments) or {}
                if call.name == "answer":
                    self._tool_log.append({"tool": "answer", "args": args})
                    cited = len(args.get("citations") or [])
                    log.info("agent_finish", round=round_index, citations=cited)
                    self._timings["agent"] = time.perf_counter() - started
                    return self._build_result(
                        question,
                        str(args.get("answer") or ""),
                        bool(args.get("answerable", False)),
                        [str(c) for c in (args.get("citations") or [])],
                        queries,
                        rounds,
                        error,
                    )
                if call.name == "search_clauses":
                    queries.append(str(args.get("query") or ""))
                output = self._dispatch(call.name, args)
                self._tool_log.append({"tool": call.name, "args": args})
                log.info("agent_tool", round=round_index, tool=call.name)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output[:6000]}
                )

        # 轮次用尽或接口失败：用已读条款做最后一次强制作答
        self._timings["agent"] = time.perf_counter() - started
        return self._force_answer(question, queries, rounds, error)

    def _force_answer(
        self, question: str, queries: list[str], rounds: int, error: str
    ) -> GraphResult:
        if not self._seen:
            reason = error or "检索轮次用尽，未读到可用条款"
            return self._build_result(
                question, "未检索到相关条款。", False, [], queries, rounds, reason
            )

        blocks = []
        for hit in list(self._seen.values())[:10]:
            record = hit.record
            blocks.append(f"[{record.clause_id}] {record.citation_label}\n{record.text}")
        messages = [
            {"role": "system", "content": GENERATE_PROMPT},
            {
                "role": "user",
                "content": f"条款片段：\n\n{chr(10).join(blocks)}\n\n问题：{question}",
            },
        ]
        try:
            result = chat(messages, self.settings, json_mode=True)
            self._prompt_tokens += result.prompt_tokens
            self._completion_tokens += result.completion_tokens
        except ApiError as exc:
            return self._build_result(
                question, "生成失败，已按降级处理。", False, [], queries, rounds, str(exc)
            )
        payload = _parse_json(result.content) or {}
        return self._build_result(
            question,
            str(payload.get("answer") or "生成结果格式异常。"),
            bool(payload.get("answerable", False)),
            [str(c) for c in (payload.get("citations") or [])],
            queries,
            rounds,
            error or "工具轮次用尽",
        )

    def _build_result(
        self,
        question: str,
        answer: str,
        answerable: bool,
        citation_ids: list[str],
        queries: list[str],
        rounds: int,
        error: str,
    ) -> GraphResult:
        hits = [self._seen[cid] for cid in citation_ids if cid in self._seen]
        citations, violations = map_citations(citation_ids, hits)
        result = GraphResult(
            question=question,
            answer=answer.strip(),
            answerable=answerable,
            citations=citations,
            hits=list(self._seen.values()),
            queries=queries,
            route="agent",
            route_reason=f"工具调用 {rounds} 轮",
            citation_violations=violations,
            attempts=rounds,
            error=error,
            verify_rounds=1,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            timings=dict(self._timings),
        )
        problems = self._verify(answer, answerable, citation_ids, hits, violations)
        result.verify_problems = problems
        result.needs_review, result.review_reasons = self._review(result)
        return result

    def _verify(
        self,
        answer: str,
        answerable: bool,
        citation_ids: list[str],
        hits: list[Hit],
        violations: list[str],
    ) -> list[str]:
        """规则层 + 模型核对，与固定链路的 verify 同一套标准（不重试）。"""
        problems: list[str] = []
        if answerable and not citation_ids:
            problems.append("标记为可答，但没有任何引用")
        if not answerable and citation_ids:
            problems.append("标记为不可答，却给出了引用")
        if violations:
            problems.append(f"引用了本次未读过的条款：{violations}")
        if answerable and not answer.strip():
            problems.append("标记为可答，但答案为空")
        if problems or not answerable or not self.settings.verify_llm:
            return problems

        by_id = {hit.clause_id: hit for hit in hits}
        cited = "\n\n".join(
            f"[{cid}] {by_id[cid].record.text}" for cid in citation_ids if cid in by_id
        )
        messages = [
            {"role": "system", "content": VERIFY_PROMPT},
            {"role": "user", "content": f"引用条款：\n\n{cited}\n\n待核对答案：\n{answer}"},
        ]
        try:
            result = chat(messages, self.settings, json_mode=True)
            self._prompt_tokens += result.prompt_tokens
            self._completion_tokens += result.completion_tokens
            payload = _parse_json(result.content)
        except ApiError as exc:
            result_error = f"verify: {exc}"
            self._timings[result_error] = 0.0
            return []
        if payload and not bool(payload.get("supported", True)):
            items = [str(p) for p in payload.get("problems", []) if str(p).strip()]
            problems.extend(items or ["模型判定答案未被引用条款支持"])
        return problems

    @staticmethod
    def _review(result: GraphResult) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if result.citation_violations:
            reasons.append(f"给出了未读条款的引用：{result.citation_violations}")
        if result.verify_problems:
            reasons.extend(f"引用校验未通过：{item}" for item in result.verify_problems)
        return bool(reasons), reasons


_agent: ContractAgent | None = None


def get_agent() -> ContractAgent:
    global _agent
    if _agent is None:
        _agent = ContractAgent()
    return _agent


def reset_agent() -> None:
    global _agent
    _agent = None
