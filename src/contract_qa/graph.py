"""主链路：route → rewrite → retrieve → generate。

为什么用图而不是函数串联：这条链路有分支（无关问题直接短路）、有条件重试
（JSON 解析失败要重来一次），用 if/else 写下去会变成嵌套金字塔，而且日志打点
散在各处。换成显式状态机之后，每个节点的输入输出都是同一个 state，
加 D6 的 verify 节点只需要多插一条边。

本文件只负责编排，具体检索与调用都在 retrieval / llm 里。

降级约定：「模型判不出来」与「接口调用失败」是两件事。
前者走内容级兜底（放行、退回原句），后者走节点级降级并记入 node_errors，
任何一步失败都不应该让整条链路抛栈。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TypedDict

import structlog
from langgraph.graph import END, StateGraph

from contract_qa.config import Settings, get_settings
from contract_qa.llm import ApiError, chat
from contract_qa.retrieval import Hit, Retriever, get_retriever, reset_retriever

log = structlog.get_logger("contract_qa.graph")

MAX_QUERIES = 3          # 复合问题最多拆 3 个子查询，再多就超出上下文预算
GENERATE_RETRIES = 1     # malformed 只重试一次，附上错误信息
MAX_VERIFY_ROUNDS = 2    # 引用校验不通过时最多回退 2 轮，避免无限重生成


class QAState(TypedDict, total=False):
    question: str
    top_k: int
    use_rerank: bool
    route: str
    route_reason: str
    queries: list[str]
    # hits 存 dict 而不是 Hit 对象：状态需要可序列化，跨检查点不会丢字段
    hits: list[dict]
    answer: str
    answerable: bool
    citation_ids: list[str]
    attempts: int
    error: str
    node_errors: list[str]
    prompt_tokens: int
    completion_tokens: int
    timings: dict[str, float]
    # verify 节点：轮次、未通过原因、回退给 generate 的修改建议
    verify_rounds: int
    verify_problems: list[str]
    verify_feedback: str
    verify_error: str


ROUTE_PROMPT = """判断用户问题是否需要查阅合同条款才能回答。

只输出 JSON：
{"route": "contract_qa" 或 "out_of_scope", "reason": "一句话理由"}

route 取值：
- contract_qa：问的是合同/招标文件里的内容（金额、期限、责任、程序、标准等），需要查条款
- out_of_scope：闲聊、与合同无关的通用问题、要求你做条款之外的判断（如"这份合同合不合法"）

注意：只要问题在问某份文件里的具体事实（金额、周期、比例、人数、地点、编号、名称），
即使听起来像通用行业常识，也走 contract_qa——本工具手上有真实文件，能查到就不能推给"常识"。
"""

REWRITE_PROMPT = """把用户问题改写成适合条款检索的查询。

规则：
1. 去掉口语和疑问词，保留实体与术语。例：「保证金多少钱？」→「投标保证金 金额」
2. 如果问题包含多个独立诉求，拆成多条查询，最多 3 条。
3. 不要臆造原文里没有的术语，也不要回答问题。
4. 只输出 JSON：{"queries": ["查询1", "查询2"]}
"""

GENERATE_PROMPT = """你是合同审查助手。只能依据给定的条款片段回答问题。

规则：
1. 只有当所有片段都与问题无关时，answer 才填「未在所给条款中找到依据」，answerable 为 false，
   citations 为空数组。
2. 不许补充片段之外的法律常识、行业惯例或推测。
3. 回答里每个事实都必须能在 citations 指向的片段中找到原文依据。
4. 若条款明确说明某事项不适用、不收取、无需提供、已免除，直接说明结论并引用，
   不要因为「问题问金额而条款没给金额」就判为不可回答。
5. 用词不同不等于没有依据：问题里的「设备总价款」「货款」与条款里的「合同总价」
   「价款」是同义表述，先对齐术语再判断片段是否够用；答案里可以注明条款的原始说法。
6. 答案分散在多条片段里时，合并成一个完整回答，不要因为单条片段不完整就判为不可回答。
   跨多份文件的问题，按文件分别列出各自的数字再汇总。
7. 问题用「这个项目」「本平台」「这份合同」这类指代时，按所给片段实际涉及的文件来理解，
   不要因为指代不明确而拒答。
8. 答案正文里不要出现条款 ID（如 DOC002-C0006），用文件名或项目名指代；条款 ID 只放在 citations 里。
9. 只输出 JSON：{"answer": "……", "citations": ["条款ID"], "answerable": true}
"""

VERIFY_PROMPT = """核对答案是否被引用条款支持。

规则：
1. 逐条检查答案里的每个事实，看能否在引用条款原文中找到依据。
2. 条款里没写的，就是没有依据，不要用常识或行业惯例替它补。
3. 答案与引用条款矛盾时判为不支持。
4. 只输出 JSON：
{"supported": true, "problems": [], "fix_hint": ""}
不支持时 problems 每条写一个具体问题，fix_hint 写一句怎么改（如「删除第 2 句」「改引用到 XX」）。"""


def map_citations(citation_ids: list[str], hits: list[Hit]) -> tuple[list[dict], list[str]]:
    """把模型给出的引用 id 映射成可展示的引用；候选之外的 id 记为违例，不静默丢弃。"""
    by_id = {hit.clause_id: hit for hit in hits}
    citations: list[dict] = []
    violations: list[str] = []
    seen: set[str] = set()
    for cid in citation_ids:
        if cid in seen:
            continue  # 模型重复给同一个 id 时只展示一次
        seen.add(cid)
        hit = by_id.get(cid)
        if hit is None:
            violations.append(cid)
            continue
        record = hit.record
        citations.append(
            {
                "clause_id": record.clause_id,
                "label": record.citation_label,
                "page_start": record.page_start,
                "page_end": record.page_end,
                "text": record.text,
            }
        )
    return citations, violations


@dataclass
class GraphResult:
    question: str
    answer: str
    answerable: bool
    citations: list[dict] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    route: str = ""
    route_reason: str = ""
    citation_violations: list[str] = field(default_factory=list)
    attempts: int = 0
    error: str = ""
    node_errors: list[str] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    verify_rounds: int = 0
    verify_problems: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _parse_json(content: str) -> dict | None:
    """容错解析：先直解，失败则截取第一个 { 到最后一个 } 再试一次。

    模型偶尔会在 JSON 前后加一句解释；严格 json.loads 会直接判失败，
    但这属于可救的格式问题，不是内容问题。
    只接受对象：下游要按字段取值，数组或标量即使合法也用不了。
    """
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(content[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None


def _timed(state: QAState, key: str, started: float) -> dict[str, float]:
    timings = dict(state.get("timings", {}))
    timings[key] = time.perf_counter() - started
    return timings


def _with_error(state: QAState, message: str) -> list[str]:
    return [*state.get("node_errors", []), message]


class QAGraph:
    def __init__(self, settings: Settings | None = None, retriever: Retriever | None = None):
        self.settings = settings or get_settings()
        self.retriever = retriever or get_retriever()
        self.app = self._build()

    # ---------- 节点 ----------

    def _route(self, state: QAState) -> QAState:
        started = time.perf_counter()
        try:
            result = chat(
                [
                    {"role": "system", "content": ROUTE_PROMPT},
                    {"role": "user", "content": state["question"]},
                ],
                self.settings,
                json_mode=True,
            )
        except ApiError as exc:
            # 调用失败也放行：宁可多查一次，不能让一次抖动让整条链路不可用
            log.warning("route_api_error", error=str(exc))
            return {
                "route": "contract_qa",
                "route_reason": f"路由调用失败，按合同域处理：{exc}",
                "node_errors": _with_error(state, f"route: {exc}"),
                "timings": _timed(state, "route", started),
            }
        payload = _parse_json(result.content) or {}
        route = payload.get("route", "contract_qa")
        if route not in {"contract_qa", "out_of_scope"}:
            route = "contract_qa"  # 判不出来就放行，宁可多查一次
        log.info("route", route=route, reason=payload.get("reason", ""))
        return {
            "route": route,
            "route_reason": str(payload.get("reason", "")),
            "prompt_tokens": state.get("prompt_tokens", 0) + result.prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + result.completion_tokens,
            "timings": _timed(state, "route", started),
        }

    def _rewrite(self, state: QAState) -> QAState:
        started = time.perf_counter()
        try:
            result = chat(
                [
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": state["question"]},
                ],
                self.settings,
                json_mode=True,
            )
        except ApiError as exc:
            log.warning("rewrite_api_error", error=str(exc))
            return {
                "queries": [state["question"]],  # 调用失败同样退回原句，不阻断链路
                "node_errors": _with_error(state, f"rewrite: {exc}"),
                "timings": _timed(state, "rewrite", started),
            }
        payload = _parse_json(result.content) or {}
        queries = [q.strip() for q in payload.get("queries", []) if str(q).strip()]
        if not queries:
            queries = [state["question"]]  # 改写失败就退回原句，不阻断链路
        log.info("rewrite", queries=queries)
        return {
            "queries": queries[:MAX_QUERIES],
            "prompt_tokens": state.get("prompt_tokens", 0) + result.prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + result.completion_tokens,
            "timings": _timed(state, "rewrite", started),
        }

    def _retrieve(self, state: QAState) -> QAState:
        started = time.perf_counter()
        top_k = int(state.get("top_k") or 8)
        merged: dict[str, Hit] = {}
        use_rerank = state.get("use_rerank")
        for query in state.get("queries", [state["question"]]):
            for hit in self.retriever.search(query, top_k=top_k, use_rerank=use_rerank):
                # 同一条款被多个子查询命中时保留排序更靠前的那次
                if (
                    hit.clause_id not in merged
                    or hit.sort_key > merged[hit.clause_id].sort_key
                ):
                    merged[hit.clause_id] = hit
        hits = sorted(merged.values(), key=lambda h: h.sort_key, reverse=True)[:top_k]
        log.info("retrieve", merged=len(merged), kept=len(hits), top_k=top_k)
        return {
            "hits": [
                {
                    "clause_id": h.clause_id,
                    "score": h.score,
                    "dense_rank": h.dense_rank,
                    "bm25_rank": h.bm25_rank,
                    "rerank_score": h.rerank_score,
                }
                for h in hits
            ],
            "timings": _timed(state, "retrieve", started),
        }

    def _format_candidates(self, hits: list[Hit]) -> str:
        blocks = []
        for index, hit in enumerate(hits, start=1):
            record = hit.record
            where = record.doc_id
            if record.chapter_path:
                where += f" / {record.chapter_path[-1]}"
            if record.clause_no:
                where += f" / {record.clause_no}"
            where += f" / 第{record.page_start}-{record.page_end}页"
            blocks.append(f"[{index}] 条款ID={record.clause_id}（{where}）\n{record.text}")
        return "\n\n".join(blocks)

    def _generate(self, state: QAState) -> QAState:
        started = time.perf_counter()
        hits = self._hits_from_state(state)
        if not hits:
            return {"answer": "未检索到相关条款。", "answerable": False, "citation_ids": []}

        candidates = self._format_candidates(hits)
        user_content = f"条款片段：\n\n{candidates}\n\n问题：{state['question']}"
        feedback = state.get("verify_feedback", "")
        if feedback:
            # 回退重生成时把上一轮的不通过原因带上，否则模型只会重复同样的答案
            user_content += f"\n\n上一轮输出未通过引用校验，请逐条修正：\n{feedback}"
        messages = [
            {"role": "system", "content": GENERATE_PROMPT},
            {"role": "user", "content": user_content},
        ]

        attempts = 0
        prompt_tokens = state.get("prompt_tokens", 0)
        completion_tokens = state.get("completion_tokens", 0)
        error = ""
        failure = "parse"
        payload: dict | None = None

        while attempts <= GENERATE_RETRIES:
            attempts += 1
            try:
                result = chat(messages, self.settings, json_mode=True)
            except ApiError as exc:
                # llm 内部已按可重试性退避过，这里直接降级，不再叠加重试
                failure = "api"
                error = f"生成调用失败：{exc}"
                log.warning("generate_api_error", attempt=attempts, error=str(exc))
                break
            prompt_tokens += result.prompt_tokens
            completion_tokens += result.completion_tokens
            payload = _parse_json(result.content)
            if payload is not None:
                break
            error = f"输出不是合法 JSON：{result.content[:160]}"
            log.warning("generate_malformed", attempt=attempts, preview=result.content[:120])
            if attempts <= GENERATE_RETRIES:
                # 把上一轮的 malformed 输出也放回对话：角色交替，模型才知道自己错在哪
                messages = [
                    *messages,
                    {"role": "assistant", "content": result.content},
                    {"role": "user",
                     "content": "上一条输出不是合法 JSON。"
                                "只输出 JSON 对象，不要任何解释文字。"},
                ]

        timings = _timed(state, "generate", started)

        if payload is None:
            answer = (
                "生成调用失败，已按降级处理。"
                if failure == "api"
                else "模型输出解析失败，已重试一次。"
            )
            return {
                "answer": answer,
                "answerable": False,
                "citation_ids": [],
                "attempts": attempts,
                "error": error,
                "node_errors": _with_error(state, f"generate: {error}"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "timings": timings,
            }

        return {
            "answer": str(payload.get("answer", "")).strip(),
            "answerable": bool(payload.get("answerable", False)),
            "citation_ids": [str(c) for c in payload.get("citations", [])],
            "attempts": attempts,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "timings": timings,
        }

    def _verify(self, state: QAState) -> QAState:
        """引用校验：先做零成本规则检查，再让模型核对答案是否被条款支持。

        规则层抓结构性错误（可答却没引用、引用不在候选里），
        模型层抓内容性错误（答案里有条款没写的推断）。
        任一不通过就回退重生成；校验接口本身失败时放行并记 node_errors，
        不能因为「校验不了」就把正确答案一起废掉。
        """
        started = time.perf_counter()
        hits = self._hits_from_state(state)
        by_id = {hit.clause_id: hit for hit in hits}
        answer = state.get("answer", "").strip()
        answerable = bool(state.get("answerable", False))
        citation_ids = state.get("citation_ids", [])
        rounds = int(state.get("verify_rounds", 0))

        problems: list[str] = []
        if answerable and not citation_ids:
            problems.append("标记为可答，但没有任何引用")
        if not answerable and citation_ids:
            problems.append("标记为不可答，却给出了引用")
        unknown = [cid for cid in citation_ids if cid not in by_id]
        if unknown:
            problems.append(f"引用了候选之外的条款：{unknown}")
        if answerable and not answer:
            problems.append("标记为可答，但答案为空")

        prompt_tokens = state.get("prompt_tokens", 0)
        completion_tokens = state.get("completion_tokens", 0)
        verify_error = ""
        feedback = ""

        # 规则层已不通过时不必再花一次模型调用
        if not problems and answerable and self.settings.verify_llm:
            cited = "\n\n".join(
                f"[{cid}] {by_id[cid].record.text}" for cid in citation_ids if cid in by_id
            )
            messages = [
                {"role": "system", "content": VERIFY_PROMPT},
                {"role": "user", "content": f"引用条款：\n\n{cited}\n\n待核对答案：\n{answer}"},
            ]
            try:
                result = chat(messages, self.settings, json_mode=True)
                prompt_tokens += result.prompt_tokens
                completion_tokens += result.completion_tokens
                payload = _parse_json(result.content)
                if payload is None:
                    verify_error = f"校验输出不是合法 JSON：{result.content[:120]}"
                elif not bool(payload.get("supported", True)):
                    problems.extend(
                        str(item) for item in payload.get("problems", []) if str(item).strip()
                    )
                    feedback = str(payload.get("fix_hint", "")).strip()
                    if not problems:
                        problems.append("模型判定答案未被引用条款支持")
            except ApiError as exc:
                verify_error = f"校验调用失败：{exc}"
                log.warning("verify_api_error", error=str(exc))

        node_errors = state.get("node_errors", [])
        if verify_error:
            node_errors = _with_error(state, f"verify: {verify_error}")
        timings = _timed(state, "verify", started)
        passed = not problems
        log.info("verify", passed=passed, round=rounds + 1, problems=len(problems))
        return {
            "verify_rounds": rounds + 1,
            "verify_problems": problems,
            "verify_feedback": feedback or "；".join(problems),
            "verify_error": verify_error,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "node_errors": node_errors,
            "timings": timings,
        }

    def _hits_from_state(self, state: QAState) -> list[Hit]:
        out = []
        for item in state.get("hits", []):
            record = self.retriever.records.get(item["clause_id"])
            if record is None:
                continue
            out.append(
                Hit(
                    clause_id=item["clause_id"],
                    score=item["score"],
                    dense_rank=item.get("dense_rank"),
                    bm25_rank=item.get("bm25_rank"),
                    record=record,
                    rerank_score=item.get("rerank_score"),
                )
            )
        return out

    # ---------- 图 ----------

    def _build(self):
        graph = StateGraph(QAState)
        graph.add_node("route", self._route)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("verify", self._verify)

        graph.set_entry_point("route")
        graph.add_conditional_edges(
            "route",
            lambda state: "rewrite" if state.get("route") == "contract_qa" else END,
            {"rewrite": "rewrite", END: END},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "verify")
        graph.add_conditional_edges(
            "verify",
            lambda state: (
                "generate"
                if state.get("verify_problems")
                and state.get("verify_rounds", 0) <= MAX_VERIFY_ROUNDS
                else END
            ),
            {"generate": "generate", END: END},
        )
        return graph.compile()

    # ---------- 对外 ----------

    def ask(self, question: str, top_k: int = 8) -> GraphResult:
        state = self.app.invoke({"question": question, "top_k": top_k})
        hits = self._hits_from_state(state)
        citations, violations = map_citations(state.get("citation_ids", []), hits)

        result = GraphResult(
            question=question,
            answer=state.get("answer", ""),
            answerable=bool(state.get("answerable", False)),
            citations=citations,
            hits=hits,
            queries=state.get("queries", []),
            route=state.get("route", ""),
            route_reason=state.get("route_reason", ""),
            citation_violations=violations,
            attempts=state.get("attempts", 0),
            error=state.get("error", ""),
            node_errors=state.get("node_errors", []),
            verify_rounds=state.get("verify_rounds", 0),
            verify_problems=state.get("verify_problems", []),
            prompt_tokens=state.get("prompt_tokens", 0),
            completion_tokens=state.get("completion_tokens", 0),
            timings=state.get("timings", {}),
        )
        if result.route == "out_of_scope":
            result.answer = "这个问题不需要查阅合同条款，超出本工具范围。"
        result.needs_review, result.review_reasons = self._review(result)
        return result

    @staticmethod
    def _review(result: GraphResult) -> tuple[bool, list[str]]:
        """引用一致性后置校验：不篡改模型答案，但绝不静默放过。"""
        reasons: list[str] = []
        if result.citation_violations:
            reasons.append(f"模型给出了候选之外的引用：{result.citation_violations}")
        if result.answerable and not result.citations:
            reasons.append("标记为可答，但没有任何可核对的引用")
        if not result.answerable and result.citations:
            reasons.append("标记为不可答，却仍给出了引用")
        if result.verify_problems:
            # 重试到上限仍未通过，答案照给但必须标出来
            reasons.extend(f"引用校验未通过：{item}" for item in result.verify_problems)
        return bool(reasons), reasons


_graph: QAGraph | None = None


def get_graph() -> QAGraph:
    global _graph
    if _graph is None:
        _graph = QAGraph()
    return _graph


def reset_graph() -> None:
    """重建索引或改配置后调用，同时清掉检索器缓存。"""
    global _graph
    _graph = None
    reset_retriever()
