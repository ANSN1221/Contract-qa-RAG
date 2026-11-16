"""Streamlit 界面：提问 → 看答案 → 逐条核对引用。

支持多轮追问：页面保留本次会话的历史，追问时把最近两轮带给改写节点，
让「那履约保证金呢」这类指代能落到具体对象上。

定位是「演示与核对工具」，不是产品前端：一屏之内能看到答案、页码、
原文摘录和这次调用的开销，方便判断这条回答可不可信。
"""

from __future__ import annotations

import time

import streamlit as st

from contract_qa.config import get_settings
from contract_qa.graph import GraphResult

st.set_page_config(page_title="合同条款问答", layout="wide")

MODE_LABELS = {
    "固定链路（快）": "graph",
    "工具调用 Agent（准，智能体）": "agent",
}

EXAMPLES = (
    "试试这些：\n"
    "- 食品安全抽检的样品要在多少个工作日内检完？\n"
    "- 疫苗项目投标要交保证金吗？\n"
    "- 这个项目的环境影响评价批复文号是多少？（应该拒答）\n"
    "多轮可以接着问：「那履约保证金呢？」"
)


@st.cache_resource(show_spinner=False)
def load_runner(mode: str):
    """按模式加载一次运行器；索引与向量库在首次调用时加载，之后复用。"""
    if mode == "agent":
        from contract_qa.agent import ContractAgent

        return ContractAgent()
    from contract_qa.graph import QAGraph

    return QAGraph()


def render_metrics(result: GraphResult, elapsed: float) -> None:
    cols = st.columns(4)
    cols[0].metric("耗时", f"{elapsed:.1f}s")
    cols[1].metric("tokens（计费字数）", f"{result.tokens:,}")
    cols[2].metric("工具轮次" if result.route == "agent" else "生成次数", f"{result.attempts}")
    cols[3].metric("引用条款", f"{len(result.citations)}")


def render_answer(result: GraphResult, elapsed: float) -> None:
    if result.answerable:
        st.success("可回答 · 已给出引用")
    elif result.route == "out_of_scope":
        st.info("超出范围：这个问题不需要查合同条款")
    else:
        st.warning("未在所给条款中找到依据（正确拒答）")

    st.markdown(result.answer)
    render_metrics(result, elapsed)

    if result.citations:
        st.markdown("**引用出处**")
        for index, cite in enumerate(result.citations, start=1):
            label = f"[{index}] {cite['label']} · 第 {cite['page_start']}-{cite['page_end']} 页"
            with st.expander(label, expanded=index == 1):
                st.text(str(cite["text"]))

    if result.queries:
        st.caption("检索用词：" + " / ".join(result.queries))

    if result.needs_review:
        st.error("需要人工复核：\n\n" + "\n".join(f"- {item}" for item in result.review_reasons))
    if result.node_errors:
        st.warning("节点降级：\n\n" + "\n".join(f"- {item}" for item in result.node_errors))


def render_hits(result: GraphResult) -> None:
    if not result.hits:
        return
    st.markdown("**召回明细**")
    rows = [
        {
            "排名": index,
            "条款": hit.record.citation_label,
            "页码": f"{hit.record.page_start}-{hit.record.page_end}",
            "融合分": round(hit.score, 4),
            "向量名次": hit.dense_rank or "—",
            "关键词名次": hit.bm25_rank or "—",
            "摘要": hit.record.text[:60].replace("\n", " "),
        }
        for index, hit in enumerate(result.hits[:10], start=1)
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def run_ask(runner, question: str, top_k: int, history: list[dict]) -> tuple[GraphResult, float]:
    started = time.perf_counter()
    result = runner.ask(question, top_k=top_k, history=history)
    return result, time.perf_counter() - started


def main() -> None:
    st.title("合同条款问答")
    st.caption(
        "每条回答都给出文件名 · 章节 · 条款 · 页码，点开引用可直接核对原文。"
        "问答基于 10 份公开招标与合同文件，共 766 条条款。"
    )

    st.session_state.setdefault("turns", [])
    turns: list[dict] = st.session_state["turns"]

    with st.sidebar:
        st.subheader("设置")
        mode_label = st.radio("问答模式", list(MODE_LABELS), index=0)
        mode = MODE_LABELS[mode_label]
        top_k = st.slider("召回条数", min_value=4, max_value=16, value=8, step=1)
        show_hits = st.checkbox("显示召回明细", value=False)
        if st.button("清空对话", width="stretch"):
            st.session_state["turns"] = []
            st.rerun()
        st.divider()
        st.caption("固定链路：改写→检索→生成→校验，约 5 秒。")
        st.caption("Agent（智能体）：模型自己决定检索几次、读哪条原文，更准，约 8 秒。")
        if mode == "agent":
            st.caption("提示：Agent 每题成本约为固定链路的 4 倍。")

        settings = get_settings()
        st.divider()
        ready = settings.has_llm and settings.has_embedding
        st.caption("接口状态：" + ("已配置" if ready else "缺少密钥，请检查 .env"))
        for index, provider in enumerate(getattr(settings, "llm_providers", ()) or ()):
            role = "主" if index == 0 else f"备{index}"
            st.caption(f"{role}：{provider.name}（{provider.model}）")

    for turn in turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            render_answer(turn["result"], turn["elapsed"])

    question = st.chat_input("问点什么，比如「履约保证金要在几天内交？」")
    if not question:
        if not turns:
            st.info(EXAMPLES)
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            label = "检索与生成…" if st.session_state.get("warmed") else "加载索引并检索…"
            with st.spinner(label):
                runner = load_runner(mode)
                history = [
                    {"question": item["question"], "answer": item["result"].answer}
                    for item in turns
                ]
                result, elapsed = run_ask(runner, question, top_k, history)
            st.session_state["warmed"] = True
        except FileNotFoundError as exc:
            st.error(
                f"索引或条款缺失：{exc}\n\n"
                "先运行 `uv run contract-qa parse` 与 `uv run contract-qa index`。"
            )
            return
        except Exception as exc:  # 网络、配额、密钥问题统一提示，不吞掉原因
            detail = str(exc)
            if "402" in detail or "Insufficient Balance" in detail:
                st.error(
                    "所有对话供应商都返回余额不足（HTTP 402）。"
                    "充值或补充备用密钥后重试；无需重启界面。"
                )
            elif "401" in detail or "Unauthorized" in detail:
                st.error("接口鉴权失败（HTTP 401）：请检查 .env 里的密钥是否有效。")
            else:
                st.error(f"调用失败：{type(exc).__name__}: {detail}")
            return

        render_answer(result, elapsed)
        if show_hits:
            render_hits(result)

    turns.append({"question": question, "result": result, "elapsed": elapsed})
    st.session_state["turns"] = turns


main()
