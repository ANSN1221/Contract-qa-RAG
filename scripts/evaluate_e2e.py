"""E 阶段端到端评测：50 题全跑一遍主链路。

和 D5 的区别：D5 只测召回（检索到没到），这里测整条链路（答对没答对、引用对不对、
该拒答的有没有硬答）。判定规则全部基于种子题标注，不靠人工逐条看。

三个口径：
- 答案正确：答案里出现该题的关键事实（去空白后做包含匹配）
- 引用命中：citations 里出现标注的任一 answer clause
- 双正确：上面两条同时成立（严格口径，用于对外报数）

不可答题只看一件事：有没有拒答（answerable=false）。

用法：
    uv run python scripts/evaluate_e2e.py --limit 3    # 先小样本校验判定
    uv run python scripts/evaluate_e2e.py              # 全量 50 题
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from contract_qa.graph import QAGraph

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "data" / "seeds" / "seeds.jsonl"
REPORT = ROOT / "docs" / "eval-e2e.md"
DETAIL = ROOT / "data" / "seeds" / "eval-e2e.json"

# 每道可答题的关键事实：答案里得出现其中之一才算答对
KEYS: dict[str, list[str]] = {
    "Q01": ["安装调试费", "售后服务费", "包装费"],
    "Q02": ["24小时", "24 小时"],
    "Q03": ["20个工作日", "20 个工作日"],
    "Q04": ["60个工作日", "60 个工作日"],
    "Q05": ["2025年11月30", "2025 年11 月30"],
    "Q06": ["70%"],
    "Q07": ["5%", "1%", "千分之六"],
    "Q08": ["488", "425", "457"],
    "Q09": ["8510000", "851万", "851 万"],
    "Q10": ["六份"],
    "Q14": ["037165962573", "6596"],
    "Q15": [],
    "Q16": ["3天", "3 天"],
    "Q17": ["10日", "10 日", "十个"],
    "Q18": ["62处", "62 处"],
    "Q19": ["9月30", "9 月30"],
    "Q20": ["5人", "5 人"],
    "Q21": ["486500", "肆拾捌万陆仟伍佰"],
    "Q22": ["1年", "1 年", "一年"],
    "Q23": ["9个月", "9 个月"],
    "Q24": ["不收取", "不要求缴纳", "无需缴纳", "免收"],
    "Q25": ["2026年12月31", "2026 年12 月31"],
    "Q31": ["7日", "7 日", "七天"],
    "Q32": ["24小时", "24 小时"],
    "Q33": ["10%"],
    "Q34": ["30克", "30 克"],
    "Q35": ["1小时", "1 小时"],
    "Q36": ["20分钟", "20 分钟"],
    "Q37": ["2026年1月", "2026 年1 月"],
    "Q38": ["9条", "36处", "36 处"],
    "Q39": ["365"],
    "Q40": ["1%"],
    "Q41": ["9个", "6个", "9 个", "6 个"],
    "Q42": ["刘俊学", "景国宁", "王金生"],
    "Q43": ["签署", "立即生效"],
    "Q44": ["不再接受", "不接收", "无需纸质"],
}


def norm(text: str) -> str:
    """去空白与标点后再比对，避免「【10】日内」这类排版干扰判定。"""
    return re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9%‰]", "", text)


def load_seeds() -> list[dict]:
    rows = []
    for line in SEEDS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


REFUSAL_MARKERS = ("未记载", "未包含", "未在所给条款", "无法确定", "没有具体", "未找到", "未提及")


def judge(seed: dict, answer: str, answerable: bool, citations: list[dict]) -> dict:
    """按标准答案判定三类结果。

    不可答题的拒答口径放宽一档：模型明确说「未记载 / 无法确定」时，
    即使 answerable 标了 true（附带描述了相关程序），也算拒答——
    用户问的是具体事实，回答「文件里没写」是正确行为。
    """
    if seed["category"] == "unanswerable":
        informative = any(marker in answer for marker in REFUSAL_MARKERS)
        return {
            "answer_hit": False,
            "cite_hit": False,
            "both": False,
            "refused": (not answerable) or informative,
        }
    keys = KEYS.get(seed["id"], [])
    text = norm(answer)
    answer_hit = any(norm(k) in text for k in keys) if keys else False
    cited = {c["clause_id"] for c in citations}
    cite_hit = bool(cited & set(seed["relevant_clause_ids"]))
    return {
        "answer_hit": answer_hit,
        "cite_hit": cite_hit,
        "both": answer_hit and cite_hit,
        "refused": not answerable,
    }


def pct(hit: int, total: int) -> str:
    return f"{hit}/{total} = {hit / total:.0%}" if total else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端评测")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题，用于校验判定")
    parser.add_argument("--out", type=Path, default=REPORT)
    parser.add_argument(
        "--reuse",
        type=Path,
        default=None,
        help="从已有明细重判（改了判定规则后不必重新调用模型）",
    )
    args = parser.parse_args()

    seeds = load_seeds()
    if args.limit:
        seeds = seeds[: args.limit]

    if args.reuse:
        rows = json.loads(args.reuse.read_text(encoding="utf-8"))["rows"]
        by_id = {seed["id"]: seed for seed in load_seeds()}
        for row in rows:
            seed = by_id[row["id"]]
            row.update(
                judge(
                    seed,
                    row["answer"],
                    row["answerable"],
                    [{"clause_id": cid} for cid in row["citations"]],
                )
            )
            row["expected"] = seed["relevant_clause_ids"]
            row["category"] = seed["category"]
            row["question"] = seed["question"]
        print(f"复用 {len(rows)} 条旧结果重判，未调用模型")
    else:
        rows = []
        graph = QAGraph()
        rows = _run_cases(graph, seeds)

    _write_report(rows, args.out)
    return 0


def _run_cases(graph: QAGraph, seeds: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for index, seed in enumerate(seeds, start=1):
        started = time.perf_counter()
        try:
            result = graph.ask(seed["question"])
            elapsed = time.perf_counter() - started
            verdict = judge(seed, result.answer, result.answerable, result.citations)
            row = {
                "id": seed["id"],
                "question": seed["question"],
                "category": seed["category"],
                "answer": result.answer,
                "answerable": result.answerable,
                "citations": [c["clause_id"] for c in result.citations],
                "expected": seed["relevant_clause_ids"],
                "verify_rounds": result.verify_rounds,
                "verify_problems": result.verify_problems,
                "needs_review": result.needs_review,
                "review_reasons": result.review_reasons,
                "elapsed": round(elapsed, 2),
                "tokens": result.tokens,
                "error": "",
                **verdict,
            }
        except Exception as exc:  # 单题异常不能让整轮评测中断
            row = {
                "id": seed["id"],
                "question": seed["question"],
                "category": seed["category"],
                "answer": "",
                "answerable": False,
                "citations": [],
                "expected": seed["relevant_clause_ids"],
                "verify_rounds": 0,
                "verify_problems": [],
                "needs_review": False,
                "review_reasons": [],
                "elapsed": round(time.perf_counter() - started, 2),
                "tokens": 0,
                "error": f"{type(exc).__name__}: {exc}",
                "answer_hit": False,
                "cite_hit": False,
                "both": False,
                "refused": False,
            }
        rows.append(row)
        flag = "✓" if row["both"] or (row["category"] == "unanswerable" and row["refused"]) else "✗"
        print(
            f"[{index:2d}/{len(seeds)}] {seed['id']} {flag} "
            f"可答={row['answerable']} 引用={len(row['citations'])} "
            f"校验轮次={row['verify_rounds']} {row['elapsed']}s"
        )
    return rows


def _write_report(rows: list[dict], out: Path) -> None:
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    answer_hits = sum(1 for r in answerable if r["answer_hit"])
    cite_hits = sum(1 for r in answerable if r["cite_hit"])
    both_hits = sum(1 for r in answerable if r["both"])
    refused = sum(1 for r in unanswerable if r["refused"])
    hard_answer = len(unanswerable) - refused
    verify_triggered = sum(1 for r in rows if r["verify_rounds"] > 1 or r["verify_problems"])
    review = sum(1 for r in rows if r["needs_review"])
    elapsed_all = [r["elapsed"] for r in rows]
    tokens_all = [r["tokens"] for r in rows]
    p95 = sorted(elapsed_all)[max(0, int(len(elapsed_all) * 0.95) - 1)]

    lines = [
        "# 端到端评测",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 样本：{len(rows)} 题（可答 {len(answerable)}、不可答 {len(unanswerable)}）",
        "- 判定：答案正确＝答案含标注的关键事实；引用命中＝引用含标注条款；双正确＝两者同时成立",
        "- 不可答题：拒答＝answerable=false，或明确说明「未记载 / 无法确定」"
        "（带解释的拒答同样算对）",
        "",
        "## 一、汇总",
        "",
        "| 指标 | 结果 |",
        "|---|---|",
        f"| 答案正确（可答题） | {pct(answer_hits, len(answerable))} |",
        f"| 引用命中（可答题） | {pct(cite_hits, len(answerable))} |",
        f"| 双正确（严格口径） | {pct(both_hits, len(answerable))} |",
        f"| 拒答准确（不可答） | {pct(refused, len(unanswerable))} |",
        f"| 硬答（不可答却给答案） | {hard_answer} 题 |",
        f"| verify 触发（回退或判不通过） | {verify_triggered} 题 |",
        f"| 标记需人工复核 | {review} 题 |",
        f"| 平均耗时 | {statistics.mean(elapsed_all):.2f}s（P95 {p95:.2f}s） |",
        f"| 平均 tokens | {statistics.mean(tokens_all):.0f} |",
        "",
        "## 二、可答题逐题",
        "",
        "| 题 | 答案对 | 引用对 | 双正确 | 校验轮次 | 耗时 | 引用条款 | 期望条款 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in answerable:
        lines.append(
            f"| {r['id']} | {'✓' if r['answer_hit'] else '✗'} | {'✓' if r['cite_hit'] else '✗'} "
            f"| {'✓' if r['both'] else '✗'} | {r['verify_rounds']} | {r['elapsed']}s "
            f"| {'、'.join(r['citations']) or '—'} | {'、'.join(r['expected'])} |"
        )

    lines += [
        "",
        "## 三、不可答题逐题",
        "",
        "| 题 | 拒答 | 校验轮次 | 耗时 | 答案摘录 |",
        "|---|---|---|---|---|",
    ]
    for r in unanswerable:
        lines.append(
            f"| {r['id']} | {'✓' if r['refused'] else '✗ 硬答'} | {r['verify_rounds']} "
            f"| {r['elapsed']}s | {r['answer'][:40]} |"
        )

    failures = [
        r
        for r in rows
        if not (r["both"] or (r["category"] == "unanswerable" and r["refused"]))
    ]
    if failures:
        lines += ["", "## 四、失败明细", ""]
        for r in failures:
            cited = "、".join(r["citations"]) or "无"
            expected = "、".join(r["expected"]) or "（应拒答）"
            lines.append(f"### {r['id']} {r['question']}")
            lines.append(f"- 答案：{r['answer'][:160]}")
            lines.append(f"- 引用：{cited}｜期望：{expected}")
            if r["verify_problems"]:
                lines.append(f"- 校验问题：{r['verify_problems']}")
            lines.append("")

    if any(r["error"] for r in rows):
        lines += ["", "## 五、异常", ""]
        lines += [f"- {r['id']}: {r['error']}" for r in rows if r["error"]]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    DETAIL.write_text(
        json.dumps(
            {"generated_at": datetime.now().isoformat(timespec="seconds"), "rows": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"答案正确 {pct(answer_hits, len(answerable))}")
    print(f"引用命中 {pct(cite_hits, len(answerable))}")
    print(f"双正确   {pct(both_hits, len(answerable))}")
    print(f"拒答准确 {pct(refused, len(unanswerable))}（硬答 {hard_answer} 题）")
    print(f"verify 触发 {verify_triggered} 题｜需复核 {review} 题")
    print(
        f"平均 {statistics.mean(elapsed_all):.2f}s｜P95 {p95:.2f}s｜"
        f"平均 {statistics.mean(tokens_all):.0f} tokens"
    )
    print(f"报告：{out}")


if __name__ == "__main__":
    raise SystemExit(main())
