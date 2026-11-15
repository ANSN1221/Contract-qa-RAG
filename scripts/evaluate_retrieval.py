"""D5 检索评测：A 仅 BM25 / B 仅向量 / C 混合 RRF / D 混合 + rerank。

四组共用同一次召回结果：每题只向量化一次，避免重复消耗接口配额，
也保证对比公平——四组看到的候选集合与向量完全一致。

口径与 data/seeds/README.md 一致：
Recall@5 = 前 5 条至少命中一个 relevant_clause_id 的题数 / 可答题数
不可答题（relevant_clause_ids 为空）不进分母，单独列出。

用法：
    uv run python scripts/evaluate_retrieval.py
    uv run python scripts/evaluate_retrieval.py --no-rerank   # 只跑 A/B/C
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from contract_qa.config import get_settings
from contract_qa.index import load_records
from contract_qa.llm import ApiError
from contract_qa.rerank import rerank
from contract_qa.retrieval import Retriever, fuse_rankings

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "data" / "seeds" / "seeds.jsonl"
REPORT = ROOT / "docs" / "eval-d5.md"
DETAIL = ROOT / "data" / "seeds" / "eval-d5.json"

TOP_K = 5
RECALL_K = 5
GROUPS = ("A", "B", "C", "D")
GROUP_NAMES = {
    "A": "仅 BM25",
    "B": "仅向量",
    "C": "混合 RRF",
    "D": "混合 + rerank",
}
# 权重扫描：复用同一份召回结果，本地重算融合，不额外消耗接口配额
WEIGHT_SWEEPS = ((1.0, 0.5), (0.5, 1.0), (1.0, 1.0), (0.3, 1.0))


@dataclass
class Row:
    qid: str
    question: str
    category: str
    expected: list[str]
    ranks: dict[str, int | None] = field(default_factory=dict)
    tops: dict[str, list[str]] = field(default_factory=dict)
    dense_ids: list[str] = field(default_factory=list)
    sparse_ids: list[str] = field(default_factory=list)
    error: str = ""


def load_seeds() -> list[dict]:
    rows = []
    for line in SEEDS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def best_rank(ids: list[str], expected: list[str]) -> int | None:
    """期望条款在结果中的最靠前名次，1 起；未出现返回 None。"""
    for position, clause_id in enumerate(ids, start=1):
        if clause_id in expected:
            return position
    return None


def evaluate(use_rerank: bool) -> tuple[list[Row], dict]:
    settings = get_settings()
    retriever = Retriever(settings)
    records = {record.clause_id: record for record in load_records(settings)}
    seeds = load_seeds()
    rows: list[Row] = []
    timings = {"bm25": [], "dense": [], "rerank": []}
    rerank_errors: list[str] = []

    for seed in seeds:
        row = Row(
            qid=seed["id"],
            question=seed["question"],
            category=seed["category"],
            expected=seed.get("relevant_clause_ids", []),
        )
        query = seed["question"]
        try:
            started = time.perf_counter()
            sparse = retriever.bm25_search(query, 20)
            timings["bm25"].append(time.perf_counter() - started)

            started = time.perf_counter()
            dense = retriever.dense_search(query, 20)
            timings["dense"].append(time.perf_counter() - started)

            sparse_ids = [cid for cid, _ in sparse]
            dense_ids = [cid for cid, _ in dense]
            row.sparse_ids = sparse_ids
            row.dense_ids = dense_ids
            fused = fuse_rankings(
                dense_ids,
                sparse_ids,
                dense_weight=settings.dense_weight,
                bm25_weight=settings.bm25_weight,
            )
            fused_ids = [cid for cid, _ in fused if cid in records]

            row.tops["A"] = sparse_ids[:TOP_K]
            row.tops["B"] = dense_ids[:TOP_K]
            row.tops["C"] = fused_ids[:TOP_K]

            if use_rerank:
                pool = fused_ids[: settings.rerank_candidates]
                started = time.perf_counter()
                result = rerank(
                    query,
                    [(cid, records[cid].text) for cid in pool],
                    top_n=TOP_K,
                    settings=settings,
                )
                timings["rerank"].append(time.perf_counter() - started)
                row.tops["D"] = [cid for cid, _ in result.ranking][:TOP_K]
        except ApiError as exc:
            row.error = str(exc)
            rerank_errors.append(f"{row.qid}: {exc}")
        except Exception as exc:  # 单题异常不能让整轮评测中断
            row.error = f"{type(exc).__name__}: {exc}"

        for group in GROUPS:
            ids = row.tops.get(group)
            row.ranks[group] = best_rank(ids, row.expected) if ids else None
        rows.append(row)

    stats = {
        "timings": {key: round(sum(v) / len(v), 4) if v else None for key, v in timings.items()},
        "rerank_errors": rerank_errors,
        "use_rerank": use_rerank,
        "settings": settings,
    }
    return rows, stats


def summarize(rows: list[Row]) -> dict:
    answerable = [row for row in rows if row.category != "unanswerable"]
    result = {}
    for group in GROUPS:
        if not any(row.tops.get(group) for row in rows):
            result[group] = None
            continue
        hits = sum(1 for row in answerable if row.ranks.get(group))
        result[group] = {
            "hits": hits,
            "total": len(answerable),
            "recall": hits / len(answerable) if answerable else 0.0,
        }
    return result


def sweep_summary(rows: list[Row]) -> list[dict]:
    """同一份召回结果换不同权重重算融合，看 Recall@5 怎么变。"""
    answerable = [row for row in rows if row.category != "unanswerable"]
    results = []
    for bm25_weight, dense_weight in WEIGHT_SWEEPS:
        hits = 0
        for row in answerable:
            if not row.dense_ids and not row.sparse_ids:
                continue
            fused = fuse_rankings(
                row.dense_ids,
                row.sparse_ids,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
            )
            ids = [cid for cid, _ in fused][:RECALL_K]
            if best_rank(ids, row.expected):
                hits += 1
        results.append(
            {
                "bm25_weight": bm25_weight,
                "dense_weight": dense_weight,
                "hits": hits,
                "total": len(answerable),
                "recall": hits / len(answerable) if answerable else 0.0,
            }
        )
    return results


def render(rows: list[Row], summary: dict, stats: dict, sweeps: list[dict]) -> str:
    settings = stats["settings"]
    answerable_count = sum(1 for row in rows if row.category != "unanswerable")
    rerank_note = (
        f"启用，候选 {settings.rerank_candidates} 条" if stats["use_rerank"] else "未启用"
    )
    lines = [
        "# D5 检索评测",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 语料：{len(rows)} 条种子题（可答 {answerable_count} 条）",
        f"- 检索参数：BM25 权重 {settings.bm25_weight}、向量权重 {settings.dense_weight}、"
        f"每路召回 20 条、Recall@{RECALL_K}",
        f"- 重排：{rerank_note}",
        "",
        "## 一、汇总",
        "",
        "| 组 | 方案 | 命中 | Recall@5 |",
        "|---|---|---|---|",
    ]
    for group in GROUPS:
        item = summary.get(group)
        if item is None:
            lines.append(f"| {group} | {GROUP_NAMES[group]} | 未运行 | — |")
        else:
            lines.append(
                f"| {group} | {GROUP_NAMES[group]} | {item['hits']}/{item['total']} | "
                f"{item['recall']:.0%} |"
            )

    lines += [
        "",
        "## 二、逐题名次",
        "",
        "| 题 | 类别 | 期望条款 | A | B | C | D |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        expected = "、".join(row.expected) if row.expected else "（不可答）"
        cells = []
        for group in GROUPS:
            rank = row.ranks.get(group)
            if not row.tops.get(group):
                cells.append("—")
            elif rank is None:
                cells.append("未命中")
            else:
                cells.append(f"第 {rank} 名")
        lines.append(
            f"| {row.qid} | {row.category} | {expected} | " + " | ".join(cells) + " |"
        )

    lines += ["", "## 三、耗时（每题均值，秒）", "", "| 环节 | 均值 |", "|---|---|"]
    timing_labels = (
        ("bm25", "BM25 检索"),
        ("dense", "向量检索（含接口）"),
        ("rerank", "重排（含接口）"),
    )
    for key, label in timing_labels:
        value = stats["timings"].get(key)
        lines.append(f"| {label} | {'—' if value is None else f'{value:.3f}'} |")

    lines += [
        "",
        "## 四、RRF 权重扫描",
        "",
        "同一份召回结果换权重重算融合，判断权重是不是当前瓶颈。",
        "",
        "| BM25 权重 | 向量权重 | 命中 | Recall@5 |",
        "|---|---|---|---|",
    ]
    for item in sweeps:
        current = (item["bm25_weight"], item["dense_weight"]) == (
            settings.bm25_weight,
            settings.dense_weight,
        )
        note = "（当前默认）" if current else ""
        lines.append(
            f"| {item['bm25_weight']} | {item['dense_weight']} | "
            f"{item['hits']}/{item['total']} | {item['recall']:.0%}{note} |"
        )

    unanswerable = [row for row in rows if row.category == "unanswerable"]
    if unanswerable:
        lines += [
            "",
            "## 五、不可答题（不参与 Recall@5）",
            "",
            "这几题语料里没有答案，测的是生成节点会不会编，不测召回。",
            "",
            "| 题 | 问题 |",
            "|---|---|",
        ]
        for row in unanswerable:
            lines.append(f"| {row.qid} | {row.question} |")

    if stats["rerank_errors"]:
        lines += ["", "## 六、重排失败", ""]
        lines += [f"- {item}" for item in stats["rerank_errors"]]

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="D5 检索四组对照评测")
    parser.add_argument("--no-rerank", action="store_true", help="只跑 A/B/C 三组")
    parser.add_argument("--out", type=Path, default=REPORT, help="报告输出路径")
    args = parser.parse_args()

    settings = get_settings()
    use_rerank = not args.no_rerank
    if use_rerank and not settings.has_rerank:
        print("缺少重排密钥，D 组跳过；补齐 RERANK_API_KEY 后重跑")
        use_rerank = False

    rows, stats = evaluate(use_rerank)
    summary = summarize(rows)
    sweeps = sweep_summary(rows)
    report = render(rows, summary, stats, sweeps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    DETAIL.parent.mkdir(parents=True, exist_ok=True)
    DETAIL.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "summary": summary,
                "weight_sweep": sweeps,
                "timings": stats["timings"],
                "rerank_errors": stats["rerank_errors"],
                "rows": [
                    {
                        "id": row.qid,
                        "question": row.question,
                        "category": row.category,
                        "expected": row.expected,
                        "ranks": row.ranks,
                        "tops": row.tops,
                        # 原始召回也落盘：后续调权重、换融合方式可离线重算，不再调接口
                        "sparse_ids": row.sparse_ids,
                        "dense_ids": row.dense_ids,
                        "error": row.error,
                    }
                    for row in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for group in GROUPS:
        item = summary.get(group)
        if item is None:
            print(f"{group} {GROUP_NAMES[group]}: 未运行")
        else:
            print(
                f"{group} {GROUP_NAMES[group]}: "
                f"{item['hits']}/{item['total']} = {item['recall']:.0%}"
            )
    print(f"报告：{args.out}")
    print(f"明细：{DETAIL}")
    print("权重扫描：")
    for item in sweeps:
        print(
            f"  BM25 {item['bm25_weight']} / 向量 {item['dense_weight']}: "
            f"{item['hits']}/{item['total']} = {item['recall']:.0%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
