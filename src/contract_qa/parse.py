"""解析编排：PDF -> 行 -> 条款 -> JSONL + manifest。

单份文件失败只记入 skipped，不中断整批；条款文件先写临时文件再替换，
避免中途失败留下半截产物。过期文件只报告、不自动删除。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

from contract_qa.models import Clause, ParseResult
from contract_qa.splitter import choose_plan, split_lines
from contract_qa.text import build_lines, extract_pages, page_stats

REGISTER_NAME = "document-register.csv"
REQUIRED_REGISTER_FIELDS = ("doc_id", "file_name")


def parse_document(pdf_path: Path, doc_id: str) -> ParseResult:
    pages, page_count = extract_pages(pdf_path)
    stats = page_stats(pages)
    lines, dropped = build_lines(pages)
    plan = choose_plan(lines)

    result = ParseResult(
        doc_id=doc_id,
        source_file=pdf_path.name,
        page_count=page_count,
        page_stats=stats,
        split_mode=plan.mode,
        dropped_lines=dropped,
    )
    result.clauses, result.merged_fragments = split_lines(doc_id, lines, plan)
    return result


def load_register(register_path: Path) -> list[dict[str, str]]:
    if not register_path.exists():
        raise FileNotFoundError(f"缺少登记表：{register_path}")
    with register_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for lineno, row in enumerate(rows, start=2):  # 第 1 行是表头
        missing = [field for field in REQUIRED_REGISTER_FIELDS if not row.get(field)]
        if missing:
            raise ValueError(f"{register_path.name} 第 {lineno} 行缺少字段：{missing}")
    return rows


def _write_clauses(path: Path, clauses: list[Clause]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        for clause in clauses:
            fh.write(json.dumps(clause.to_dict(), ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_parse(project_root: Path, out_dir: Path | None = None) -> dict[str, object]:
    raw_dir = project_root / "data" / "raw"
    register_path = project_root / "data" / REGISTER_NAME
    out_dir = out_dir or project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_register(register_path)
    manifest: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    # 只报告、不删除：这些文件可能是人工保留的样本，删数据必须有人确认
    expected = {f"{row['doc_id']}.jsonl" for row in rows}
    stale_files = sorted(
        path.name for path in out_dir.glob("DOC*.jsonl") if path.name not in expected
    )

    for row in rows:
        pdf_path = raw_dir / row["file_name"]
        if not pdf_path.exists():
            skipped.append({"doc_id": row["doc_id"], "reason": "文件缺失"})
            continue

        try:
            result = parse_document(pdf_path, row["doc_id"])
        except Exception as exc:  # 单份坏文件不应拖垮整批
            skipped.append(
                {"doc_id": row["doc_id"], "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue

        summary = result.summary()
        summary["document_type"] = row.get("document_type", "")
        summary["title"] = row.get("title", "")
        summary["short_name"] = row.get("short_name", "")
        summary["status"] = row.get("status", "")

        if result.is_scanned:
            summary["clauses_written"] = 0
            summary["note"] = "纯扫描件，无可抽取文本，跳过切分"
            manifest.append(summary)
            continue

        _write_clauses(out_dir / f"{row['doc_id']}.jsonl", result.clauses)
        summary["clauses_written"] = len(result.clauses)
        manifest.append(summary)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"documents": manifest, "skipped": skipped, "stale_files": stale_files},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "skipped": skipped,
        "stale_files": stale_files,
        "manifest_path": str(manifest_path),
    }


def format_report(payload: dict[str, object]) -> str:
    docs = cast(list[dict], payload.get("manifest") or [])
    lines = [
        "doc_id  类型     页数  字符     条款  模式     页均条款  中位长度",
        "-" * 78,
    ]
    total_clauses = 0
    for doc in docs:
        if doc["scanned"]:
            lines.append(
                f"{doc['doc_id']}  {doc['document_type']:5s} {doc['page_count']:>4}  "
                f"{doc['total_chars']:>6}  {'扫描':>6}  {'-':6s}  {'-':>7}  {'-':>7}"
            )
            continue
        total_clauses += int(doc["clause_count"])
        per_page = doc["clause_count"] / doc["page_count"] if doc["page_count"] else 0
        lines.append(
            f"{doc['doc_id']}  {doc['document_type']:5s} {doc['page_count']:>4}  "
            f"{doc['total_chars']:>6}  {doc['clause_count']:>6}  {doc['split_mode']:6s}  "
            f"{per_page:>7.2f}  {doc['clause_len_median']:>7}"
        )
    lines.append("-" * 78)
    lines.append(f"条款总数 {total_clauses}")
    return "\n".join(lines)
