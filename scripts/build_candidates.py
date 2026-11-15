"""导出候选条款清单，供人工标注种子题时浏览。"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from contract_qa.index import read_clause_rows

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SEEDS = ROOT / "data" / "seeds"


def main() -> int:
    rows = []
    for path in sorted(PROCESSED.glob("DOC*.jsonl")):
        for r in read_clause_rows(path):
            rows.append({
                "clause_id": r["clause_id"],
                "doc_id": r["doc_id"],
                "clause_no": r["clause_no"],
                "chapter": " > ".join(r["chapter_path"]),
                "page": r["page_start"],
                "chars": r["char_len"],
                "preview": r["text"][:80].replace("\n", " "),
            })
    if not rows:
        print("没有找到条款文件，请先运行 contract-qa parse 与 contract-qa index")
        return 1

    SEEDS.mkdir(parents=True, exist_ok=True)
    out = SEEDS / "candidates.csv"
    # 用无 BOM 的 UTF-8：带 BOM 的首列名会读成带不可见前缀的字符串
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"导出 {len(rows)} 条 -> {out}")
    for doc, n in sorted(Counter(r["doc_id"] for r in rows).items()):
        print(f"  {doc}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
