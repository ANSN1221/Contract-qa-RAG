"""命令行入口：check 检查环境，parse 解析切分，index 建索引，ask 提问。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from contract_qa.config import PROJECT_ROOT, get_settings

REQUIRED_DIRECTORIES = (
    "data/raw",
    "data/processed",
    "docs",
    "src/contract_qa",
    "tests",
)
REGISTER_PATH = PROJECT_ROOT / "data" / "document-register.csv"


def discover_documents(raw_dir: Path) -> list[Path]:
    """Return local PDF documents in stable order."""
    if not raw_dir.exists():
        return []
    return sorted(
        path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )


def read_register() -> list[dict[str, str]]:
    if not REGISTER_PATH.exists():
        return []
    with REGISTER_PATH.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def run_check() -> int:
    """Check the scaffold; missing source documents are reported, not treated as an error."""
    print("合同条款智能问答 · 环境检查")
    print(f"Python: {sys.version.split()[0]}")

    missing = [name for name in REQUIRED_DIRECTORIES if not (PROJECT_ROOT / name).is_dir()]
    if missing:
        print("失败：缺少目录 " + ", ".join(missing))
        return 1

    documents = discover_documents(PROJECT_ROOT / "data/raw")
    rows = read_register()
    ready = sum(1 for row in rows if row.get("status") == "ready")
    scanned = sum(1 for row in rows if row.get("status") == "scan_only")

    print("目录结构：通过")
    print(f"data/raw PDF：{len(documents)} 份")
    print(f"登记表：{len(rows)} 行（可解析 {ready} / 纯扫描 {scanned}）")

    # 只报告「是否配置」，绝不打印密钥内容
    from contract_qa.index import BM25_FILE

    settings = get_settings()
    env_exists = (PROJECT_ROOT / ".env").exists()
    print(f".env：{'已存在' if env_exists else '缺失，请复制 .env.example 后填入密钥'}")
    print(f"对话密钥：{'已配置' if settings.has_llm else '未配置（ask 会失败）'}")
    print(f"向量密钥：{'已配置' if settings.has_embedding else '未配置（向量索引会跳过）'}")
    bm25_path = settings.processed_dir / BM25_FILE
    legacy_path = settings.processed_dir / "bm25.pkl"
    if bm25_path.exists():
        print(f"BM25 索引：就绪（{bm25_path.name}）")
    elif legacy_path.exists():
        print("BM25 索引：仍是旧格式 bm25.pkl，请运行 contract-qa index 重建")
    else:
        print("BM25 索引：未建，请运行 contract-qa index")
    print(f"检索权重：BM25 {settings.bm25_weight} / 向量 {settings.dense_weight}")

    if not documents:
        print("待办：将脱敏 PDF 放入 data/raw，并更新 data/document-register.csv")
    return 0


def run_parse(out_dir: Path | None) -> int:
    try:
        from contract_qa.parse import format_report
        from contract_qa.parse import run_parse as do_parse
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的兜底提示
        print(f"失败：缺少依赖（{exc}）。请先安装 pymupdf：pip install pymupdf")
        return 1

    try:
        payload = do_parse(PROJECT_ROOT, out_dir)
    except Exception as exc:  # 与 run_ask 一致：统一提示，不吞掉原因
        print(f"失败：{type(exc).__name__}: {exc}")
        return 1

    print(format_report(payload))
    print(f"manifest: {payload.get('manifest_path', '')}")
    skipped = payload.get("skipped") or []
    if skipped:
        print(f"跳过 {len(skipped)} 份：" + ", ".join(
            f"{item['doc_id']}({item['reason']})" for item in skipped
        ))
    stale = payload.get("stale_files") or []
    if stale:
        print("警告：以下过期条款文件不在登记表内，未自动删除：" + ", ".join(stale))
    return 0


def run_index(rebuild: bool) -> int:
    try:
        from contract_qa.index import BM25_FILE, build_bm25, build_chroma, load_records
    except ImportError as exc:  # pragma: no cover
        print(f"失败：缺少依赖（{exc}）")
        return 1

    if rebuild:
        print("提示：索引始终整体重建，--rebuild 已废弃，可省略")

    try:
        records = load_records()
        if not records:
            print("失败：data/processed 下没有条款，请先运行 contract-qa parse")
            return 1
        print(f"条款 {len(records)} 条")

        bm25_path = build_bm25(records)
        print(f"BM25 索引：{bm25_path}")

        settings = get_settings()
        if not settings.has_embedding:
            print("跳过向量索引：.env 缺少 EMBEDDING_API_KEY（BM25 索引已可用）")
            return 0

        count = build_chroma(records, settings)
        print(f"向量索引：{count} 条 -> {settings.chroma_path}")
    except Exception as exc:
        print(f"失败：{type(exc).__name__}: {exc}")
        return 1
    print("提示：长驻进程请调用 reset_graph() 后再提问，避免使用旧索引缓存")
    legacy = get_settings().processed_dir / "bm25.pkl"
    if legacy.exists():
        print(
            f"提示：旧格式索引 {legacy.name} 已不再读取，"
            f"新格式 {BM25_FILE} 就绪后可手动删除旧文件"
        )
    return 0


def run_ask(question: str, top_k: int, show_hits: bool, use_agent: bool = False) -> int:
    try:
        if use_agent:
            from contract_qa.agent import get_agent

            runner = get_agent()
        else:
            from contract_qa.graph import get_graph

            runner = get_graph()
    except ImportError as exc:  # pragma: no cover
        print(f"失败：缺少依赖（{exc}）")
        return 1

    try:
        result = runner.ask(question, top_k=top_k)
    except FileNotFoundError as exc:
        print(f"失败：{exc}")
        return 1
    except Exception as exc:  # 网络、配额、密钥问题统一提示，不吞掉原因
        print(f"失败：{type(exc).__name__}: {exc}")
        return 1

    print(f"\n问题：{result.question}")
    print(f"路由：{result.route}" + (f"（{result.route_reason}）" if result.route_reason else ""))
    if result.queries:
        print(f"改写：{' / '.join(result.queries)}")
    print(f"回答：{result.answer}")
    print(f"可答：{'是' if result.answerable else '否'}")

    if result.citations:
        print("\n引用：")
        for index, cite in enumerate(result.citations, start=1):
            print(f"  [{index}] {cite['label']} · 第{cite['page_start']}-{cite['page_end']}页")
            snippet = str(cite["text"])[:100].replace("\n", " ")
            print(f"      {snippet}……")
    if result.citation_violations:
        print(f"\n警告：模型给出了候选之外的引用 {result.citation_violations}")
    if result.needs_review:
        print("\n需要人工复核：")
        for reason in result.review_reasons:
            print(f"  - {reason}")
    if result.node_errors:
        print("\n节点降级：")
        for item in result.node_errors:
            print(f"  - {item}")
    if result.error:
        print(f"\n降级：{result.error}（尝试 {result.attempts} 次）")

    if show_hits:
        print("\n召回明细：")
        for rank, hit in enumerate(result.hits, start=1):
            r = hit.record
            print(
                f"  {rank}. {r.clause_id} rrf={hit.score:.4f} "
                f"dense={hit.dense_rank} bm25={hit.bm25_rank} — {r.clause_no}"
            )
    timing = " ".join(f"{k}={v:.2f}s" for k, v in result.timings.items())
    print(f"\ntokens：prompt {result.prompt_tokens} / completion {result.completion_tokens}")
    if timing:
        print(f"节点耗时：{timing}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="合同条款智能问答：引用可核对的合同条款 RAG")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check", help="检查项目结构、密钥与索引状态")
    parse_cmd = subparsers.add_parser("parse", help="解析 PDF 并切分条款")
    parse_cmd.add_argument("--out", type=Path, default=None, help="输出目录，默认 data/processed")
    index_cmd = subparsers.add_parser("index", help="建 BM25 与向量索引")
    index_cmd.add_argument("--rebuild", action="store_true", help="已废弃：索引始终整体重建")
    ask_cmd = subparsers.add_parser("ask", help="提问并返回答案与引用")
    ask_cmd.add_argument("question", help="问题文本")
    ask_cmd.add_argument("--top-k", type=int, default=8, help="送入模型的条款数，默认 8")
    ask_cmd.add_argument(
        "--agent",
        action="store_true",
        help="改用工具调用 Agent（模型自己决定检索几次、读哪几条）",
    )
    ask_cmd.add_argument("--show-hits", action="store_true", help="打印召回明细")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in (None, "check"):
        raise SystemExit(run_check())
    if args.command == "parse":
        raise SystemExit(run_parse(args.out))
    if args.command == "index":
        raise SystemExit(run_index(args.rebuild))
    if args.command == "ask":
        raise SystemExit(run_ask(args.question, args.top_k, args.show_hits, args.agent))
    parser.error(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
