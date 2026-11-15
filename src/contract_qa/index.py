"""建索引：Chroma 向量索引 + BM25 关键词索引。

两份索引共享同一套 clause_id，检索阶段才能做融合。
BM25 用中文二元切分，不引入分词依赖——对条款这种短文本，二元词足够，
也不会因为自定义词典缺失把「合同价款」这类词切碎。

索引一律存纯数据（JSON），不反序列化 Python 对象：
pickle 文件一旦被替换，加载时可能执行任意代码。
"""

from __future__ import annotations

import gzip
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from contract_qa.config import Settings, get_settings
from contract_qa.llm import embed_texts

COLLECTION_NAME = "clauses"
BM25_FILE = "bm25.json.gz"
LEGACY_BM25_FILE = "bm25.pkl"
BM25_FORMAT_VERSION = 1
_ID_PATTERN = re.compile(r"[0-9a-zA-Z_.]+")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")

REQUIRED_ROW_FIELDS = (
    "clause_id",
    "doc_id",
    "clause_no",
    "chapter_path",
    "page_start",
    "page_end",
    "text",
)


def tokenize(text: str) -> list[str]:
    """中文取二元词，英文数字保留原词。"""
    lowered = text.lower()
    tokens: list[str] = []
    for chunk in _CJK_PATTERN.findall(lowered):
        if len(chunk) == 1:
            tokens.append(chunk)
        else:
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    tokens.extend(_ID_PATTERN.findall(lowered))
    return tokens


@dataclass
class ClauseRecord:
    clause_id: str
    doc_id: str
    doc_title: str
    clause_no: str
    chapter_path: list[str]
    page_start: int
    page_end: int
    text: str

    def metadata(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "clause_no": self.clause_no,
            "chapter_path": " > ".join(self.chapter_path),
            "page_start": self.page_start,
            "page_end": self.page_end,
        }

    @property
    def citation_label(self) -> str:
        parts = [self.doc_id]
        if self.chapter_path:
            parts.append(self.chapter_path[-1])
        if self.clause_no:
            parts.append(self.clause_no)
        return " · ".join(parts)


def read_clause_rows(path: Path) -> list[dict]:
    """读一份条款 JSONL，带行号报错与结构校验。脚本与建索引用同一套，避免各写一份。"""
    if not path.exists():
        raise FileNotFoundError(f"条款文件不存在：{path}")
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} 第 {lineno} 行不是合法 JSON：{exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} 第 {lineno} 行不是对象")
        missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{path.name} 第 {lineno} 行缺少字段：{missing}")
        if not isinstance(row["chapter_path"], list):
            raise ValueError(f"{path.name} 第 {lineno} 行 chapter_path 必须是数组")
        rows.append(row)
    return rows


def load_records(settings: Settings | None = None) -> list[ClauseRecord]:
    """从 data/processed 读全部条款，并用 manifest 补上文档标题。"""
    settings = settings or get_settings()
    manifest_path = settings.processed_dir / "manifest.json"
    titles: dict[str, str] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for doc in manifest.get("documents", []):
                titles[doc["doc_id"]] = doc.get("title", "")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            # 标题只影响展示，不应让检索整个不可用；但必须显式告警
            warnings.warn(f"manifest.json 解析失败，文档标题将留空：{exc}", stacklevel=2)

    records: list[ClauseRecord] = []
    seen: dict[str, str] = {}
    for path in sorted(settings.processed_dir.glob("DOC*.jsonl")):
        for row in read_clause_rows(path):
            clause_id = row["clause_id"]
            if clause_id in seen:
                raise ValueError(
                    f"clause_id 重复：{clause_id}（{seen[clause_id]} 与 {path.name}），"
                    "会导致检索结果互相覆盖"
                )
            seen[clause_id] = path.name
            records.append(
                ClauseRecord(
                    clause_id=clause_id,
                    doc_id=row["doc_id"],
                    doc_title=titles.get(row["doc_id"], ""),
                    clause_no=row["clause_no"],
                    chapter_path=row["chapter_path"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    text=row["text"],
                )
            )
    return records


def build_bm25(records: list[ClauseRecord], settings: Settings | None = None) -> Path:
    """建 BM25 索引并落盘（gzip + JSON，只存纯数据）。返回文件路径。"""
    settings = settings or get_settings()
    payload = {
        "version": BM25_FORMAT_VERSION,
        "clause_ids": [record.clause_id for record in records],
        "corpus": [tokenize(record.text) for record in records],
    }
    target = settings.processed_dir / BM25_FILE
    temporary = target.with_suffix(target.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    temporary.replace(target)
    return target


def _drop_collection(client, name: str) -> None:
    """删除集合。只有「不存在」可以忽略，其余失败必须抛出。"""
    try:
        client.delete_collection(name)
    except Exception as exc:
        message = str(exc).lower()
        if "not found" in message or "not exist" in message or "does not exist" in message:
            return
        raise


def _add_in_batches(collection, records: list[ClauseRecord], vectors: list[list[float]]) -> None:
    batch = 128
    for start in range(0, len(records), batch):
        chunk = records[start : start + batch]
        collection.add(
            ids=[record.clause_id for record in chunk],
            embeddings=vectors[start : start + batch],
            documents=[record.text for record in chunk],
            metadatas=[record.metadata() for record in chunk],
        )


def build_chroma(records: list[ClauseRecord], settings: Settings | None = None) -> int:
    """建向量索引。返回写入条数。

    顺序很关键：先完成全部远程向量化，再动旧集合。
    新索引先写进 staging 集合并校验条数，通过后才替换正式集合，
    因此远程调用失败不会让线上索引消失。
    """
    import chromadb

    settings = settings or get_settings()
    client = chromadb.PersistentClient(path=str(settings.chroma_path))

    vectors = embed_texts([record.text for record in records], settings)
    if len(vectors) != len(records):
        raise RuntimeError(f"向量条数 {len(vectors)} 与条款条数 {len(records)} 不一致")

    staging_name = f"{COLLECTION_NAME}__staging"
    metadata = {"hnsw:space": "cosine"}
    _drop_collection(client, staging_name)
    staging = client.create_collection(name=staging_name, metadata=metadata)
    _add_in_batches(staging, records, vectors)
    if staging.count() != len(records):
        raise RuntimeError(f"临时集合条数 {staging.count()} 与条款条数 {len(records)} 不一致")

    try:
        _drop_collection(client, COLLECTION_NAME)
        collection = client.create_collection(name=COLLECTION_NAME, metadata=metadata)
        _add_in_batches(collection, records, vectors)
        if collection.count() != len(records):
            raise RuntimeError(
                f"正式集合条数 {collection.count()} 与条款条数 {len(records)} 不一致"
            )
    except Exception as exc:
        raise RuntimeError(
            f"正式集合替换失败：{exc}。完整新索引仍保留在 {staging_name}，可据此恢复"
        ) from exc

    _drop_collection(client, staging_name)
    return len(records)
