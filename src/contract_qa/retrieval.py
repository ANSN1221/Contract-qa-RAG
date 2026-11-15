"""检索：BM25 与向量双路召回，用加权 RRF 融合。

为什么用 RRF 而不是加权分数：BM25 分数无上界、余弦相似度在 0–1 之间，
加权求和必须先做归一化，而归一化方式本身又是一个需要调的超参。
RRF 只看排名，天然免疫量纲问题。

为什么默认权重 1:1：D5 三轮评测（10 题、20 题、切分规整后 20 题）一致显示
1:1 优于 2:1。权重比超过 (k+召回深度)/(k+1) ≈ 1.31 时，权重高的一路的
长尾会压过另一路的头部，把单路命中的答案挤出候选池（Q06 实测：向量第 1 名
被 BM25 第 20 名压制到融合第 21 名）。
权重来自 Settings（.env 与进程环境变量都生效），消融实验时可以直接改。
"""

from __future__ import annotations

import gzip
import json
import warnings
from dataclasses import dataclass
from functools import lru_cache

from contract_qa.config import Settings, get_settings
from contract_qa.index import (
    BM25_FILE,
    COLLECTION_NAME,
    LEGACY_BM25_FILE,
    ClauseRecord,
    load_records,
    tokenize,
)
from contract_qa.llm import ApiError, embed_texts
from contract_qa.rerank import rerank

RRF_K = 60


@dataclass
class Hit:
    clause_id: str
    score: float            # 融合分（RRF）
    dense_rank: int | None  # 向量召回排名，1 起
    bm25_rank: int | None   # 关键词召回排名，1 起
    record: ClauseRecord
    rerank_score: float | None = None  # 重排分，未启用重排时为 None

    @property
    def sort_key(self) -> float:
        """多路结果合并时的排序依据：重排分优先，否则退回融合分。"""
        return self.rerank_score if self.rerank_score is not None else self.score


def fuse_rankings(
    dense_ids: list[str],
    sparse_ids: list[str],
    *,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """加权 RRF 融合。生产与测试共用这一个实现，避免测试里另写一套公式。"""
    fused: dict[str, float] = {}
    for ids, weight in ((dense_ids, dense_weight), (sparse_ids, bm25_weight)):
        for rank, clause_id in enumerate(ids, start=1):
            fused[clause_id] = fused.get(clause_id, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


class Retriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.records = {record.clause_id: record for record in load_records(self.settings)}
        self.bm25_weight = self.settings.bm25_weight
        self.dense_weight = self.settings.dense_weight
        self.index_warning = ""
        self.rerank_error = ""
        self._bm25 = None
        self._clause_ids: list[str] = []
        self._collection = None

    # ---- 懒加载：只用 BM25 时不需要向量库，反之亦然 ----

    def _load_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            path = self.settings.processed_dir / BM25_FILE
            legacy = self.settings.processed_dir / LEGACY_BM25_FILE
            if not path.exists():
                hint = (
                    "检测到旧格式 bm25.pkl，请重新运行 contract-qa index 重建索引"
                    if legacy.exists()
                    else "请先运行 contract-qa index"
                )
                raise FileNotFoundError(f"缺少 BM25 索引 {path}，{hint}")
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
            clause_ids = payload["clause_ids"]
            corpus = payload["corpus"]
            if len(clause_ids) != len(corpus):
                raise ValueError(
                    f"BM25 索引损坏：编号 {len(clause_ids)} 条、语料 {len(corpus)} 条"
                )
            self._bm25 = BM25Okapi(corpus)
            self._clause_ids = clause_ids
            self._check_index_consistency()
        return self._bm25

    def _check_index_consistency(self) -> None:
        """索引与条款文件不同步时必须显式告警，不能静默少返回。"""
        known = set(self.records)
        indexed = set(self._clause_ids)
        stale = sorted(indexed - known)
        missing = sorted(known - indexed)
        if stale or missing:
            self.index_warning = (
                f"BM25 索引与条款文件不一致：索引独有 {len(stale)} 条、"
                f"条款文件独有 {len(missing)} 条，请运行 contract-qa index 重建"
            )
            warnings.warn(self.index_warning, stacklevel=3)

    def _load_collection(self):
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
            self._collection = client.get_collection(COLLECTION_NAME)
        return self._collection

    # ---- 两路召回 ----

    def bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        index = self._load_bm25()
        scores = index.get_scores(tokenize(query))
        if len(scores) != len(self._clause_ids):
            raise ValueError("BM25 分数数量与条款编号数量不一致，请重建索引")
        ranked = sorted(
            zip(self._clause_ids, scores, strict=True), key=lambda item: item[1], reverse=True
        )
        return [(cid, float(score)) for cid, score in ranked[:top_k] if score > 0]

    def dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        collection = self._load_collection()
        vector = embed_texts([query], self.settings)[0]
        result = collection.query(query_embeddings=[vector], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        # score 仅用于展示与排查，融合只看排名
        return [(cid, 1.0 - float(dist)) for cid, dist in zip(ids, distances, strict=False)]

    # ---- 融合 ----

    def _should_rerank(self, use_rerank: bool | None) -> bool:
        """显式参数优先于配置；开了但没配密钥时记下原因并跳过。"""
        enabled = self.settings.rerank_enabled if use_rerank is None else use_rerank
        if not enabled:
            return False
        if not self.settings.has_rerank:
            self.rerank_error = "已开启重排，但缺少 RERANK_API_KEY 或 RERANK_BASE_URL"
            return False
        return True

    def search(
        self,
        query: str,
        top_k: int = 8,
        candidates: int = 50,
        *,
        use_rerank: bool | None = None,
    ) -> list[Hit]:
        if top_k <= 0:
            return []
        dense = self.dense_search(query, candidates)
        sparse = self.bm25_search(query, candidates)

        ordered = fuse_rankings(
            [cid for cid, _ in dense],
            [cid for cid, _ in sparse],
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
        )
        dense_rank = {cid: rank for rank, (cid, _) in enumerate(dense, start=1)}
        bm25_rank = {cid: rank for rank, (cid, _) in enumerate(sparse, start=1)}
        fused_scores = dict(ordered)

        # 先过滤失效编号再截断，避免结果不足 top_k
        pool = [cid for cid, _ in ordered if cid in self.records]
        pool = pool[: max(self.settings.rerank_candidates, top_k)]

        rerank_scores: dict[str, float] = {}
        if pool and self._should_rerank(use_rerank):
            try:
                result = rerank(
                    query,
                    [(cid, self.records[cid].text) for cid in pool],
                    top_n=len(pool),
                    settings=self.settings,
                )
                self.rerank_error = ""
                rerank_scores = dict(result.ranking)
                head = [cid for cid, _ in result.ranking]
            except ApiError as exc:
                # 重排故障不能拖垮问答：退回融合顺序，并留下可排查的原因
                self.rerank_error = str(exc)
                head = []
            seen = set(head)
            ordered_ids = head + [cid for cid in pool if cid not in seen]
        else:
            ordered_ids = pool

        hits = [
            Hit(
                clause_id=cid,
                score=fused_scores.get(cid, 0.0),
                dense_rank=dense_rank.get(cid),
                bm25_rank=bm25_rank.get(cid),
                record=self.records[cid],
                rerank_score=rerank_scores.get(cid),
            )
            for cid in ordered_ids[:top_k]
        ]
        return hits


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


def reset_retriever() -> None:
    """重建索引或改配置后调用，避免长驻进程一直用旧缓存。"""
    get_retriever.cache_clear()
