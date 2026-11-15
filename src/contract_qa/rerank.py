"""重排：用交叉编码器（cross-encoder）对候选条款重新打分。

召回负责「不漏」，重排负责「排对」。
交叉编码器要把「问题 + 条款」拼在一起逐对推理，成本随候选数线性增长，
所以只在融合后的少量候选上做，不能对全量语料做。

三个约束：
1. 分数不与 RRF 分相加——两者量纲不同，相加等于引入一个没调过的超参；
2. 接口只返回 top_n 条，未返回的候选由调用方按原顺序补在尾部，保证结果凑得齐；
3. 任何失败都抛 ApiError 由调用方降级，不在本模块里静默吞掉。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contract_qa.config import Settings, get_settings
from contract_qa.llm import ApiError, post_json


@dataclass
class RerankResult:
    ranking: list[tuple[str, float]]
    ranked_count: int
    missing: list[str] = field(default_factory=list)


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    top_n: int | None = None,
    settings: Settings | None = None,
) -> RerankResult:
    """对 [(clause_id, text)] 重新打分，返回按相关度降序的 [(clause_id, score)]。"""
    settings = settings or get_settings()
    if not candidates:
        return RerankResult(ranking=[], ranked_count=0)
    if not settings.has_rerank:
        raise ApiError("缺少 RERANK_API_KEY 或 RERANK_BASE_URL", retryable=False)

    limit = len(candidates) if top_n is None else max(1, min(top_n, len(candidates)))
    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": [text for _, text in candidates],
        "top_n": limit,
        "return_documents": False,
    }
    data = post_json(f"{settings.rerank_base_url}/rerank", payload, settings.rerank_api_key)

    raw = data.get("results")
    if not isinstance(raw, list):
        raise ApiError("rerank 响应缺少 results 数组")

    ranking: list[tuple[str, float]] = []
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("relevance_score")
        # bool 是 int 的子类，必须单独排除，否则 True 会被当成下标 1
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if not 0 <= index < len(candidates) or index in seen:
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        seen.add(index)
        ranking.append((candidates[index][0], float(score)))

    if not ranking:
        raise ApiError("rerank 未返回任何有效结果")

    # 接口一般已按分数降序，这里再排一次，避免个别实现返回乱序
    ranking.sort(key=lambda item: item[1], reverse=True)
    missing = [cid for index, (cid, _) in enumerate(candidates) if index not in seen]
    return RerankResult(ranking=ranking, ranked_count=len(ranking), missing=missing)
