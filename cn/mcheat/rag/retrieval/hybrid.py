"""混合粗召回 + 多路融合 —— 《rag编码注意事项》第 1 条的前半段。

流程::

    向量通路 (dense, Top-K)  ─┐
                             ├─ 融合(RRF) → 去重 → 粗候选 Top-30~50
    BM25 通路 (sparse, Top-K)┘                      │
                                                   ▼
                                        交给 Reranker 精排到 3~5

**这里只产粗候选，绝不直接进 prompt**。粗召回数量与「进 prompt 数量」在
配置与代码里是两个独立参数（``coarse_top_k`` / ``rerank_top_n``），且有
``rerank_top_n <= coarse_top_k`` 的硬校验。

融合默认用 RRF（Reciprocal Rank Fusion）：只依赖名次、不依赖各通路分数的
量纲，因此余弦分（0~1）与 BM25 分（可能 >10）可以直接混用而无需归一化。
"""
from __future__ import annotations

from typing import Any, Sequence

from .._utils import get_logger
from ..core.interfaces import FusionStrategy, RecallRetriever
from ..core.registry import FUSION_REGISTRY
from ..exceptions import EmptyRecallError, RetrievalError
from ..models import FusedHit, Hit, Ranking

logger = get_logger(__name__)


@FUSION_REGISTRY.register("rrf")
class RRFFusion(FusionStrategy):
    """Reciprocal Rank Fusion。

    每个候选的融合分为 ``sum(weight / (k + rank))``，rank 从 1 开始。
    ``k=60`` 是 Cormack 等人在 TREC 上的经验值，用于压低头部名次的绝对优势。

    Args:
        k: RRF 平滑常数。
        dense_weight / sparse_weight: 按通路名覆盖 ``Ranking.weight`` 的权重，
            可用于「更信任 BM25 的精确匹配」这类调参（配合 Golden Set 验证）。
    """

    def __init__(self, k: int = 60, dense_weight: float = 1.0,
                 sparse_weight: float = 1.0) -> None:
        if k <= 0:
            raise ValueError("RRF 的 k 必须为正整数")
        self.k = k
        self.weights = {"dense": float(dense_weight), "sparse": float(sparse_weight)}

    def fuse(self, rankings: Sequence[Ranking], limit: int) -> list[FusedHit]:
        if limit <= 0:
            return []
        accumulator: dict[str, dict[str, Any]] = {}
        for ranking in rankings:
            weight = self.weights.get(ranking.source, ranking.weight)
            for rank, hit in enumerate(ranking.hits, start=1):
                chunk_id = hit.chunk.id
                entry = accumulator.setdefault(chunk_id, {
                    "chunk": hit.chunk,
                    "score": 0.0,
                    "sources": {},
                    "ranks": {},
                })
                entry["score"] += weight / (self.k + rank)
                entry["sources"][ranking.source] = float(hit.score)
                entry["ranks"][ranking.source] = rank

        # 融合分降序；同分按 ID 排序，保证评测可复现
        ordered = sorted(
            accumulator.values(),
            key=lambda item: (-item["score"], item["chunk"].id),
        )[:limit]
        return [
            FusedHit(
                chunk=item["chunk"],
                score=float(item["score"]),
                sources=dict(item["sources"]),
                ranks=dict(item["ranks"]),
            )
            for item in ordered
        ]


@FUSION_REGISTRY.register("weighted")
class WeightedSumFusion(FusionStrategy):
    """加权分数融合：每路先做 min-max 归一化，再按权重求和。

    作为 RRF 的对照实现，用于验证「RRF 是否真的更好」（注意事项第 4 条：
    别拍脑袋，用 Golden Set 量化）。缺点是受各通路分数分布影响较大。
    """

    def __init__(self, dense_weight: float = 1.0,
                 sparse_weight: float = 1.0,
                 k: int = 60) -> None:
        # ``k`` 是 RRF 的参数，这里刻意接受并忽略：切换 fusion 实现时
        # 配置文件的 params 会与默认值深度合并，保留这个形参可以让
        # "rrf" <-> "weighted" 只改 name 就生效，无需删参数。
        self.weights = {"dense": float(dense_weight), "sparse": float(sparse_weight)}

    def fuse(self, rankings: Sequence[Ranking], limit: int) -> list[FusedHit]:
        if limit <= 0:
            return []
        accumulator: dict[str, dict[str, Any]] = {}
        for ranking in rankings:
            if not ranking.hits:
                continue
            scores = [hit.score for hit in ranking.hits]
            low, high = min(scores), max(scores)
            span = high - low
            weight = self.weights.get(ranking.source, ranking.weight)
            for rank, hit in enumerate(ranking.hits, start=1):
                normalized = 1.0 if span == 0 else (hit.score - low) / span
                entry = accumulator.setdefault(hit.chunk.id, {
                    "chunk": hit.chunk, "score": 0.0,
                    "sources": {}, "ranks": {},
                })
                entry["score"] += weight * normalized
                entry["sources"][ranking.source] = float(hit.score)
                entry["ranks"][ranking.source] = rank

        ordered = sorted(
            accumulator.values(),
            key=lambda item: (-item["score"], item["chunk"].id),
        )[:limit]
        return [
            FusedHit(chunk=item["chunk"], score=float(item["score"]),
                     sources=dict(item["sources"]), ranks=dict(item["ranks"]))
            for item in ordered
        ]


class HybridRetriever:
    """编排「稠密 + 稀疏」两路召回并融合成粗候选。

    单路失败（例如向量库不可用）时**不会让整条链路崩掉**：记录错误后继续
    用另一路，诊断信息里能看到 ``degraded`` 标记，后续由兜底链路决定是否
    扩 Top-K 重试。这正是「RAG 稳不稳取决于检索失败时兜得全不全」的工程落点。

    Args:
        dense / sparse: 两路 :class:`RecallRetriever`。
        fusion: 融合策略。
        enable_dense / enable_sparse: 消融开关。
    """

    def __init__(self, dense: RecallRetriever, sparse: RecallRetriever,
                 fusion: FusionStrategy, enable_dense: bool = True,
                 enable_sparse: bool = True) -> None:
        if not (enable_dense or enable_sparse):
            raise RetrievalError("粗召回两条通路都被关闭")
        self.dense = dense
        self.sparse = sparse
        self.fusion = fusion
        self.enable_dense = enable_dense
        self.enable_sparse = enable_sparse

    def recall(
        self,
        query: str,
        dense_k: int,
        sparse_k: int,
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> tuple[list[FusedHit], dict[str, Any]]:
        """执行一次粗召回。

        Args:
            dense_k / sparse_k: 各路单独召回条数。
            limit: 融合后**粗候选上限**（正常为 ``coarse_top_k``，
                兜底重试时会被放大）。

        Returns:
            ``(粗候选, 诊断信息)``。诊断里含各路命中数、降级标记与
            ``both_recallers_failed``。

        Raises:
            EmptyRecallError: 两条启用的通路**都抛异常**（不是「都返回空」，
                空结果交给兜底链路处理，语义不同）。
        """
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("查询不能为空")
        rankings: list[Ranking] = []
        diagnostics: dict[str, Any] = {
            "query": query, "dense_k": dense_k, "sparse_k": sparse_k,
            "limit": limit, "per_path": {}, "degraded": False,
        }

        paths: list[tuple[str, RecallRetriever, int]] = []
        if self.enable_dense:
            paths.append(("dense", self.dense, dense_k))
        if self.enable_sparse:
            paths.append(("sparse", self.sparse, sparse_k))

        failures = 0
        for source, retriever, k in paths:
            try:
                hits: list[Hit] = retriever.search(query, k, metadata_filter)
                rankings.append(Ranking(source=source, hits=hits))
                diagnostics["per_path"][source] = {
                    "count": len(hits),
                    "top_score": hits[0].score if hits else 0.0,
                }
            except Exception as exc:
                failures += 1
                diagnostics["degraded"] = True
                diagnostics["per_path"][source] = {
                    "count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                logger.warning("粗召回通路 %s 失败，继续用其余通路：%s", source, exc)

        if failures == len(paths):
            raise EmptyRecallError(
                f"所有粗召回通路均失败：{diagnostics['per_path']}"
            )

        fused = self.fusion.fuse(rankings, limit)
        diagnostics["fused_count"] = len(fused)
        diagnostics["fusion"] = self.fusion.name
        return fused, diagnostics


__all__: Sequence[str] = ["RRFFusion", "WeightedSumFusion", "HybridRetriever"]
