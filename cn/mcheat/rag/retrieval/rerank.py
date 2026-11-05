"""精排（Reranker）—— 《rag编码注意事项》第 1 条的后半段。

粗召回的 Top-30~50 性价比很差：直接全塞进 prompt 会稀释关键信息、撑爆
token，还会触发 Lost-in-the-Middle。本模块把候选压到 Top-3~5，编排层
（``retrieval/pipeline.py``）**只把这里的返回值**放进 prompt。

三种实现，体现可插拔：

* :class:`CrossEncoderReranker` —— 生产首选。注入任意符合
  ``CrossEncoderScorer`` 协议的交叉编码器（默认懒加载
  ``sentence_transformers.CrossEncoder``）；query 与 passage 拼接后联合编码，
  比双塔向量更准。
* :class:`LexicalOverlapReranker` —— **零依赖默认实现**，用查询词覆盖率 +
  词项聚集度 + 融合名次先验打分。它不是真正的语义模型，只用于让框架开箱
  即跑、可测试；生产请切换 ``components.reranker.name = "cross_encoder"``。
* :class:`NoopReranker` —— 保持融合顺序，用于消融「精排到底有没有收益」。
"""
from __future__ import annotations

from typing import Any, Sequence

from .._utils import batched, get_logger, tokenize
from ..core.interfaces import CrossEncoderScorer, Reranker
from ..core.registry import RERANKER_REGISTRY
from ..exceptions import ConfigError, DependencyMissingError
from ..models import FusedHit, ScoredChunk

logger = get_logger(__name__)


def _finalize(candidates: Sequence[FusedHit], scores: Sequence[float],
              top_n: int) -> list[ScoredChunk]:
    """统一排序 + 截断 + 填充诊断字段。

    所有 Reranker 都必须走这里，保证「只返回 top_n 条」这一约束不会被
    某个自定义实现无意绕过；同分按 ID 排序，结果可复现。
    """
    if len(candidates) != len(scores):
        raise ConfigError("精排分数数量与候选数量不一致")
    paired = list(zip(candidates, scores))
    paired.sort(key=lambda item: (-float(item[1]), item[0].chunk.id))
    limit = max(0, top_n)
    result: list[ScoredChunk] = []
    for rank, (candidate, score) in enumerate(paired[:limit]):
        result.append(ScoredChunk(
            chunk=candidate.chunk,
            score=float(score),
            rank=rank,
            recall_score=candidate.score,
            recall_sources=tuple(sorted(candidate.sources)),
        ))
    return result


@RERANKER_REGISTRY.register("cross_encoder")
class CrossEncoderReranker(Reranker):
    """交叉编码器精排器。

    Args:
        scorer: 已构造好的打分器，需实现
            ``predict([(query, passage), ...]) -> [float, ...]``。
            与 ``sentence_transformers.CrossEncoder`` 兼容。注入后不触发任何
            第三方 import，也方便用假模型做单元测试。
        model_name: 未提供 ``scorer`` 时懒加载的 HuggingFace 模型名。
        batch_size: 每批送入打分器的 ``(query, passage)`` 对数。
    """

    def __init__(self, scorer: CrossEncoderScorer | Any | None = None,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 batch_size: int = 16) -> None:
        if batch_size <= 0:
            raise ConfigError("batch_size 必须为正整数")
        self._scorer = scorer if scorer is not None else _build_cross_encoder(model_name)
        if not callable(getattr(self._scorer, "predict", None)):
            raise ConfigError("CrossEncoder 打分器必须实现 predict(pairs) 方法")
        self.batch_size = batch_size
        self.model_name = model_name
        logger.info("CrossEncoderReranker 就绪（model=%s, injective=%s）",
                    model_name, scorer is not None)

    def rerank(self, query: str, candidates: Sequence[FusedHit],
               top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores: list[float] = []
        items = list(candidates)
        for batch in batched(items, self.batch_size):
            pairs = [(query, candidate.chunk.text) for candidate in batch]
            try:
                raw = self._scorer.predict(pairs)
            except Exception as exc:
                raise ConfigError(f"CrossEncoder 打分失败：{type(exc).__name__}: {exc}") from exc
            batch_scores = [float(value) for value in raw]
            if len(batch_scores) != len(batch):
                raise ConfigError(
                    f"CrossEncoder 返回 {len(batch_scores)} 个分数，期望 {len(batch)} 个"
                )
            scores.extend(batch_scores)
        return _finalize(items, scores, top_n)


def _build_cross_encoder(model_name: str) -> Any:
    """延迟加载 sentence-transformers 的 CrossEncoder。"""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise DependencyMissingError(
            "sentence-transformers", "Cross-Encoder 精排",
            "或向 CrossEncoderReranker 注入 scorer 对象",
        ) from exc
    return CrossEncoder(model_name)


@RERANKER_REGISTRY.register("lexical")
class LexicalOverlapReranker(Reranker):
    """零依赖的词面重叠精排器（框架默认，仅作占位实现）。

    分数 = ``(覆盖率 * coverage_weight + 聚集度 * proximity_weight)
             / (coverage_weight + proximity_weight) * 0.9 + 融合名次先验 * 0.1``

    * 覆盖率：查询词元在候选中出现的比例；
    * 聚集度：命中的查询词元在候选文本中的分布是否紧凑（越紧凑越可能
      是「整段都在讲这件事」）；
    * 融合名次先验：给混合召回排名靠前的候选一点保底优势。

    Args:
        coverage_weight: 覆盖率权重。
        proximity_weight: 聚集度权重。
    """

    def __init__(self, coverage_weight: float = 0.7,
                 proximity_weight: float = 0.3) -> None:
        if coverage_weight < 0 or proximity_weight < 0 or \
                coverage_weight + proximity_weight <= 0:
            raise ConfigError("精排权重不能为负，且至少有一个为正")
        self.coverage_weight = float(coverage_weight)
        self.proximity_weight = float(proximity_weight)

    def rerank(self, query: str, candidates: Sequence[FusedHit],
               top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        query_terms = set(tokenize(query))
        total = len(candidates)
        scores: list[float] = []
        for index, candidate in enumerate(candidates):
            lexical = self._lexical_score(query_terms, candidate.chunk.text)
            prior = 0.1 * (1.0 - index / max(total, 1))
            scores.append(0.9 * lexical + prior)
        return _finalize(candidates, scores, top_n)

    def _lexical_score(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        tokens = tokenize(text)
        if not tokens:
            return 0.0
        matched = [i for i, term in enumerate(tokens) if term in query_terms]
        if not matched:
            return 0.0
        present_terms = {tokens[i] for i in matched}
        coverage = len(present_terms) / len(query_terms)
        # 聚集度：命中词元跨度越小越好；跨度为 0（单点命中）时取 1.0
        span = matched[-1] - matched[0]
        proximity = 1.0 / (1.0 + span / max(len(query_terms), 1))
        weight_sum = self.coverage_weight + self.proximity_weight
        return (
            self.coverage_weight * coverage + self.proximity_weight * proximity
        ) / weight_sum


@RERANKER_REGISTRY.register("noop")
class NoopReranker(Reranker):
    """不做重打分，直接沿用融合名次（仅截断到 top_n）。

    用于消融实验：把 ``reranker`` 换成 ``"noop"`` 再跑 Golden Set，就能量化
    精排到底带来了多少 Context Precision 提升（注意事项第 4 条）。
    """

    def rerank(self, query: str, candidates: Sequence[FusedHit],
               top_n: int) -> list[ScoredChunk]:
        scores = [candidate.score for candidate in candidates]
        return _finalize(candidates, scores, top_n)


__all__: Sequence[str] = [
    "CrossEncoderReranker", "LexicalOverlapReranker", "NoopReranker",
]
