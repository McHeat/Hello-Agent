"""精排（Reranker）—— 《rag编码注意事项》第 1 条的后半段。

粗召回的 Top-30~50 性价比很差：直接全塞进 prompt 会稀释关键信息、撑爆
token，还会触发 Lost-in-the-Middle。本模块把候选压到 Top-3~5，编排层
（``retrieval/pipeline.py``）**只把这里的返回值**放进 prompt。

三种实现，体现可插拔：

* :class:`CrossEncoderReranker` —— 生产首选（默认中文模型
  ``BAAI/bge-reranker-large``）。注入任意符合 ``CrossEncoderScorer``
  协议的交叉编码器；原始 logit 经 sigmoid 归一化到 ``[0,1]``。
* :class:`LexicalOverlapReranker` —— **零依赖默认实现**，用查询词覆盖率 +
  词项聚集度打分；融合名次先验仅作同分 tie-break，不再给所有候选保底地板。
* :class:`NoopReranker` —— 保持融合顺序，用于消融；分数按名次映射拉伸
  到 ``[0,1]``，保证非零阈值不会把消融结果团灭。

**统一契约**：``Reranker.normalize_scores()`` 把原始分映射到 ``[0,1]``；
``ScoredChunk.score`` 存归一化分（质量门禁唯一口径），``raw_score`` 存原始分。
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

from .._utils import batched, get_logger, tokenize
from ..core.interfaces import CrossEncoderScorer, Reranker
from ..core.registry import RERANKER_REGISTRY
from ..exceptions import ConfigError, DependencyMissingError
from ..models import FusedHit, ScoredChunk

logger = get_logger(__name__)


def _finalize(
    candidates: Sequence[FusedHit],
    scores: Sequence[float],
    top_n: int,
    normalize: Callable[[Sequence[float]], list[float]],
    tie_breakers: Sequence[float] | None = None,
) -> list[ScoredChunk]:
    """统一排序 + 归一化 + 截断 + 填充诊断字段。

    所有 Reranker 都必须走这里，保证「只返回 top_n 条」与「score 恒为
    ``[0,1]`` 归一化分」这两个约束不会被某个自定义实现无意绕过。
    ``tie_breakers`` 只影响同分排序，不参与最终分数。
    """
    if len(candidates) != len(scores):
        raise ConfigError("精排分数数量与候选数量不一致")
    ties = list(tie_breakers) if tie_breakers is not None \
        else [0.0] * len(scores)
    if len(ties) != len(scores):
        raise ConfigError("精排 tie-break 数量与候选数量不一致")

    order = list(range(len(scores)))
    order.sort(key=lambda i: (-float(scores[i]), -float(ties[i]),
                              candidates[i].chunk.id))
    normalized = normalize([float(scores[i]) for i in order])
    if len(normalized) != len(order):
        raise ConfigError("normalize_scores 返回数量与候选数量不一致")

    result: list[ScoredChunk] = []
    for rank, index in enumerate(order[:max(0, top_n)]):
        candidate = candidates[index]
        result.append(ScoredChunk(
            chunk=candidate.chunk,
            score=float(normalized[rank]),
            rank=rank,
            recall_score=candidate.score,
            recall_sources=tuple(sorted(candidate.sources)),
            raw_score=float(scores[index]),
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
        model_name: 未提供 ``scorer`` 时懒加载的 HuggingFace 模型名
            （默认中文 ``BAAI/bge-reranker-large``）。
        batch_size: 每批送入打分器的 ``(query, passage)`` 对数。
    """

    def __init__(self, scorer: CrossEncoderScorer | None = None,
                 model_name: str = "BAAI/bge-reranker-large",
                 batch_size: int = 16) -> None:
        if batch_size <= 0:
            raise ConfigError("batch_size 必须为正整数")
        self._scorer = (scorer if scorer is not None
                        else _build_cross_encoder(model_name))
        if not callable(getattr(self._scorer, "predict", None)):
            raise ConfigError("CrossEncoder 打分器必须实现 predict(pairs) 方法")
        self.batch_size = batch_size
        self.model_name = model_name
        logger.info("CrossEncoderReranker 就绪（model=%s, injected=%s）",
                    model_name, scorer is not None)

    def normalize_scores(self, raw_scores: Sequence[float]) -> list[float]:
        """交叉编码器原始 logit → sigmoid，值域 ``(0, 1)``。"""
        normalized: list[float] = []
        for raw in raw_scores:
            value = float(raw)
            if value >= 0.0:
                normalized.append(1.0 / (1.0 + math.exp(-value)))
            else:
                exp_value = math.exp(value)
                normalized.append(exp_value / (1.0 + exp_value))
        return normalized

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
        return _finalize(items, scores, top_n, self.normalize_scores)


def _build_cross_encoder(model_name: str):
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
             / (coverage_weight + proximity_weight) * 0.9``

    * 覆盖率：查询词元在候选中出现的比例；
    * 聚集度：命中的查询词元在候选文本中的分布是否紧凑；
    * 融合名次先验 ``0.1 * (1 - i/n)`` **已从最终分中移除**，仅作同分
      tie-break，避免给无关查询 0.38~0.44 的保底分。

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
        ties: list[float] = []
        for index, candidate in enumerate(candidates):
            lexical = self._lexical_score(query_terms, candidate.chunk.text)
            scores.append(0.9 * lexical)
            ties.append(0.1 * (1.0 - index / max(total, 1)))
        return _finalize(candidates, scores, top_n, self.normalize_scores, ties)

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

    消融实验用的分数按名次映射 ``1 - rank / n`` 拉伸到 ``[0,1]``，避免
    RRF 的 ≈0.016 量级被非零 ``min_best_score`` 团灭；若需完全排除阈值
    干扰，可临时把 ``min_best_score`` 置 0 旁路门禁（框架会输出 WARNING）。
    """

    def normalize_scores(self, raw_scores: Sequence[float]) -> list[float]:
        """名次映射：``score = 1 - rank / n``（rank 从 0 起，top1 ≈ 1.0）。"""
        total = len(raw_scores)
        if total == 0:
            return []
        return [1.0 - rank / total for rank in range(total)]

    def rerank(self, query: str, candidates: Sequence[FusedHit],
               top_n: int) -> list[ScoredChunk]:
        scores = [candidate.score for candidate in candidates]
        return _finalize(candidates, scores, top_n, self.normalize_scores)


__all__: Sequence[str] = [
    "CrossEncoderReranker", "LexicalOverlapReranker", "NoopReranker",
]
