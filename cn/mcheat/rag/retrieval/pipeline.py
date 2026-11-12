"""在线检索引擎：把各阶段串成一条**可观测、可降级**的流水线。

完整链路（对应《rag编码注意事项》全部 5 条 + P0/P1 新增能力）::

    query
      ├─[P1] 查询变换 → 结果缓存 → 多查询分别粗召回 → 复用 FusionStrategy 融合
      ├─[1] 混合粗召回：dense + BM25（可并行，默认串行）→ RRF → Top-30~50
      ├─[5] 兜底判定：召回不足 → 扩 Top-K 重试（回到粗召回，受调用预算约束）
      ├─[1] 精排：归一化分 [0,1]，严格 Top-3~5（唯一进 prompt 的内容）
      ├─[2] Small-to-Big：子块 → 父块全文，按父块去重
      ├─[3] Lost-in-the-Middle：高相关片段放首/尾，末尾重复问题
      ├─[5] 兜底终判：转人工 / 如实说没找到相关信息
      └─[4] diagnostics：阶段计数、分数、重试轨迹、阶段耗时、缓存与版本

``RetrievalResult`` 同时暴露 ``coarse``（粗召回）与 ``final``（进 prompt），
让评测能把 Context Recall 与 Context Precision **分开**度量。注意
``coarse[].score`` 保持 RRF 融合域（≈0.016 量级），**不纳入归一化契约**，
与 ``final[].score``（``[0,1]``）量纲不同，不可直接比较。
"""
from __future__ import annotations

import json
import time
from copy import copy
from typing import Any, Callable, Sequence

from .._utils import get_logger, stable_id
from ..config import RAGConfig
from ..core.interfaces import ContextBuilder, QueryTransformer, Reranker
from ..exceptions import EmptyRecallError, RetrievalError
from ..models import (
    FallbackContext,
    FallbackStep,
    FusedHit,
    Hit,
    Ranking,
    RetrievalAction,
    RetrievalResult,
    ScoredChunk,
)
from .cache import NullCache
from .context import SmallToBigExpander, render_prompt
from .fallback import FallbackChain
from .hybrid import HybridRetriever
from .query_transform import IdentityQueryTransformer, QueryTransformPipeline

logger = get_logger(__name__)

STAGE_NAMES: tuple[str, ...] = (
    "transform", "recall", "fusion", "rerank", "expand", "context", "fallback",
)


def _filter_key(metadata_filter: dict[str, Any] | None) -> str:
    """规范化元数据过滤条件，保证缓存 key 稳定。"""
    return json.dumps(metadata_filter or {}, sort_keys=True,
                      ensure_ascii=False, default=str)


def _config_digest(config: RAGConfig) -> str:
    """影响检索结果的配置摘要，纳入结果缓存 key。"""
    r = config.retrieval
    return stable_id(
        r.min_best_score, r.coarse_top_k, r.rerank_top_n,
        r.enable_dense, r.enable_sparse, r.parallel_recall,
        config.component("reranker").name,
        config.component("context_builder").name,
        config.fallback.enabled, config.prompt.system_instruction,
    )


class OnlineRetriever:
    """在线检索门面。

    所有依赖都是接口（``HybridRetriever`` / ``Reranker`` / ``ParentStore`` /
    ``ContextBuilder`` / ``FallbackChain``），可单独替换或打桩测试。新增的
    查询变换与缓存都有零依赖默认实现，关闭时行为与改造前一致。
    """

    def __init__(
        self,
        hybrid: HybridRetriever,
        reranker: Reranker,
        expander: SmallToBigExpander,
        context_builder: ContextBuilder,
        fallback_chain: FallbackChain,
        config: RAGConfig,
        query_transformer: QueryTransformer | None = None,
        cache: Any | None = None,
        version_provider: Callable[[], int] | None = None,
    ) -> None:
        self.hybrid = hybrid
        self.reranker = reranker
        self.expander = expander
        self.context_builder = context_builder
        self.fallback_chain = fallback_chain
        self.config = config
        self.cache = cache or NullCache()
        self.version_provider = version_provider
        max_variants = config.retrieval.max_query_variants
        on_error = config.fallback.on_transform_error
        if isinstance(query_transformer, QueryTransformPipeline):
            self.transformer = query_transformer
        elif query_transformer is not None:
            self.transformer = QueryTransformPipeline(
                [query_transformer], max_variants=max_variants,
                on_error=on_error,
            )
        else:
            self.transformer = QueryTransformPipeline(
                [IdentityQueryTransformer()], max_variants=max_variants,
                on_error=on_error,
            )

    def retrieve(self, query: str,
                 metadata_filter: dict[str, Any] | None = None) -> RetrievalResult:
        """执行一次完整检索。

        Args:
            query: 用户问题。
            metadata_filter: 精确元数据过滤（如按租户 / 来源限定检索范围），
                会同时传给两路召回。

        Returns:
            :class:`RetrievalResult`。当 ``result.answer_ready`` 为 ``False``
            时，调用方**必须**直接使用 ``result.fallback.message`` 作为回答，
            不要把 ``result.prompt`` 交给 LLM。

        Raises:
            RetrievalError: 查询为空或组件配置非法。
        """
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("查询不能为空字符串")
        started = time.perf_counter()
        retrieval_cfg = self.config.retrieval
        fallback_cfg = self.config.fallback
        timings: dict[str, float] = {name: 0.0 for name in STAGE_NAMES}

        # --- [P1] 查询变换（失败一律降级为 [原查询]）--------------------------
        mark = time.perf_counter()
        variants = [
            str(variant).strip() for variant in self.transformer.transform(query)
            if str(variant).strip()
        ][:retrieval_cfg.max_query_variants]
        if not variants:
            variants = [query]
        if variants[0] != query:
            variants.insert(0, query)
            variants = variants[:retrieval_cfg.max_query_variants]
        timings["transform"] += (time.perf_counter() - mark) * 1000.0

        # --- [P1] 结果缓存（key 含 index_version，索引更新后自然失效）--------
        index_version = self.version_provider() if self.version_provider else 0
        cache_key = (query, _filter_key(metadata_filter),
                     _config_digest(self.config))
        cached = self.cache.get_result(cache_key)
        if cached is not None:
            # 返回副本：命中标记只属于本次结果，不污染缓存中的对象
            hit_result = copy(cached)
            hit_result.diagnostics = {**cached.diagnostics, "cache_hit": True}
            logger.debug("检索结果缓存命中，直接返回")
            return hit_result

        attempt = 0
        current_top_k = retrieval_cfg.coarse_top_k
        recall_calls = 0
        retry_history: list[dict[str, Any]] = []
        retry_steps: list[FallbackStep] = []
        recall_diagnostics: dict[str, Any] = {}

        while True:
            # --- [1] 混合粗召回（多查询分别召回后复用融合器）-----------------
            mark = time.perf_counter()
            per_variant: list[list[FusedHit]] = []
            variant_diags: list[dict[str, Any]] = []
            for variant in variants:
                if recall_calls >= retrieval_cfg.max_recall_calls:
                    break
                try:
                    fused, diagnostics = self.hybrid.recall(
                        query=variant,
                        dense_k=max(retrieval_cfg.dense_k, current_top_k),
                        sparse_k=max(retrieval_cfg.sparse_k, current_top_k),
                        limit=current_top_k,
                        metadata_filter=metadata_filter,
                    )
                    per_variant.append(fused)
                    variant_diags.append(diagnostics)
                except EmptyRecallError as exc:
                    logger.error("粗召回全部失败：%s", exc)
                    per_variant.append([])
                    variant_diags.append({
                        "per_path": {}, "degraded": True,
                        "both_recallers_failed": True,
                        "error": str(exc), "fused_count": 0,
                    })
                recall_calls += 1
            timings["recall"] += (time.perf_counter() - mark) * 1000.0

            if len(per_variant) > 1:
                mark = time.perf_counter()
                rankings = [
                    Ranking(
                        source=f"query_{index}",
                        hits=[
                            Hit(chunk=item.chunk, score=item.score,
                                source=f"query_{index}")
                            for item in fused
                        ],
                    )
                    for index, fused in enumerate(per_variant)
                ]
                fused = self.hybrid.fusion.fuse(rankings, current_top_k)
                timings["fusion"] += (time.perf_counter() - mark) * 1000.0
                recall_diagnostics = dict(variant_diags[0])
                recall_diagnostics["fused_count"] = len(fused)
                recall_diagnostics["query_variants"] = len(per_variant)
            elif per_variant:
                fused = per_variant[0]
                recall_diagnostics = variant_diags[0]
            else:
                fused, recall_diagnostics = [], {
                    "per_path": {}, "degraded": True,
                    "both_recallers_failed": True, "fused_count": 0,
                }

            # coarse 保持 RRF 融合域（≈0.016 量级），不纳入 final 的 [0,1] 契约
            coarse: list[ScoredChunk] = [
                ScoredChunk(
                    chunk=item.chunk,
                    score=item.score,
                    rank=index,
                    recall_score=item.score,
                    recall_sources=tuple(sorted(item.sources)),
                )
                for index, item in enumerate(fused)
            ]

            # --- [1] 精排：严格截断到 rerank_top_n，只用主查询控制 CE 成本 ---
            mark = time.perf_counter()
            ranked = self.reranker.rerank(query, fused, retrieval_cfg.rerank_top_n)
            timings["rerank"] += (time.perf_counter() - mark) * 1000.0

            # --- [2] Small-to-Big 父块还原 -----------------------------------
            mark = time.perf_counter()
            final = self.expander.expand(ranked)
            timings["expand"] += (time.perf_counter() - mark) * 1000.0

            # --- [3] Lost-in-the-Middle 上下文编排 ---------------------------
            mark = time.perf_counter()
            built = self.context_builder.build(query, final)
            prompt_text = render_prompt(query, built, self.config.prompt)
            timings["context"] += (time.perf_counter() - mark) * 1000.0

            # --- [5] 兜底判定（基于 final 的归一化分）-------------------------
            mark = time.perf_counter()
            corpus_size = max(
                self.hybrid.dense.count() if self.hybrid.enable_dense else 0,
                self.hybrid.sparse.count() if self.hybrid.enable_sparse else 0,
            )
            fallback_ctx = FallbackContext(
                query=query,
                stage="rerank",
                coarse_count=len(coarse),
                final_count=len(final),
                best_score=max((item.score for item in ranked), default=0.0),
                rerank_scores=[item.score for item in ranked],
                context_text=built.text,
                attempt=attempt,
                current_top_k=current_top_k,
                max_retries=fallback_cfg.max_retries,
                widen_factor=fallback_cfg.widen_factor,
                max_coarse_top_k=fallback_cfg.max_coarse_top_k,
                handoff_enabled=fallback_cfg.handoff_enabled,
                no_answer_text=fallback_cfg.no_answer_text,
                handoff_text=fallback_cfg.handoff_text,
                min_recall_hits=retrieval_cfg.min_recall_hits,
                min_best_score=retrieval_cfg.min_best_score,
                corpus_size=corpus_size,
            )
            decision = self.fallback_chain.evaluate(fallback_ctx)
            timings["fallback"] += (time.perf_counter() - mark) * 1000.0

            if decision.should_retry and decision.retry_top_k is not None:
                retry_history.append({
                    "attempt": attempt,
                    "from_top_k": current_top_k,
                    "to_top_k": decision.retry_top_k,
                    "reason": decision.triggers,
                })
                retry_steps.extend(decision.steps)
                attempt += 1
                current_top_k = decision.retry_top_k
                logger.info("兜底重试 %d：Top-K 放大到 %d", attempt, current_top_k)
                continue
            break

        # 把重试轮的动作并入最终结论，让 steps 呈现完整的降级梯度
        if retry_steps:
            decision.steps = retry_steps + decision.steps

        # 转人工 / 如实回答时，绝不允许把 prompt 交给 LLM
        if decision.final_action != RetrievalAction.ANSWER:
            prompt_text = ""

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        diagnostics: dict[str, Any] = {
            "attempts": attempt + 1,
            "final_coarse_top_k": current_top_k,
            "coarse_count": len(coarse),
            "final_count": len(final),
            "best_rerank_score": max((item.score for item in ranked), default=0.0),
            "reranker": self.reranker.name,
            "context_builder": self.context_builder.name,
            "context_placement": getattr(self.context_builder, "placement", "unknown"),
            "context_chars": len(built.text),
            "parent_expanded": sum(1 for item in final if item.parent is not None),
            "recall": recall_diagnostics,
            "fallback": {
                "triggered": decision.triggered,
                "triggers": decision.triggers,
                "final_action": decision.final_action,
                "steps": [
                    {"strategy": step.strategy, "action": step.action,
                     "reason": step.reason, "detail": step.detail}
                    for step in decision.steps
                ],
            },
            "retry_history": retry_history,
            "elapsed_ms": round(elapsed_ms, 3),
            "stage_timings_ms": {
                name: round(value, 3) for name, value in timings.items()
            },
            "query_variants": list(variants),
            "cache_hit": False,
            "index_version": index_version,
            "recall_calls": recall_calls,
        }

        result = RetrievalResult(
            query=query,
            coarse=coarse,
            final=final,
            context=built,
            prompt=prompt_text,
            fallback=decision,
            diagnostics=diagnostics,
        )
        self.cache.put_result(cache_key, result)
        logger.info(
            "检索完成：coarse=%d → final=%d，动作=%s，耗时=%.1fms",
            len(coarse), len(final), decision.final_action, elapsed_ms,
        )
        return result


__all__: Sequence[str] = ["OnlineRetriever"]
