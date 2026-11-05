"""在线检索引擎：把各阶段串成一条**可观测、可降级**的流水线。

完整链路（对应《rag编码注意事项》全部 5 条）::

    query
      │
      ├─[1] 混合粗召回：dense + BM25，各 Top-K → RRF 融合 → Top-30~50
      │        └─ 单路失败自动降级为另一路（不中断）
      │
      ├─[5] 兜底判定：召回不足 → 扩 Top-K 重试（回到 [1]）
      │
      ├─[1] 精排：Cross-Encoder 把 30~50 压到 Top-3~5（唯一进 prompt 的内容）
      │
      ├─[2] Small-to-Big：子块 → 父块全文，按父块去重
      │
      ├─[3] Lost-in-the-Middle：高相关片段放首/尾，末尾重复问题
      │
      ├─[5] 兜底终判：转人工 / 如实说没找到相关信息
      │
      └─[4] diagnostics：阶段计数、分数、重试轨迹，供 Golden Set 回归量化

``RetrievalResult`` 同时暴露 ``coarse``（粗召回）与 ``final``（进 prompt），
让评测能把 Context Recall 与 Context Precision **分开**度量。
"""
from __future__ import annotations

import time
from typing import Any

from .._utils import get_logger
from ..config import RAGConfig
from ..core.interfaces import ContextBuilder, Reranker
from ..exceptions import EmptyRecallError, RetrievalError
from ..models import (
    FallbackContext,
    FallbackStep,
    RetrievalAction,
    RetrievalResult,
    ScoredChunk,
)
from .context import SmallToBigExpander, render_prompt
from .fallback import FallbackChain
from .hybrid import HybridRetriever

logger = get_logger(__name__)


class OnlineRetriever:
    """在线检索门面。

    所有依赖都是接口（``HybridRetriever`` / ``Reranker`` / ``ParentStore`` /
    ``ContextBuilder`` / ``FallbackChain``），可单独替换或打桩测试。
    """

    def __init__(
        self,
        hybrid: HybridRetriever,
        reranker: Reranker,
        expander: SmallToBigExpander,
        context_builder: ContextBuilder,
        fallback_chain: FallbackChain,
        config: RAGConfig,
    ) -> None:
        self.hybrid = hybrid
        self.reranker = reranker
        self.expander = expander
        self.context_builder = context_builder
        self.fallback_chain = fallback_chain
        self.config = config

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

        attempt = 0
        current_top_k = retrieval_cfg.coarse_top_k
        retry_history: list[dict[str, Any]] = []
        retry_steps: list[FallbackStep] = []   # 前面几轮重试中执行过的兜底动作
        recall_diagnostics: dict[str, Any] = {}

        while True:
            # --- [1] 混合粗召回 ---------------------------------------------
            try:
                fused, recall_diagnostics = self.hybrid.recall(
                    query=query,
                    dense_k=max(retrieval_cfg.dense_k, current_top_k),
                    sparse_k=max(retrieval_cfg.sparse_k, current_top_k),
                    limit=current_top_k,
                    metadata_filter=metadata_filter,
                )
            except EmptyRecallError as exc:
                # 两路都异常：不直接抛给终端用户，交给兜底链路（注意事项 5）
                logger.error("粗召回全部失败：%s", exc)
                fused, recall_diagnostics = [], {
                    "per_path": {}, "degraded": True,
                    "both_recallers_failed": True,
                    "error": str(exc), "fused_count": 0,
                }

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

            # --- [1] 精排：严格截断到 rerank_top_n ---------------------------
            ranked = self.reranker.rerank(query, fused, retrieval_cfg.rerank_top_n)

            # --- [2] Small-to-Big 父块还原 -----------------------------------
            final = self.expander.expand(ranked)

            # --- [3] Lost-in-the-Middle 上下文编排 ---------------------------
            built = self.context_builder.build(query, final)
            prompt_text = render_prompt(query, built, self.config.prompt)

            # --- [5] 兜底判定 ------------------------------------------------
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
        logger.info(
            "检索完成：coarse=%d → final=%d，动作=%s，耗时=%.1fms",
            len(coarse), len(final), decision.final_action, elapsed_ms,
        )
        return result


__all__: Sequence[str] = ["OnlineRetriever"]
