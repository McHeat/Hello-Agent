"""兜底链路 —— 《rag编码注意事项》第 5 条。

原文：「RAG 稳不稳，往往取决于『检索失败时兜得全不全』，而非『检索多准』。」

本模块把降级做成**有序、可插拔、可观测**的策略链：

1. :class:`WidenTopKStrategy` —— 召回不足时**扩 Top-K 重试**（带
   ``max_retries`` 与 ``max_coarse_top_k`` 双重硬上限，防止无限放大）；
2. :class:`HumanHandoffStrategy` —— 重试用尽后**转人工**；
3. :class:`HonestNoAnswerStrategy` —— 最后一道：**如实说「没找到相关信息」**，
   绝不把空/低质上下文交给 LLM 自由发挥。

触发条件（``diagnose``）显式列在 ``FallbackDecision.triggers`` 里，便于
统计「检索失败模式」并针对性优化；整条链路动作记在 ``steps`` 里用于审计。
"""
from __future__ import annotations

from typing import Sequence

from .._utils import get_logger
from ..config import FallbackConfig
from ..core.interfaces import FallbackStrategy
from ..core.registry import FALLBACK_STRATEGY_REGISTRY
from ..models import (
    FallbackContext,
    FallbackDecision,
    FallbackStep,
    RetrievalAction,
)

logger = get_logger(__name__)


def diagnose(context: FallbackContext) -> list[str]:
    """判断当前检索结果是否需要兜底，返回**全部**触发原因（可观测）。

    触发条件：

    * 粗候选数低于 ``min_recall_hits``（会按语料规模自动收敛，避免小库误判）；
    * 精排后没有任何候选；
    * 最高精排分低于 ``min_best_score``；
    * 上下文为空。
    """
    reasons: list[str] = []
    effective_min = min(context.min_recall_hits, max(context.corpus_size, 1))
    if context.coarse_count < effective_min:
        reasons.append(
            f"召回不足：粗候选 {context.coarse_count} < {effective_min} "
            f"（语料规模 {context.corpus_size}）"
        )
    if context.final_count == 0:
        reasons.append("精排后无可用片段")
    if context.rerank_scores and max(context.rerank_scores) < context.min_best_score:
        reasons.append(
            f"最高精排分 {max(context.rerank_scores):.4f} < 阈值 {context.min_best_score}"
        )
    if not context.context_text.strip():
        reasons.append("上下文为空")
    return reasons


@FALLBACK_STRATEGY_REGISTRY.register("widen_top_k")
class WidenTopKStrategy(FallbackStrategy):
    """扩 Top-K 重试：把 ``current_top_k`` 乘 ``widen_factor`` 后重跑粗召回。

    达到 ``max_retries`` 或 ``max_coarse_top_k`` 后返回 ``None``，把控制权
    交给下一档策略（转人工 / 如实回答）。
    """

    def evaluate(self, context: FallbackContext) -> FallbackStep | None:
        if context.attempt >= context.max_retries:
            return None
        if context.current_top_k >= context.max_coarse_top_k:
            return None
        new_k = min(
            context.max_coarse_top_k,
            max(context.current_top_k + 1,
                int(context.current_top_k * context.widen_factor)),
        )
        if new_k <= context.current_top_k:
            return None
        return FallbackStep(
            strategy=self.name,
            action="retry",
            reason=f"扩 Top-K 重试：{context.current_top_k} -> {new_k}",
            detail={
                "attempt": context.attempt + 1,
                "top_k": new_k,
                "from_top_k": context.current_top_k,
            },
        )


@FALLBACK_STRATEGY_REGISTRY.register("human_handoff")
class HumanHandoffStrategy(FallbackStrategy):
    """转人工：重试仍不足且业务允许时，返回 ``HANDOFF`` 动作。"""

    def evaluate(self, context: FallbackContext) -> FallbackStep | None:
        if not context.handoff_enabled:
            return None
        return FallbackStep(
            strategy=self.name,
            action=RetrievalAction.HANDOFF,
            reason="扩 Top-K 重试后证据仍不足，转人工处理",
            detail={"message": context.handoff_text},
        )


@FALLBACK_STRATEGY_REGISTRY.register("honest_no_answer")
class HonestNoAnswerStrategy(FallbackStrategy):
    """最后一道：如实说明「没有找到相关信息」，禁止 LLM 编造答案。"""

    def evaluate(self, context: FallbackContext) -> FallbackStep | None:
        return FallbackStep(
            strategy=self.name,
            action=RetrievalAction.NO_ANSWER,
            reason="检索证据不足，按约定如实回答而非生成无依据内容",
            detail={"message": context.no_answer_text},
        )


class FallbackChain:
    """按配置顺序组合兜底策略，产出一次检索的 :class:`FallbackDecision`。"""

    def __init__(self, strategies: Sequence[FallbackStrategy],
                 enabled: bool = True) -> None:
        self.strategies = list(strategies)
        self.enabled = enabled

    @classmethod
    def from_config(cls, config: FallbackConfig) -> "FallbackChain":
        """按 ``fallback.strategies`` 里的名字从注册表构造策略链。"""
        strategies = [
            FALLBACK_STRATEGY_REGISTRY.create(name)
            for name in config.strategies
        ]
        return cls(strategies=strategies, enabled=config.enabled)

    def evaluate(self, context: FallbackContext) -> FallbackDecision:
        """评估是否需要兜底，并返回首个生效策略的动作。"""
        if not self.enabled:
            return FallbackDecision(triggered=False)

        triggers = diagnose(context)
        if not triggers:
            return FallbackDecision(triggered=False)

        decision = FallbackDecision(triggered=True, triggers=triggers)
        logger.info("触发兜底：%s", "；".join(triggers))
        for strategy in self.strategies:
            step = strategy.evaluate(context)
            if step is None:
                continue
            step.detail.setdefault("triggers", triggers)
            decision.steps.append(step)
            if step.action == "retry":
                decision.retry_top_k = int(step.detail["top_k"])
                decision.final_action = RetrievalAction.ANSWER
                decision.message = step.reason
            elif step.action == RetrievalAction.HANDOFF:
                decision.final_action = RetrievalAction.HANDOFF
                decision.message = step.detail.get("message", context.handoff_text)
            else:
                decision.final_action = RetrievalAction.NO_ANSWER
                decision.message = step.detail.get("message", context.no_answer_text)
            break

        if not decision.steps:
            # 策略链配置异常（例如被清空）时的安全兜底：宁可如实回答，也不幻觉
            logger.warning("兜底策略链未产出任何动作，退化为如实回答")
            decision.steps.append(FallbackStep(
                strategy="FallbackChain", action=RetrievalAction.NO_ANSWER,
                reason="策略链为空", detail={"triggers": triggers},
            ))
            decision.final_action = RetrievalAction.NO_ANSWER
            decision.message = context.no_answer_text
        return decision


__all__: Sequence[str] = [
    "diagnose", "WidenTopKStrategy", "HumanHandoffStrategy",
    "HonestNoAnswerStrategy", "FallbackChain",
]
