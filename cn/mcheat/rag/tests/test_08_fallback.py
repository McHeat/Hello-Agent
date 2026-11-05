"""兜底链路：触发判定、扩 Top-K 重试、转人工、如实回答、可观测轨迹。"""
from __future__ import annotations

import unittest

from cn.mcheat.rag import RAGSystem
from cn.mcheat.rag.config import FallbackConfig, RAGConfig
from cn.mcheat.rag.models import FallbackContext, RetrievalAction
from cn.mcheat.rag.retrieval.fallback import (
    FallbackChain,
    HumanHandoffStrategy,
    WidenTopKStrategy,
    diagnose,
)


def _context(**overrides) -> FallbackContext:
    defaults = dict(
        query="测试问题",
        stage="rerank",
        coarse_count=0,
        final_count=0,
        best_score=0.0,
        rerank_scores=[],
        context_text="",
        attempt=0,
        current_top_k=50,
        max_retries=2,
        widen_factor=2.0,
        max_coarse_top_k=200,
        handoff_enabled=True,
        no_answer_text="没有找到相关信息，无法基于现有知识库回答该问题。",
        handoff_text="检索到的资料不足以可靠回答，已转人工处理。",
        min_recall_hits=3,
        min_best_score=0.0,
        corpus_size=0,
    )
    defaults.update(overrides)
    return FallbackContext(**defaults)


class TestDiagnose(unittest.TestCase):
    def test_empty_recall_triggers_multiple_reasons(self):
        reasons = diagnose(_context())
        self.assertGreaterEqual(len(reasons), 3)
        self.assertTrue(any("召回不足" in r for r in reasons))
        self.assertTrue(any("上下文为空" in r for r in reasons))

    def test_healthy_result_does_not_trigger(self):
        reasons = diagnose(_context(
            coarse_count=30, final_count=5, best_score=0.82,
            rerank_scores=[0.82, 0.7], context_text="足够长的资料内容",
            corpus_size=1000,
        ))
        self.assertEqual(reasons, [])

    def test_low_best_score_triggers(self):
        reasons = diagnose(_context(
            coarse_count=30, final_count=5, best_score=0.1,
            rerank_scores=[0.1], context_text="资料",
            min_best_score=0.4, corpus_size=1000,
        ))
        self.assertTrue(any("精排分" in r for r in reasons))

    def test_small_corpus_does_not_false_alarm(self):
        # 语料只有 2 条，不应因为 min_recall_hits=3 而误判
        reasons = diagnose(_context(
            coarse_count=2, final_count=2, best_score=0.9,
            rerank_scores=[0.9], context_text="资料", corpus_size=2,
        ))
        self.assertEqual(reasons, [])


class TestStrategies(unittest.TestCase):
    def test_widen_top_k_produces_retry(self):
        step = WidenTopKStrategy().evaluate(_context())
        self.assertIsNotNone(step)
        self.assertEqual(step.action, "retry")
        self.assertEqual(step.detail["top_k"], 100)

    def test_widen_top_k_respects_limits(self):
        self.assertIsNone(WidenTopKStrategy().evaluate(_context(attempt=2)))
        self.assertIsNone(
            WidenTopKStrategy().evaluate(_context(current_top_k=200))
        )

    def test_handoff_respects_switch(self):
        self.assertIsNone(HumanHandoffStrategy().evaluate(
            _context(handoff_enabled=False)
        ))
        self.assertEqual(
            HumanHandoffStrategy().evaluate(_context()).action,
            RetrievalAction.HANDOFF,
        )


class TestFallbackChain(unittest.TestCase):
    def test_retry_before_handoff(self):
        chain = FallbackChain.from_config(FallbackConfig())
        decision = chain.evaluate(_context())
        self.assertTrue(decision.triggered)
        self.assertTrue(decision.should_retry)
        self.assertEqual(decision.retry_top_k, 100)
        self.assertEqual(len(decision.triggers) >= 3, True)

    def test_exhausted_retries_then_handoff(self):
        chain = FallbackChain.from_config(FallbackConfig())
        decision = chain.evaluate(_context(attempt=2))
        self.assertEqual(decision.final_action, RetrievalAction.HANDOFF)
        self.assertFalse(decision.should_retry)
        self.assertEqual(decision.steps[0].strategy, "HumanHandoffStrategy")

    def test_handoff_disabled_falls_through_to_honest_answer(self):
        chain = FallbackChain.from_config(FallbackConfig(handoff_enabled=False))
        decision = chain.evaluate(_context(attempt=2, handoff_enabled=False))
        self.assertEqual(decision.final_action, RetrievalAction.NO_ANSWER)
        self.assertIn("没有找到相关信息", decision.message)

    def test_disabled_chain_never_triggers(self):
        chain = FallbackChain.from_config(FallbackConfig(enabled=False))
        self.assertFalse(chain.evaluate(_context()).triggered)

    def test_empty_strategy_chain_is_safe(self):
        chain = FallbackChain([], enabled=True)
        decision = chain.evaluate(_context(attempt=2))
        self.assertEqual(decision.final_action, RetrievalAction.NO_ANSWER)


class TestFallbackInPipeline(unittest.TestCase):
    def test_empty_index_retries_then_handsoff(self):
        system = RAGSystem(config=RAGConfig.load(use_env=False))
        result = system.retrieve("任意问题")
        self.assertEqual(result.fallback.final_action, RetrievalAction.HANDOFF)
        self.assertFalse(result.answer_ready)
        self.assertEqual(result.prompt, "")  # 绝不允许把空上下文交给 LLM
        self.assertEqual(len(result.diagnostics["retry_history"]), 2)
        self.assertEqual(result.diagnostics["final_coarse_top_k"], 200)
        # 完整降级梯度可审计：两次扩 Top-K 重试 + 最终转人工
        self.assertEqual(
            [step.action for step in result.fallback.steps],
            ["retry", "retry", RetrievalAction.HANDOFF],
        )

    def test_empty_index_without_handoff_answers_honestly(self):
        cfg = RAGConfig.load(
            overrides={"fallback": {"handoff_enabled": False}}, use_env=False
        )
        result = RAGSystem(config=cfg).retrieve("任意问题")
        self.assertEqual(result.fallback.final_action, RetrievalAction.NO_ANSWER)
        self.assertIn("没有找到相关信息", result.fallback.message)

    def test_fallback_disabled_keeps_answer_action(self):
        cfg = RAGConfig.load(
            overrides={"fallback": {"enabled": False}}, use_env=False
        )
        result = RAGSystem(config=cfg).retrieve("任意问题")
        self.assertTrue(result.answer_ready)
        self.assertFalse(result.fallback.triggered)


if __name__ == "__main__":
    unittest.main()
