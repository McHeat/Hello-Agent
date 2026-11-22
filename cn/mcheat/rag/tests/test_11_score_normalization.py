"""P0-1：精排分数归一化契约、质量门禁、阈值标定与拒答行为。"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

from cn.mcheat.rag import RAGConfig, RAGSystem
from cn.mcheat.rag.config import FallbackConfig
from cn.mcheat.rag.evaluation import GoldenSet
from cn.mcheat.rag.models import FallbackContext, RetrievalAction
from cn.mcheat.rag.retrieval.fallback import FallbackChain
from cn.mcheat.rag.retrieval.rerank import (
    CrossEncoderReranker,
    LexicalOverlapReranker,
    NoopReranker,
)

from cn.mcheat.rag.tests._helpers import make_fused

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
GOLDEN_SET = Path(__file__).resolve().parents[1] / "examples" / "golden_set.json"


class _ConstantScorer:
    """满足 CrossEncoderScorer Protocol 的零依赖打分器。"""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def predict(self, pairs):
        return [self.value for _ in pairs]


def _healthy_context(**overrides) -> FallbackContext:
    defaults = dict(
        query="q", stage="rerank", coarse_count=30, final_count=5,
        best_score=0.9, rerank_scores=[0.9], context_text="足够长的资料内容",
        attempt=0, current_top_k=50, max_retries=2, widen_factor=2.0,
        max_coarse_top_k=200, handoff_enabled=True,
        no_answer_text="没有找到相关信息，无法基于现有知识库回答该问题。",
        handoff_text="检索到的资料不足以可靠回答，已转人工处理。",
        min_recall_hits=3, min_best_score=0.35, corpus_size=1000,
    )
    defaults.update(overrides)
    return FallbackContext(**defaults)


class TestNormalizationContract(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            make_fused("c1", "完全无关的天气描述", 0.030),
            make_fused("c2", "父子切分 Small-to-Big 用小片段检索返回大段落", 0.025),
            make_fused("c3", "又一段无关内容", 0.020),
        ]

    def test_default_contract_clamps_to_unit_interval(self):
        reranker = LexicalOverlapReranker()
        self.assertEqual(
            reranker.normalize_scores([-1.0, 0.25, 0.5, 2.0]),
            [0.0, 0.25, 0.5, 1.0],
        )

    def test_cross_encoder_sigmoid_is_numerically_correct(self):
        reranker = CrossEncoderReranker(scorer=_ConstantScorer(), batch_size=2)
        scores = reranker.normalize_scores([0.0, 2.0, -2.0, 1000.0])
        self.assertAlmostEqual(scores[0], 0.5)
        self.assertAlmostEqual(scores[1], 1 / (1 + math.exp(-2)))
        self.assertAlmostEqual(scores[2], 1 / (1 + math.exp(2)))
        self.assertAlmostEqual(scores[3], 1.0)  # 不溢出

    def test_noop_uses_rank_mapping_stretched_to_unit_interval(self):
        reranker = NoopReranker()
        scores = reranker.normalize_scores([0.016, 0.015, 0.014])
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 2 / 3)
        self.assertAlmostEqual(scores[2], 1 / 3)

    def test_all_rerankers_emit_normalized_score_and_raw_score(self):
        for reranker in (
            LexicalOverlapReranker(),
            CrossEncoderReranker(scorer=_ConstantScorer()),
            NoopReranker(),
        ):
            with self.subTest(reranker=type(reranker).__name__):
                result = reranker.rerank(
                    "父子切分 Small-to-Big", self.candidates, top_n=3
                )
                self.assertEqual(len(result), 3)
                for item in result:
                    self.assertGreaterEqual(item.score, 0.0)
                    self.assertLessEqual(item.score, 1.0)
                    self.assertIsInstance(item.raw_score, float)
                self.assertEqual(
                    [item.score for item in result],
                    sorted((item.score for item in result), reverse=True),
                )

    def test_lexical_prior_is_only_a_tie_breaker(self):
        # prior 地板移除后，完全无关的候选分数应为 0（原先至少有 0.1 的地板）
        result = LexicalOverlapReranker().rerank(
            "完全无关的量子色动力学",
            [make_fused("x", "甲乙丙丁", 0.03)], top_n=1,
        )
        self.assertAlmostEqual(result[0].score, 0.0)
        self.assertAlmostEqual(result[0].raw_score, 0.0)

    def test_noop_ablation_is_not_killed_by_nonzero_threshold(self):
        cfg = RAGConfig.load(use_env=False)
        system = RAGSystem(config=cfg, overrides={"reranker": NoopReranker()})
        system.index([DATA_DIR])
        result = system.retrieve("父子切分 Small-to-Big 解决什么问题")
        self.assertTrue(result.answer_ready)
        self.assertTrue(result.final)
        self.assertGreaterEqual(result.final[0].score,
                                cfg.retrieval.min_best_score)

    def test_threshold_blocks_low_quality_results(self):
        cfg = RAGConfig.load(
            overrides={"retrieval": {"min_best_score": 0.99}}, use_env=False
        )
        system = RAGSystem(config=cfg)
        system.index([DATA_DIR])
        result = system.retrieve(
            "父子切分 Small-to-Big 是怎么解决 chunk 大小两难的"
        )
        self.assertFalse(result.answer_ready)
        self.assertEqual(result.prompt, "")
        self.assertIn(result.fallback.final_action,
                      {RetrievalAction.HANDOFF, RetrievalAction.NO_ANSWER})

    def test_gate_closed_by_zero_threshold_emits_warning(self):
        with self.assertLogs("cn.mcheat.rag", level="WARNING") as logs:
            decision = FallbackChain.from_config(FallbackConfig()).evaluate(
                _healthy_context(min_best_score=0.0)
            )
        self.assertFalse(decision.triggered)
        self.assertTrue(any("质量门禁" in line for line in logs.output))

    def test_golden_positives_are_never_falsely_refused(self):
        system = RAGSystem(config=RAGConfig.load(use_env=False))
        system.index([DATA_DIR])
        golden = GoldenSet.from_json(GOLDEN_SET)
        positives = [case for case in golden if case.answerable]
        negatives = [case for case in golden if not case.answerable]
        self.assertGreaterEqual(len(positives), 8)
        self.assertGreaterEqual(len(negatives), 3)
        for case in positives:
            with self.subTest(kind="positive", query=case.query):
                self.assertTrue(system.retrieve(case.query).answer_ready)
        for case in negatives:
            with self.subTest(kind="negative", query=case.query):
                result = system.retrieve(case.query)
                self.assertFalse(result.answer_ready)
                self.assertEqual(result.prompt, "")


if __name__ == "__main__":
    unittest.main()
