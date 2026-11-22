"""P1-3：拒答指标、负样本 schema、零依赖裁判与答案级评测门禁。"""
from __future__ import annotations

import unittest
from pathlib import Path

from cn.mcheat.rag import RAGConfig
from cn.mcheat.rag.evaluation import (
    EvalReport,
    GoldenCase,
    GoldenSet,
)
from cn.mcheat.rag.evaluation.answer_metrics import (
    AnswerEvalRunner,
    LexicalJudge,
    answer_relevancy,
    faithfulness,
)
from cn.mcheat.rag.evaluation.metrics import (
    GATED_METRICS,
    false_accept_rate,
    false_refusal_rate,
    refusal_accuracy,
)
from cn.mcheat.rag.evaluation.runner import EvalRunner
from cn.mcheat.rag.exceptions import EvaluationError
from cn.mcheat.rag.models import (
    BuiltContext,
    FallbackDecision,
    RetrievalAction,
    RetrievalResult,
)

from cn.mcheat.rag.tests._helpers import make_fake_judge, make_negative_golden_set

GOLDEN_SET = Path(__file__).resolve().parents[1] / "examples" / "golden_set.json"


class _FakeSystem:
    """按 query 前缀决定「回答 / 拒答」的评测替身。"""

    def __init__(self) -> None:
        self.config = RAGConfig.load(use_env=False)

    def retrieve(self, query, metadata_filter=None) -> RetrievalResult:
        refused = query.startswith("拒绝")
        return RetrievalResult(
            query=query,
            context=BuiltContext(text="向量检索与 BM25 混合召回的资料"),
            fallback=FallbackDecision(
                triggered=refused,
                final_action=(RetrievalAction.NO_ANSWER if refused
                              else RetrievalAction.ANSWER),
            ),
        )


class TestRefusalMetrics(unittest.TestCase):
    def test_metric_definitions(self):
        # (should_refuse, refused)：1 正样本误杀、1 负样本误接受、2 正确
        outcomes = [(False, True), (True, False), (False, False), (True, True)]
        self.assertAlmostEqual(false_refusal_rate(outcomes), 0.5)
        self.assertAlmostEqual(false_accept_rate(outcomes), 0.5)
        self.assertAlmostEqual(refusal_accuracy(outcomes), 0.5)
        self.assertEqual(false_refusal_rate([]), 0.0)
        self.assertEqual(false_accept_rate([]), 0.0)
        self.assertEqual(refusal_accuracy([]), 0.0)

    def test_refusal_metrics_are_gated(self):
        for metric in ("false_refusal_rate", "false_accept_rate",
                       "refusal_accuracy"):
            self.assertIn(metric, GATED_METRICS)

    def test_gate_skips_metrics_missing_from_old_baseline(self):
        old_baseline = EvalReport(summary={
            "context_recall": 1.0, "context_precision": 1.0,
            "average_precision": 1.0, "hit_rate": 1.0,
        })
        current = EvalReport(summary={
            **old_baseline.summary,
            "false_accept_rate": 0.8,
        })
        self.assertTrue(current.regression_check(old_baseline).passed)

    def test_gate_detects_refusal_regression_in_both_directions(self):
        baseline = EvalReport(summary={
            "false_accept_rate": 0.0, "false_refusal_rate": 0.0,
            "refusal_accuracy": 1.0,
        })
        regressed = EvalReport(summary={
            "false_accept_rate": 0.5, "false_refusal_rate": 0.5,
            "refusal_accuracy": 0.5,
        })
        verdict = regressed.regression_check(baseline, max_drop=0.02)
        self.assertFalse(verdict.passed)
        for metric in ("false_accept_rate", "false_refusal_rate",
                       "refusal_accuracy"):
            self.assertTrue(any(metric in item for item in verdict.failed_metrics))


class TestNegativeGoldenSchema(unittest.TestCase):
    def test_negative_cases_allow_empty_relevant_ids(self):
        golden = make_negative_golden_set(3)
        self.assertEqual(len(golden), 3)
        for case in golden:
            self.assertTrue(case.should_refuse)
            self.assertEqual(case.relevant_ids, frozenset())
        self.assertEqual(GoldenSet.from_dict(golden.as_dict()).as_dict(),
                         golden.as_dict())

    def test_answerable_cases_still_require_relevant_ids(self):
        with self.assertRaises(EvaluationError):
            GoldenCase.from_dict({"query": "q", "relevant_ids": []})

    def test_example_golden_set_contains_negative_cases(self):
        golden = GoldenSet.from_json(GOLDEN_SET)
        self.assertGreaterEqual(len([c for c in golden if not c.answerable]), 3)
        self.assertGreaterEqual(len([c for c in golden if c.answerable]), 8)


class TestAnswerLevelEvaluation(unittest.TestCase):
    def test_lexical_judge_is_zero_dependency_and_bounded(self):
        judge = LexicalJudge()
        score = judge.judge_faithfulness(
            "向量检索是什么", "向量检索用于混合召回", "向量检索用于混合召回"
        )
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(
            judge.judge_relevancy("向量检索", "向量检索的原理"),
            judge.judge_relevancy("向量检索", "烤箱温度"),
        )
        self.assertEqual(judge.judge_faithfulness("q", "", "ctx"), 0.0)

    def test_faithfulness_and_answer_relevancy_use_injected_judge(self):
        fake = make_fake_judge(faithfulness=0.8, relevancy=0.6)
        self.assertAlmostEqual(faithfulness("q", "a", "c", fake), 0.8)
        self.assertAlmostEqual(answer_relevancy("q", "a", fake), 0.6)

    def test_answer_eval_runner_aggregates_faithfulness_and_relevancy(self):
        golden = GoldenSet.from_cases([
            GoldenCase("向量检索是什么", frozenset({"a.md"})),
        ])
        answers = {"向量检索是什么": "向量检索与 BM25 混合召回"}
        report = AnswerEvalRunner(
            _FakeSystem(), golden, answers, judge=LexicalJudge()
        ).run()
        self.assertIn("faithfulness", report.summary)
        self.assertIn("answer_relevancy", report.summary)
        self.assertGreater(report.summary["faithfulness"], 0.0)
        self.assertEqual(len(report.cases), 1)

    def test_negative_cases_do_not_join_recall_precision_means(self):
        golden = GoldenSet.from_cases([
            GoldenCase("向量检索是什么", frozenset({"a.md"})),
            GoldenCase("拒绝：无关问题", frozenset(), answerable=False),
        ])
        report = EvalRunner(_FakeSystem(), golden).run()
        summary = report.summary
        for metric in ("context_recall", "false_refusal_rate",
                       "false_accept_rate", "refusal_accuracy"):
            self.assertIn(metric, summary)
        # 正样本未被误杀、负样本被拒答，故误杀 / 误接受率均为 0
        self.assertAlmostEqual(summary["false_refusal_rate"], 0.0)
        self.assertAlmostEqual(summary["false_accept_rate"], 0.0)
        self.assertAlmostEqual(summary["refusal_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
