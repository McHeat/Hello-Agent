"""Golden Set、Context Recall / Precision 指标、分阶段评测与回归门禁。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cn.mcheat.rag import RAGConfig
from cn.mcheat.rag.config import EvaluationConfig
from cn.mcheat.rag.evaluation import (
    EvalReport,
    EvalRunner,
    GoldenCase,
    GoldenSet,
    average_precision,
    context_precision,
    context_recall,
    f1_score,
    hit_rate,
    precision_at_k,
    reciprocal_rank,
)
from cn.mcheat.rag.exceptions import EvaluationError
from cn.mcheat.rag.models import (
    ChildChunk,
    FallbackDecision,
    RetrievalResult,
    ScoredChunk,
)

EXAMPLE_GOLDEN = (
    Path(__file__).resolve().parents[1] / "examples" / "golden_set.json"
)


def _item(chunk_id: str, filename: str, rank: int) -> ScoredChunk:
    return ScoredChunk(
        chunk=ChildChunk(
            id=chunk_id, parent_id="p-" + chunk_id, doc_id="d0",
            text=f"{filename} 的内容", start=0, end=8,
            metadata={"source": f"/tmp/{filename}", "filename": filename},
        ),
        score=1.0 - rank * 0.1, rank=rank,
    )


class TestMetrics(unittest.TestCase):
    def test_recall_and_precision_are_separate(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = ["a", "c", "e"]
        self.assertAlmostEqual(context_recall(retrieved, relevant), 2 / 3)
        self.assertAlmostEqual(context_precision(retrieved, relevant), 0.5)

    def test_empty_inputs_are_defined(self):
        self.assertEqual(context_recall([], ["a"]), 0.0)
        self.assertEqual(context_precision([], ["a"]), 0.0)
        self.assertEqual(context_recall(["a"], []), 0.0)

    def test_rank_aware_metrics(self):
        ranked = ["x", "a", "b", "c"]
        relevant = ["a", "c"]
        self.assertEqual(hit_rate(ranked, relevant), 1.0)
        self.assertEqual(reciprocal_rank(ranked, relevant), 0.5)
        self.assertAlmostEqual(precision_at_k(ranked, relevant, 2), 0.5)
        self.assertGreater(average_precision(["a", "x"], relevant),
                           average_precision(["x", "a"], relevant))
        self.assertAlmostEqual(f1_score(0.5, 0.5), 0.5)
        self.assertEqual(f1_score(0.0, 0.0), 0.0)

    def test_duplicates_do_not_inflate_precision(self):
        self.assertAlmostEqual(
            context_precision(["a", "a", "a"], ["a"]), 1.0
        )
        self.assertAlmostEqual(
            context_precision(["a", "x", "x", "x"], ["a"]), 0.5
        )


class TestGoldenSet(unittest.TestCase):
    def test_load_example_golden_set(self):
        golden = GoldenSet.from_json(EXAMPLE_GOLDEN)
        self.assertEqual(golden.name, "rag-notes-v1")
        self.assertGreaterEqual(len(golden), 8)
        self.assertTrue(all(case.relevant_ids for case in golden))

    def test_invalid_cases_are_rejected(self):
        for payload in (
            {"cases": [{"query": "q", "relevant_ids": []}]},
            {"cases": [{"relevant_ids": ["a"]}]},
            {"cases": []},
            {"cases": [{"query": "q", "relevant_ids": ["a"]},
                       {"query": "q", "relevant_ids": ["b"]}]},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(EvaluationError):
                    GoldenSet.from_dict(payload)

    def test_save_and_reload(self):
        golden = GoldenSet.from_cases([
            GoldenCase("问题一", frozenset({"a.md"})),
            GoldenCase("问题二", frozenset({"b.md", "c.md"})),
        ], name="test")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.json"
            golden.save(path)
            restored = GoldenSet.from_json(path)
            self.assertEqual(restored.as_dict(), golden.as_dict())


class _FakeSystem:
    """评测用的检索系统替身：固定返回「粗召回命中、精排全错」的结果。"""

    def __init__(self):
        self.config = RAGConfig.load(use_env=False)

    def retrieve(self, query):
        return RetrievalResult(
            query=query,
            coarse=[_item("c1", "a.md", 0), _item("c2", "b.md", 1),
                    _item("c3", "c.md", 2)],
            final=[_item("c2", "b.md", 0), _item("c3", "c.md", 1)],
            fallback=FallbackDecision(),
        )


class TestEvalRunner(unittest.TestCase):
    def test_recall_and_precision_are_measured_on_different_stages(self):
        golden = GoldenSet.from_cases([
            GoldenCase("测试问题", frozenset({"a.md"})),
        ])
        report = EvalRunner(_FakeSystem(), golden, EvaluationConfig()).run()
        self.assertEqual(len(report.cases), 1)
        # 粗召回里有 a.md → Recall=1；精排后进 prompt 的只有 b/c → Precision=0
        self.assertAlmostEqual(report.summary["context_recall"], 1.0)
        self.assertAlmostEqual(report.summary["context_precision"], 0.0)
        self.assertIn("average_precision", report.summary)
        self.assertEqual(report.config_snapshot["recall_stage"], "coarse")
        self.assertEqual(report.config_snapshot["precision_stage"], "final")

    def test_report_roundtrip_and_regression_gate(self):
        golden = GoldenSet.from_cases([
            GoldenCase("测试问题", frozenset({"a.md"})),
        ])
        report = EvalRunner(_FakeSystem(), golden).run()
        baseline = EvalReport(summary={
            "context_recall": 1.0, "context_precision": 0.9,
            "average_precision": 0.9, "hit_rate": 1.0,
        })
        verdict = report.regression_check(baseline, max_drop=0.02)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("context_precision" in m for m in verdict.failed_metrics))

        passing = EvalReport(summary=dict(report.summary))
        self.assertTrue(passing.regression_check(report).passed)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            report.save(path)
            loaded = EvalReport.load(path)
            self.assertEqual(loaded.summary, report.summary)


if __name__ == "__main__":
    unittest.main()
