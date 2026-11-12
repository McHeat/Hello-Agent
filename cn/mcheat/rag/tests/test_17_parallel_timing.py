"""P1-4：并行召回一致性、单路超时降级、阶段耗时、预算与线程安全。"""
from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

from cn.mcheat.rag import RAGConfig, RAGSystem
from cn.mcheat.rag.core.interfaces import RecallRetriever
from cn.mcheat.rag.indexing.parent_store import InMemoryParentStore
from cn.mcheat.rag.retrieval.bm25 import BM25Index
from cn.mcheat.rag.retrieval.hybrid import HybridRetriever, RRFFusion

from cn.mcheat.rag.tests._helpers import make_child, make_hit, make_parent

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


class _StaticRetriever(RecallRetriever):
    def __init__(self, hits) -> None:
        self._hits = list(hits)

    def index(self, chunks):
        return 0

    def search(self, query, k, metadata_filter=None):
        return list(self._hits[:k])

    def delete(self, ids):
        return 0

    def ids(self):
        return {hit.chunk.id for hit in self._hits}

    def count(self):
        return len(self._hits)


class _SlowRetriever(_StaticRetriever):
    def search(self, query, k, metadata_filter=None):
        time.sleep(0.05)
        return super().search(query, k, metadata_filter)


def _paths():
    dense = _StaticRetriever([
        make_hit("d1", "甲文本", 0.9, "dense"),
        make_hit("d2", "乙文本", 0.8, "dense"),
    ])
    sparse = _StaticRetriever([
        make_hit("s1", "丙文本", 5.0, "sparse"),
        make_hit("d1", "甲文本", 4.0, "sparse"),
    ])
    return dense, sparse


class TestParallelRecall(unittest.TestCase):
    def test_parallel_matches_serial_and_keeps_deterministic_diagnostics(self):
        dense, sparse = _paths()
        serial = HybridRetriever(dense, sparse, RRFFusion(),
                                 parallel_recall=False)
        parallel = HybridRetriever(dense, sparse, RRFFusion(),
                                   parallel_recall=True)
        fused_a, diag_a = serial.recall("q", 5, 5, 10)
        fused_b, diag_b = parallel.recall("q", 5, 5, 10)
        self.assertEqual([item.chunk.id for item in fused_a],
                         [item.chunk.id for item in fused_b])
        self.assertEqual(diag_a["per_path"], diag_b["per_path"])
        self.assertEqual(list(diag_b["per_path"]), ["dense", "sparse"])

    def test_single_path_timeout_degrades_without_breaking_other_path(self):
        slow = _SlowRetriever([make_hit("slow", "慢通路", 0.9, "dense")])
        sparse = _StaticRetriever([make_hit("fast", "快通路", 3.0, "sparse")])
        hybrid = HybridRetriever(slow, sparse, RRFFusion(),
                                 parallel_recall=True, recall_timeout_s=0.01)
        fused, diag = hybrid.recall("q", 5, 5, 10)
        self.assertTrue(diag["degraded"])
        self.assertIn("error", diag["per_path"]["dense"])
        self.assertEqual([item.chunk.id for item in fused], ["fast"])


class TestObservability(unittest.TestCase):
    def test_stage_timings_and_budget_fields(self):
        system = RAGSystem(config=RAGConfig.load(use_env=False))
        system.index([DATA_DIR])
        result = system.retrieve("RRF 融合")
        diagnostics = result.diagnostics
        self.assertEqual(set(diagnostics["stage_timings_ms"]), {
            "transform", "recall", "fusion", "rerank", "expand",
            "context", "fallback",
        })
        self.assertGreaterEqual(diagnostics["elapsed_ms"], 0.0)
        self.assertGreaterEqual(diagnostics["recall_calls"], 1)
        self.assertLessEqual(diagnostics["recall_calls"],
                             system.config.retrieval.max_recall_calls)
        self.assertFalse(system.config.retrieval.parallel_recall)  # D10 默认关闭

    def test_max_recall_calls_caps_variants_times_retries(self):
        cfg = RAGConfig.load(
            overrides={"retrieval": {"max_recall_calls": 1}}, use_env=False
        )
        system = RAGSystem(config=cfg)
        system.index([DATA_DIR])
        result = system.retrieve("任意问题")
        self.assertLessEqual(result.diagnostics["recall_calls"], 1)


class TestThreadSafety(unittest.TestCase):
    def test_memory_components_survive_concurrent_read_write(self):
        errors: list[Exception] = []
        bm25 = BM25Index()
        parents = InMemoryParentStore()

        def worker(index: int) -> None:
            try:
                for j in range(20):
                    chunk = make_child(f"c{index}-{j}", f"语料片段 {j}")
                    bm25.index([chunk])
                    bm25.search("语料", k=3)
                    parents.put([
                        make_parent(f"p{index}-{j}", f"父块 {j}")
                    ])
                    parents.get_many([f"p{index}-{j}"])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(bm25.count(), 80)
        self.assertEqual(parents.count(), 80)


if __name__ == "__main__":
    unittest.main()
