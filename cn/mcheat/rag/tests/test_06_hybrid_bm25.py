"""BM25 稀疏检索、RRF/加权融合、混合召回的单路降级。"""
from __future__ import annotations

import unittest

from cn.mcheat.rag.core.interfaces import RecallRetriever
from cn.mcheat.rag.exceptions import EmptyRecallError
from cn.mcheat.rag.models import Ranking
from cn.mcheat.rag.retrieval.bm25 import BM25Index
from cn.mcheat.rag.retrieval.hybrid import (
    HybridRetriever,
    RRFFusion,
    WeightedSumFusion,
)

from cn.mcheat.rag.tests._helpers import make_child, make_hit


class BrokenRetriever(RecallRetriever):
    """测试替身：召回时永远抛异常，用于验证单路降级。"""

    def index(self, chunks):
        return 0

    def search(self, query, k, metadata_filter=None):
        raise RuntimeError("service down")

    def delete(self, ids):
        return 0

    def ids(self):
        return set()

    def count(self):
        return 0


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index(k1=1.5, b=0.75)
        self.chunks = [
            make_child("a", "向量检索 混合召回 RRF 融合", tenant="t1"),
            make_child("b", "父子切分 Small-to-Big 大段落", tenant="t1"),
            make_child("c", "Lost in the Middle 上下文编排", tenant="t2"),
        ]
        self.index.index(self.chunks)

    def test_exact_term_ranks_correct_document_first(self):
        hits = self.index.search("Small-to-Big 父子切分", k=3)
        self.assertEqual(hits[0].chunk.id, "b")
        self.assertEqual(hits[0].source, "sparse")
        self.assertGreater(hits[0].score, 0.0)

    def test_metadata_filter(self):
        hits = self.index.search("上下文编排 Lost", k=5,
                                metadata_filter={"tenant": "t1"})
        # t2 文档被过滤掉，只剩 t1 的候选
        self.assertTrue(all(h.chunk.metadata["tenant"] == "t1" for h in hits))
        self.assertNotIn("c", [h.chunk.id for h in hits])

    def test_upsert_updates_statistics(self):
        self.index.index([make_child("b", "完全替换后的新文本")])
        self.assertEqual(self.index.count(), 3)

    def test_export_import_keeps_results(self):
        before = [h.chunk.id for h in self.index.search("RRF 融合", k=3)]
        restored = BM25Index()
        restored.import_state(self.index.export_state())
        after = [h.chunk.id for h in restored.search("RRF 融合", k=3)]
        self.assertEqual(before, after)

    def test_delete_and_empty_query(self):
        self.assertEqual(self.index.delete(["a"]), 1)
        self.assertEqual(self.index.count(), 2)
        self.assertEqual(self.index.search("", k=3), [])


class TestFusion(unittest.TestCase):
    def test_rrf_dedupes_and_tracks_sources(self):
        rankings = [
            Ranking(source="dense", hits=[
                make_hit("x", "甲文本", 0.9, "dense"),
                make_hit("y", "乙文本", 0.8, "dense"),
            ], weight=1.0),
            Ranking(source="sparse", hits=[
                make_hit("y", "乙文本", 12.0, "sparse"),
                make_hit("z", "丙文本", 9.0, "sparse"),
            ], weight=1.0),
        ]
        fused = RRFFusion(k=60).fuse(rankings, limit=10)
        self.assertEqual(len(fused), 3)
        self.assertEqual([item.chunk.id for item in fused], ["y", "x", "z"])
        by_id = {item.chunk.id: item for item in fused}
        self.assertEqual(set(by_id["y"].sources), {"dense", "sparse"})
        self.assertEqual(by_id["y"].ranks, {"dense": 2, "sparse": 1})

    def test_rrf_limit_and_weights(self):
        rankings = [Ranking(source="dense", hits=[
            make_hit(f"d{i}", f"文本{i}", 1.0 - i * 0.1) for i in range(5)
        ])]
        self.assertEqual(len(RRFFusion(k=60).fuse(rankings, limit=2)), 2)
        heavy = RRFFusion(k=60, dense_weight=5.0).fuse(rankings, limit=5)
        self.assertAlmostEqual(heavy[0].score, 5.0 / 61)

    def test_weighted_sum_fusion(self):
        rankings = [
            Ranking(source="dense", hits=[
                make_hit("x", "甲", 1.0), make_hit("y", "乙", 0.0),
            ]),
            Ranking(source="sparse", hits=[
                make_hit("y", "乙", 10.0), make_hit("x", "甲", 0.0),
            ]),
        ]
        fused = WeightedSumFusion().fuse(rankings, limit=5)
        # x 与 y 各拿到一次 1.0 归一化分，结果同分按 ID 稳定排序
        self.assertEqual([item.chunk.id for item in fused], ["x", "y"])
        self.assertAlmostEqual(fused[0].score, fused[1].score)


class TestHybridRetriever(unittest.TestCase):
    def test_single_path_failure_degrades_gracefully(self):
        sparse = BM25Index()
        sparse.index([make_child("a", "混合召回 RRF 融合")])
        hybrid = HybridRetriever(BrokenRetriever(), sparse, RRFFusion())
        fused, diag = hybrid.recall("RRF", dense_k=5, sparse_k=5, limit=10)
        self.assertTrue(diag["degraded"])
        self.assertIn("service down", diag["per_path"]["dense"]["error"])
        self.assertEqual([item.chunk.id for item in fused], ["a"])

    def test_all_paths_failure_raises_internal_signal(self):
        hybrid = HybridRetriever(BrokenRetriever(), BrokenRetriever(), RRFFusion())
        with self.assertRaises(EmptyRecallError):
            hybrid.recall("query", dense_k=5, sparse_k=5, limit=10)

    def test_disabled_path_is_not_called(self):
        sparse = BM25Index()
        sparse.index([make_child("a", "混合召回 RRF")])
        hybrid = HybridRetriever(BrokenRetriever(), sparse, RRFFusion(),
                                 enable_dense=False)
        fused, diag = hybrid.recall("RRF", dense_k=5, sparse_k=5, limit=10)
        self.assertNotIn("dense", diag["per_path"])
        self.assertEqual(len(fused), 1)

    def test_empty_query_rejected(self):
        hybrid = HybridRetriever(BrokenRetriever(), BM25Index(), RRFFusion())
        from cn.mcheat.rag.exceptions import RetrievalError
        with self.assertRaises(RetrievalError):
            hybrid.recall("   ", dense_k=5, sparse_k=5, limit=10)


if __name__ == "__main__":
    unittest.main()
