"""Embedding 与可插拔向量库：确定性、维度校验、过滤、持久化、注册表扩展。"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from cn.mcheat.rag.core.registry import VECTOR_STORE_REGISTRY
from cn.mcheat.rag.embeddings import HashingEmbedder
from cn.mcheat.rag.exceptions import ConfigError, DimensionMismatchError
from cn.mcheat.rag.vectorstores import FileBackedVectorStore, InMemoryVectorStore

from cn.mcheat.rag.tests._helpers import make_child


class TestHashingEmbedder(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        embedder = HashingEmbedder(dimension=128)
        first = embedder.embed_query("向量检索与 BM25 混合召回")
        second = embedder.embed_query("向量检索与 BM25 混合召回")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 128)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in first)), 1.0, places=6)

    def test_batch_preserves_order(self):
        embedder = HashingEmbedder(dimension=64)
        texts = ["文本甲", "文本乙", "文本丙"]
        vectors = embedder.embed_documents(texts)
        self.assertEqual(vectors, [embedder.embed_query(t) for t in texts])

    def test_similar_text_scores_higher(self):
        embedder = HashingEmbedder(dimension=256)
        from cn.mcheat.rag._utils import cosine_similarity
        query = embedder.embed_query("父子切分 Small-to-Big")
        same = embedder.embed_query("父子切分 Small-to-Big 的实现")
        other = embedder.embed_query("完全无关的量子色动力学胶子")
        self.assertGreater(cosine_similarity(query, same),
                           cosine_similarity(query, other))

    def test_invalid_dimension(self):
        with self.assertRaises(ConfigError):
            HashingEmbedder(dimension=0)


class TestInMemoryVectorStore(unittest.TestCase):
    def _populate(self, store: InMemoryVectorStore):
        store.upsert(
            [
                make_child("a", "甲", tenant="t1"),
                make_child("b", "乙", tenant="t2"),
                make_child("c", "丙", tenant="t1"),
            ],
            [(1.0, 0.0), (0.0, 1.0), (0.9, 0.1)],
        )

    def test_search_ranks_by_cosine(self):
        store = InMemoryVectorStore()
        self._populate(store)
        hits = store.search((1.0, 0.0), k=3)
        self.assertEqual([h.chunk.id for h in hits][0], "a")
        self.assertEqual([h.source for h in hits], ["dense"] * 3)
        self.assertEqual(store.count(), 3)

    def test_metadata_filter(self):
        store = InMemoryVectorStore()
        self._populate(store)
        hits = store.search((1.0, 0.0), k=5, metadata_filter={"tenant": "t2"})
        self.assertEqual([h.chunk.id for h in hits], ["b"])

    def test_dimension_mismatch_is_explicit(self):
        store = InMemoryVectorStore()
        store.upsert([make_child("a", "甲")], [(1.0, 0.0)])
        with self.assertRaises(DimensionMismatchError) as ctx:
            store.upsert([make_child("b", "乙")], [(1.0, 0.0, 0.0)])
        self.assertIn("重建索引", str(ctx.exception))
        with self.assertRaises(DimensionMismatchError):
            store.search((1.0, 0.0, 0.0), k=1)

    def test_upsert_is_idempotent_and_delete_works(self):
        store = InMemoryVectorStore()
        self._populate(store)
        store.upsert([make_child("a", "甲（更新）")], [(1.0, 0.0)])
        self.assertEqual(store.count(), 3)
        self.assertEqual(store.delete(["b"]), 1)
        self.assertEqual(store.ids(), {"a", "c"})
        store.clear()
        self.assertEqual(store.count(), 0)

    def test_export_import_state_roundtrip(self):
        store = InMemoryVectorStore()
        self._populate(store)
        restored = InMemoryVectorStore()
        restored.import_state(store.export_state())
        self.assertEqual(
            [h.chunk.id for h in restored.search((1.0, 0.0), k=3)],
            [h.chunk.id for h in store.search((1.0, 0.0), k=3)],
        )

    def test_unknown_metric_rejected(self):
        with self.assertRaises(ConfigError):
            InMemoryVectorStore(metric="euclidean")


class TestFileBackedVectorStore(unittest.TestCase):
    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.json"
            first = FileBackedVectorStore(path=path)
            first.upsert([make_child("a", "甲")], [(1.0, 0.0)])
            second = FileBackedVectorStore(path=path)
            self.assertEqual(second.count(), 1)
            self.assertEqual(second.search((1.0, 0.0), k=1)[0].chunk.id, "a")


class TestVectorStoreRegistry(unittest.TestCase):
    def test_custom_store_can_be_registered_and_built(self):
        class CustomStore(InMemoryVectorStore):
            pass

        name = "test_custom_store"
        try:
            VECTOR_STORE_REGISTRY.register(name, CustomStore)
            self.assertIn(name, VECTOR_STORE_REGISTRY.keys())
            self.assertIsInstance(
                VECTOR_STORE_REGISTRY.create(name), CustomStore
            )
        finally:
            VECTOR_STORE_REGISTRY.unregister(name)

    def test_registry_rejects_wrong_contract(self):
        name = "test_bad_store"
        try:
            VECTOR_STORE_REGISTRY.register(name, dict)
            with self.assertRaises(ConfigError) as ctx:
                VECTOR_STORE_REGISTRY.create(name)
            self.assertIn("VectorStore", str(ctx.exception))
        finally:
            VECTOR_STORE_REGISTRY.unregister(name)


if __name__ == "__main__":
    unittest.main()
