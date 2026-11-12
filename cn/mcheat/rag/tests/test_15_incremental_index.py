"""P1-2：SourceManifest 变更检测、文档级增量索引、幂等与版本递增。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cn.mcheat.rag import RAGConfig, RAGSystem
from cn.mcheat.rag.indexing.manifest import SourceManifest, SourceRecord
from cn.mcheat.rag.retrieval.bm25 import BM25Index
from cn.mcheat.rag.vectorstores import InMemoryVectorStore

from cn.mcheat.rag.tests._helpers import make_child

TEXT_A = "向量检索与 BM25 混合召回，粗召回后再精排到少量片段。" * 4
TEXT_B = "父子切分把大段落切成小子块，命中后还原完整上下文。" * 4
TEXT_A_NEW = "帕米尔高原雪豹栖息地海拔调查记录与冬季觅食习性报告。" * 4


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class TestSourceManifest(unittest.TestCase):
    def _manifest(self) -> SourceManifest:
        return SourceManifest(
            index_version=1,
            embedding_fingerprint="fp",
            sources={
                "a": SourceRecord("h1", ["c1"], ["p1"]),
                "b": SourceRecord("h2", ["c2"], ["p2"]),
            },
        )

    def test_changed_sources_detects_new_changed_and_removed(self):
        manifest = self._manifest()
        self.assertEqual(
            manifest.changed_sources({"a": "h1", "b": "h2", "c": "h3"}), ["c"]
        )
        self.assertEqual(
            manifest.changed_sources({"a": "h1", "b": "CHANGED"}), ["b"]
        )
        self.assertEqual(manifest.changed_sources({"a": "h1"}), ["b"])
        self.assertEqual(manifest.changed_sources({"a": "h1", "b": "h2"}), [])

    def test_bump_and_serialization_roundtrip(self):
        manifest = self._manifest()
        bumped = manifest.bump()
        self.assertEqual(bumped.index_version, 2)
        self.assertEqual(manifest.index_version, 1)
        self.assertEqual(SourceManifest.from_dict(manifest.as_dict()).as_dict(),
                         manifest.as_dict())


class TestIncrementalIndex(unittest.TestCase):
    def test_incremental_rebuilds_only_changed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_a, file_b = root / "a.txt", root / "b.txt"
            _write(file_a, TEXT_A)
            _write(file_b, TEXT_B)
            system = RAGSystem(config=RAGConfig.load(use_env=False))
            dense = system.component("dense_retriever")

            first = system.index([root])
            self.assertEqual(first.mode, "full")
            self.assertEqual(first.index_version, 1)
            self.assertGreater(first.children, 0)
            ids_before = set(dense.ids())
            self.assertEqual({str(file_a.resolve()), str(file_b.resolve())},
                             set(system.indexer.manifest.sources))

            _write(file_a, TEXT_A_NEW)
            second = system.index([root])
            self.assertEqual(second.mode, "incremental")
            self.assertEqual(second.changed_sources, 1)
            self.assertGreater(second.deleted_chunks, 0)
            self.assertEqual(second.index_version, 2)
            self.assertEqual(second.children, dense.count())
            self.assertNotEqual(set(dense.ids()), ids_before)

            ids_after = set(dense.ids())
            third = system.index([root])
            self.assertEqual(third.mode, "incremental")
            self.assertEqual(third.changed_sources, 0)
            self.assertEqual(third.index_version, 3)
            self.assertEqual(third.children, first.children)
            self.assertEqual(set(dense.ids()), ids_after)

            full = system.index([root], mode="full")
            self.assertEqual(full.mode, "full")
            self.assertEqual(full.index_version, 4)
            self.assertEqual(set(dense.ids()), ids_after)
            self.assertTrue(system.indexer.manifest.embedding_fingerprint)

    def test_mode_full_is_available_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "a.txt", TEXT_A)
            system = RAGSystem(config=RAGConfig.load(use_env=False))
            system.index([root], mode="full")
            stats = system.index([root], mode="full")
            self.assertEqual(stats.mode, "full")


class TestDeleteBySourceHook(unittest.TestCase):
    def test_memory_vector_store_and_bm25_delete_by_source(self):
        chunks = [
            make_child("a", "甲", source="/tmp/x.txt", filename="x.txt"),
            make_child("b", "乙", source="/tmp/y.txt", filename="y.txt"),
        ]
        store = InMemoryVectorStore()
        store.upsert(chunks, [(1.0, 0.0), (0.0, 1.0)])
        self.assertEqual(store.delete_by_source("x.txt"), 1)
        self.assertEqual(store.ids(), {"b"})

        bm25 = BM25Index()
        bm25.index(chunks)
        self.assertEqual(bm25.delete_by_source("/tmp/y.txt"), 1)
        self.assertEqual(bm25.ids(), {"a"})


if __name__ == "__main__":
    unittest.main()
