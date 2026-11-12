"""P0-3：写放大修复（单次 flush）、auto_save 语义与 SQLite 向量库。"""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from cn.mcheat.rag import RAGConfig, RAGSystem
from cn.mcheat.rag.vectorstores import (
    FileBackedVectorStore,
    InMemoryVectorStore,
    SqliteVectorStore,
)

from cn.mcheat.rag.tests._helpers import make_child

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


class _CountingStore(FileBackedVectorStore):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1
        super().flush()


class TestWriteAmplification(unittest.TestCase):
    def test_single_build_triggers_exactly_one_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _CountingStore(path=Path(tmp) / "vectors.json")
            system = RAGSystem(config=RAGConfig.load(use_env=False),
                               overrides={"vector_store": store})
            system.index([DATA_DIR])
            self.assertEqual(store.flush_calls, 1)
            self.assertTrue(store.path.is_file())

    def test_auto_save_is_off_by_default_and_flush_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.json"
            store = FileBackedVectorStore(path=path)
            self.assertFalse(store.auto_save)
            store.upsert([make_child("a", "甲")], [(1.0, 0.0)])
            self.assertFalse(path.is_file())
            store.flush()
            self.assertTrue(path.is_file())
            self.assertEqual(store.get_many(["a"])["a"].id, "a")


class TestSqliteVectorStore(unittest.TestCase):
    def test_roundtrip_keeps_cosine_ranking_and_external_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.sqlite3"
            store = SqliteVectorStore(path=path)
            store.upsert(
                [make_child("a", "甲"), make_child("b", "乙")],
                [(1.0, 0.0), (0.0, 1.0)],
            )
            self.assertEqual(store.count(), 2)
            before = [hit.chunk.id for hit in store.search((1.0, 0.0), k=2)]
            self.assertEqual(before[0], "a")
            self.assertTrue(store.export_state()["external"])
            self.assertEqual(set(store.get_many(["a", "b"])), {"a", "b"})
            store.close()

            reopened = SqliteVectorStore(path=path)
            self.assertEqual(
                [hit.chunk.id for hit in reopened.search((1.0, 0.0), k=2)],
                before,
            )
            self.assertEqual(reopened.delete(["b"]), 1)
            self.assertEqual(reopened.count(), 1)
            reopened.close()

    def test_metadata_filter_and_delete_by_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SqliteVectorStore(path=Path(tmp) / "v.sqlite3")
            store.upsert(
                [
                    make_child("a", "甲", source="/tmp/x.txt", filename="x.txt"),
                    make_child("b", "乙", source="/tmp/y.txt", filename="y.txt"),
                ],
                [(1.0, 0.0), (0.0, 1.0)],
            )
            hits = store.search((1.0, 0.0), k=2,
                                metadata_filter={"filename": "y.txt"})
            self.assertEqual([hit.chunk.id for hit in hits], ["b"])
            self.assertEqual(store.delete_by_source("x.txt"), 1)
            self.assertEqual(store.ids(), {"b"})
            store.close()


class TestConcurrentSafety(unittest.TestCase):
    def test_memory_and_sqlite_stores_survive_concurrent_read_write(self):
        errors: list[Exception] = []
        memory = InMemoryVectorStore()

        def worker(index: int) -> None:
            try:
                for j in range(20):
                    memory.upsert(
                        [make_child(f"c{index}-{j}", f"文本{j}")], [(1.0, 0.0)]
                    )
                    memory.search((1.0, 0.0), k=3)
            except Exception as exc:  # 收集而不是吞掉，便于断言
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(memory.count(), 80)

        with tempfile.TemporaryDirectory() as tmp:
            sqlite_store = SqliteVectorStore(path=Path(tmp) / "v.sqlite3")

            def sqlite_worker(index: int) -> None:
                try:
                    for j in range(10):
                        sqlite_store.upsert(
                            [make_child(f"s{index}-{j}", f"语料{j}")],
                            [(1.0, 0.0)],
                        )
                        sqlite_store.search((1.0, 0.0), k=2)
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=sqlite_worker, args=(i,)) for i in range(3)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(sqlite_store.count(), 30)
            sqlite_store.close()


if __name__ == "__main__":
    unittest.main()
