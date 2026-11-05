"""端到端：离线索引 → 在线检索 → 5 条注意事项 → 持久化 → 评测 → 可插拔替换。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cn.mcheat.rag import RAGConfig, RAGSystem
from cn.mcheat.rag.exceptions import ConfigError
from cn.mcheat.rag.models import RetrievalAction
from cn.mcheat.rag.vectorstores import InMemoryVectorStore

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
GOLDEN_SET = Path(__file__).resolve().parents[1] / "examples" / "golden_set.json"


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = RAGConfig.load(use_env=False)
        cls.system = RAGSystem(config=cls.config)
        cls.stats = cls.system.index([DATA_DIR])

    def test_offline_index_covers_all_five_formats(self):
        self.assertGreater(self.stats.documents_loaded, 5)
        self.assertGreater(self.stats.parents, 1)
        self.assertGreater(self.stats.children, self.stats.parents)
        self.assertEqual(self.stats.vectors_upserted, self.stats.children)
        report = self.system.indexer.last_cleaning_report
        self.assertEqual(report.total_in, self.stats.documents_loaded)
        # changelog.txt 里的页码 / 版权行应被清洗掉
        for child in self.system.component("dense_retriever").vector_store._chunks.values():
            if child.metadata.get("filename") == "changelog.txt":
                self.assertNotIn("第 1 页", child.text)

    def test_note_1_coarse_recall_is_pruned_to_top_n_before_prompt(self):
        result = self.system.retrieve("Top-K 为什么不是越大越好")
        self.assertLessEqual(len(result.coarse), self.config.retrieval.coarse_top_k)
        self.assertLessEqual(len(result.final), self.config.retrieval.rerank_top_n)
        self.assertGreater(len(result.final), 0)
        # prompt 里不应出现粗召回的所有片段编号
        self.assertLessEqual(result.prompt.count("来源："),
                             self.config.retrieval.rerank_top_n)
        self.assertEqual(
            result.diagnostics["final_count"], len(result.final)
        )

    def test_note_2_small_to_big_returns_full_parent(self):
        result = self.system.retrieve("父子切分 Small-to-Big 解决什么问题")
        self.assertGreater(result.diagnostics["parent_expanded"], 0)
        for item in result.final:
            if item.parent is not None:
                self.assertGreaterEqual(len(item.parent.text), len(item.chunk.text))
                self.assertEqual(
                    item.parent.text[item.chunk.start:item.chunk.end],
                    item.chunk.text,
                )
        # 同一父块不会被重复塞进上下文
        parent_ids = [item.chunk.parent_id for item in result.final]
        self.assertEqual(len(parent_ids), len(set(parent_ids)))

    def test_note_3_key_fragments_are_not_buried_in_the_middle(self):
        result = self.system.retrieve("Lost in the Middle 怎么摆放检索片段")
        placements = result.context.placements
        self.assertGreaterEqual(len(placements), 2)
        self.assertEqual(placements[0].relevance_rank,
                         min(p.relevance_rank for p in placements))
        # 问题在 prompt 末尾重复，强化近因效应
        self.assertIn(result.query, result.prompt.splitlines()[-1])

    def test_note_4_golden_set_report_separates_metrics(self):
        report = self.system.evaluate(GOLDEN_SET)
        self.assertIn("context_recall", report.summary)
        self.assertIn("context_precision", report.summary)
        self.assertEqual(report.config_snapshot["recall_stage"], "coarse")
        self.assertEqual(report.config_snapshot["precision_stage"], "final")
        self.assertGreaterEqual(len(report.cases), 8)
        self.assertEqual(report.golden_set, "rag-notes-v1")
        for case in report.cases:
            self.assertGreaterEqual(len(case.coarse_ids), len(case.final_ids))

    def test_note_5_fallback_observability(self):
        empty = RAGSystem(config=RAGConfig.load(use_env=False))
        result = empty.retrieve("任意问题")
        self.assertIn(result.fallback.final_action,
                      {RetrievalAction.HANDOFF, RetrievalAction.NO_ANSWER})
        self.assertTrue(result.fallback.triggers)
        self.assertTrue(result.fallback.steps)
        self.assertEqual(result.prompt, "")
        self.assertFalse(result.answer_ready)

    def test_save_and_load_in_another_process_like_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.system.save(tmp)
            self.assertTrue(Path(path).is_file())
            restored = RAGSystem.load(tmp)
            self.assertEqual(
                restored.component("dense_retriever").count(),
                self.system.component("dense_retriever").count(),
            )
            self.assertEqual(
                restored.component("parent_store").count(),
                self.system.component("parent_store").count(),
            )
            a = self.system.retrieve("兜底链路有哪些降级手段")
            b = restored.retrieve("兜底链路有哪些降级手段")
            self.assertEqual([x.chunk.id for x in a.final],
                             [x.chunk.id for x in b.final])

    def test_index_is_idempotent(self):
        system = RAGSystem(config=RAGConfig.load(use_env=False))
        first = system.index([DATA_DIR])
        count_before = system.component("dense_retriever").count()
        second = system.index([DATA_DIR])
        self.assertEqual(first.children, second.children)
        self.assertEqual(count_before,
                         system.component("dense_retriever").count())


class TestPluggability(unittest.TestCase):
    def test_vector_store_can_be_injected_via_overrides(self):
        custom = InMemoryVectorStore()
        system = RAGSystem(config=RAGConfig.load(use_env=False),
                           overrides={"vector_store": custom})
        system.index([DATA_DIR])
        self.assertGreater(custom.count(), 0)
        self.assertGreater(len(system.retrieve("RRF 融合").final), 0)

    def test_config_can_switch_component_implementation(self):
        cfg = RAGConfig.load(
            overrides={"components": {
                "parent_store": {"name": "sqlite", "params": {"path": ":memory:"}},
                "fusion": {"name": "weighted", "params": {}},
                "reranker": {"name": "noop", "params": {}},
                "context_builder": {"name": "sequential", "params": {}},
            }},
            use_env=False,
        )
        system = RAGSystem(config=cfg)
        system.index([DATA_DIR])
        result = system.retrieve("Golden Set 回归评测")
        self.assertEqual(result.diagnostics["recall"]["fusion"],
                         "WeightedSumFusion")
        self.assertGreater(len(result.final), 0)
        self.assertEqual(
            [p.relevance_rank for p in result.context.placements],
            sorted(p.relevance_rank for p in result.context.placements),
        )

    def test_bad_overrides_are_rejected(self):
        with self.assertRaises(ConfigError):
            RAGSystem(config=RAGConfig.load(use_env=False),
                      overrides={"unknown_component": object()})
        with self.assertRaises(ConfigError):
            RAGSystem(config=RAGConfig.load(use_env=False),
                      overrides={"vector_store": object()})
        cfg = RAGConfig.load(
            overrides={"components": {"embedder": {"name": "not_registered"}}},
            use_env=False,
        )
        with self.assertRaises(ConfigError):
            RAGSystem(config=cfg)


if __name__ == "__main__":
    unittest.main()
