"""P1-1：查询变换层、LLM Protocol 注入、两级缓存与调用预算。"""
from __future__ import annotations

import time
import unittest
from pathlib import Path

from cn.mcheat.rag import RAGConfig, RAGSystem
from cn.mcheat.rag.core.interfaces import QueryTransformer
from cn.mcheat.rag.embeddings.hashing import HashingEmbedder
from cn.mcheat.rag.evaluation import GoldenSet
from cn.mcheat.rag.exceptions import QueryTransformError
from cn.mcheat.rag.retrieval.cache import (
    EmbeddingQueryCache,
    InMemoryCache,
    NullCache,
    RetrievalResultCache,
)
from cn.mcheat.rag.retrieval.query_transform import (
    IdentityQueryTransformer,
    LLMQueryTransformer,
    QueryTransformPipeline,
    RuleBasedQueryTransformer,
)
from cn.mcheat.rag.retrieval.recall import EmbeddingDenseRetriever
from cn.mcheat.rag.vectorstores import InMemoryVectorStore

from cn.mcheat.rag.tests._helpers import make_child

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
GOLDEN_SET = Path(__file__).resolve().parents[1] / "examples" / "golden_set.json"


class _SuffixTransformer(QueryTransformer):
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    def transform(self, query: str) -> list[str]:
        return [query, f"{query}{self.suffix}"]


class _BrokenTransformer(QueryTransformer):
    def transform(self, query: str) -> list[str]:
        raise RuntimeError("transform service down")


class _Rewriter:
    def rewrite(self, query: str) -> str:
        return f"{query} 改写"


class TestQueryTransform(unittest.TestCase):
    def test_identity_and_rule_based_keep_original_as_main_query(self):
        self.assertEqual(IdentityQueryTransformer().transform("问题"), ["问题"])
        variants = RuleBasedQueryTransformer().transform("什么是 RRF 融合？")
        self.assertEqual(variants[0], "什么是 RRF 融合？")
        self.assertTrue(all(variants))
        self.assertGreaterEqual(len(variants), 1)
        self.assertLessEqual(len(variants), 3)

    def test_pipeline_applies_transformers_in_order_and_caps_variants(self):
        pipeline = QueryTransformPipeline(
            [_SuffixTransformer("A"), _SuffixTransformer("B")],
            max_variants=2,
        )
        self.assertEqual(pipeline.transform("q"), ["q", "qA"])

    def test_failure_degrades_to_original_or_raises_by_policy(self):
        with self.assertLogs("cn.mcheat.rag", level="WARNING"):
            self.assertEqual(
                QueryTransformPipeline(
                    [_BrokenTransformer()], on_error="use_original"
                ).transform("q"),
                ["q"],
            )
        with self.assertRaises(QueryTransformError):
            QueryTransformPipeline(
                [_BrokenTransformer()], on_error="raise"
            ).transform("q")

    def test_llm_transformer_only_needs_protocol_injection(self):
        self.assertEqual(
            LLMQueryTransformer(rewriter=_Rewriter()).transform("q"),
            ["q", "q 改写"],
        )
        self.assertEqual(LLMQueryTransformer().transform("q"), ["q"])


class TestCaches(unittest.TestCase):
    def test_embedding_query_cache_is_lru(self):
        cache = EmbeddingQueryCache(max_size=2)
        for index in range(3):
            cache.put("fp", f"q{index}", 1, (float(index), 0.0))
        self.assertEqual(len(cache), 2)
        self.assertIsNone(cache.get("fp", "q0", 1))
        self.assertEqual(cache.get("fp", "q2", 1), (2.0, 0.0))

    def test_result_cache_ttl_and_index_version(self):
        cache = InMemoryCache(enabled=True, query_lru_size=8, result_ttl_s=0.02)
        cache.put_result(("q", "{}", "cfg"), "result")
        self.assertEqual(cache.get_result(("q", "{}", "cfg")), "result")
        cache.set_index_version(2)
        self.assertIsNone(cache.get_result(("q", "{}", "cfg")))
        cache.put_result(("q", "{}", "cfg"), "result-v2")
        time.sleep(0.03)
        self.assertIsNone(cache.get_result(("q", "{}", "cfg")))

    def test_null_cache_never_stores(self):
        cache = NullCache()
        cache.put_query_vector("fp", "q", (1.0, 0.0))
        cache.put_result(("q", "{}", "cfg"), object())
        self.assertIsNone(cache.get_query_vector("fp", "q"))
        self.assertIsNone(cache.get_result(("q", "{}", "cfg")))

    def test_repeated_query_uses_one_embedding_call(self):
        class _CountingEmbedder(HashingEmbedder):
            def __init__(self) -> None:
                super().__init__(dimension=64)
                self.calls = 0

            def embed_query(self, text: str):
                self.calls += 1
                return super().embed_query(text)

        embedder = _CountingEmbedder()
        store = InMemoryVectorStore()
        child = make_child("a", "向量检索与 RRF 融合")
        store.upsert([child], embedder.embed_documents([child.text]))
        cache = InMemoryCache(enabled=True, query_lru_size=512, result_ttl_s=300)
        retriever = EmbeddingDenseRetriever(
            embedder=embedder, vector_store=store, cache=cache
        )
        for _ in range(100):
            self.assertTrue(retriever.search("向量检索", k=1))
        self.assertEqual(embedder.calls, 1)

    def test_ttl_cache_class_accepts_expired_entries(self):
        cache = RetrievalResultCache(ttl_s=0.01)
        cache.put("key", "value", 1)
        self.assertEqual(cache.get("key", 1), "value")
        time.sleep(0.02)
        self.assertIsNone(cache.get("key", 1))


class TestSystemIntegration(unittest.TestCase):
    def test_result_cache_hit_and_index_version_invalidation(self):
        cfg = RAGConfig.load(
            overrides={
                "components": {"cache": {"name": "memory", "params": {}}},
                "cache": {"enabled": True},
            },
            use_env=False,
        )
        system = RAGSystem(config=cfg)
        system.index([DATA_DIR])
        first = system.retrieve("RRF 融合")
        second = system.retrieve("RRF 融合")
        self.assertFalse(first.diagnostics["cache_hit"])
        self.assertTrue(second.diagnostics["cache_hit"])
        system.index([DATA_DIR])
        third = system.retrieve("RRF 融合")
        self.assertFalse(third.diagnostics["cache_hit"])

    def test_multi_query_fusion_and_recall_budget(self):
        cfg = RAGConfig.load(
            overrides={
                "components": {
                    "query_transformer": {"name": "rule_based", "params": {}}
                },
                "retrieval": {"max_recall_calls": 1},
            },
            use_env=False,
        )
        system = RAGSystem(config=cfg)
        system.index([DATA_DIR])
        result = system.retrieve("父子切分 Small-to-Big 解决什么问题")
        self.assertTrue(result.final)
        self.assertGreaterEqual(len(result.diagnostics["query_variants"]), 2)
        self.assertEqual(result.diagnostics["recall_calls"], 1)

    def test_rule_based_transform_does_not_regress_recall(self):
        golden = GoldenSet.from_json(GOLDEN_SET)
        positives = GoldenSet.from_cases(
            [case for case in golden if case.answerable], name=golden.name
        )
        identity = RAGSystem(config=RAGConfig.load(use_env=False))
        identity.index([DATA_DIR])
        transformed = RAGSystem(config=RAGConfig.load(
            overrides={"components": {
                "query_transformer": {"name": "rule_based", "params": {}}
            }},
            use_env=False,
        ))
        transformed.index([DATA_DIR])
        base = identity.evaluate(positives).summary
        after = transformed.evaluate(positives).summary
        self.assertGreaterEqual(after["context_recall"], base["context_recall"])
        self.assertGreaterEqual(after["answer_ready_rate"],
                                base["answer_ready_rate"])


if __name__ == "__main__":
    unittest.main()
