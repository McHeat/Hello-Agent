"""配置管理：默认值、深度合并、环境变量、校验与依赖引用解析。"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cn.mcheat.rag.config import RAGConfig, _deep_merge
from cn.mcheat.rag.core.registry import resolve_references
from cn.mcheat.rag.exceptions import ConfigError


class TestConfig(unittest.TestCase):
    def test_defaults_load_and_match_recommended_ranges(self):
        cfg = RAGConfig.load(use_env=False)
        # 《注意事项》第 1 条的建议区间
        self.assertEqual(cfg.retrieval.coarse_top_k, 50)
        self.assertEqual(cfg.retrieval.rerank_top_n, 5)
        self.assertEqual(cfg.component("fusion").name, "rrf")
        self.assertEqual(cfg.component("reranker").name, "lexical")
        self.assertEqual(cfg.component("context_builder").params["placement"], "edges")

    def test_user_config_deep_merges_without_losing_defaults(self):
        cfg = RAGConfig.load(
            overrides={"retrieval": {"coarse_top_k": 30, "rerank_top_n": 3}},
            use_env=False,
        )
        self.assertEqual(cfg.retrieval.coarse_top_k, 30)
        self.assertEqual(cfg.retrieval.rerank_top_n, 3)
        # 未覆盖的兄弟字段仍然保留
        self.assertEqual(cfg.retrieval.dense_k, 50)
        self.assertEqual(cfg.component("embedder").name, "hashing")

    def test_config_file_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({"retrieval": {"rerank_top_n": 4}}),
                            encoding="utf-8")
            cfg = RAGConfig.load(path=path, use_env=False)
            self.assertEqual(cfg.retrieval.rerank_top_n, 4)

    def test_env_override_and_type_coercion(self):
        env = {
            "RAG_COARSE_TOP_K": "42",
            "RAG_FALLBACK_ENABLED": "false",
            "RAG_CONTEXT_PLACEMENT": "sequential",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = RAGConfig.load()
        self.assertEqual(cfg.retrieval.coarse_top_k, 42)
        self.assertIs(cfg.fallback.enabled, False)
        self.assertEqual(cfg.component("context_builder").params["placement"],
                         "sequential")

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            RAGConfig.from_dict({"retrieval": {"typo_key": 1}})
        self.assertIn("typo_key", str(ctx.exception))

    def test_hard_invariants_are_enforced(self):
        cases = [
            {"retrieval": {"rerank_top_n": 80, "coarse_top_k": 10}},
            {"retrieval": {"enable_dense": False, "enable_sparse": False}},
            {"components": {"splitter": {"name": "character",
                                         "params": {"child_size": 2000,
                                                    "parent_size": 500}}}},
            {"components": {"context_builder": {"name": "edges",
                                                "params": {"placement": "magic"}}}},
            {"fallback": {"widen_factor": 1.0}},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ConfigError):
                    RAGConfig.load(overrides=overrides, use_env=False)

    def test_missing_component_is_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            RAGConfig.load(overrides={"components": {"embedder": None}},
                           use_env=False)
        self.assertIn("components", str(ctx.exception))
        # 直接构造缺少必需组件的配置，也必须在 validate 阶段被拦下
        cfg = RAGConfig.from_dict({"components": {"embedder": {"name": "hashing"}}})
        with self.assertRaises(ConfigError) as ctx2:
            cfg.validate()
        self.assertIn("缺少必需组件", str(ctx2.exception))

    def test_resolve_references_injects_built_components(self):
        sentinel = object()
        params = resolve_references(
            {"embedder": "@embedder", "nested": {"a": ["@vector_store"]}},
            {"embedder": sentinel, "vector_store": 7},
        )
        self.assertIs(params["embedder"], sentinel)
        self.assertEqual(params["nested"]["a"], [7])
        with self.assertRaises(ConfigError):
            resolve_references({"x": "@missing"}, {})

    def test_deep_merge_replaces_lists_not_appends(self):
        merged = _deep_merge({"a": [1, 2], "b": {"c": 1}}, {"a": [3], "b": {"d": 2}})
        self.assertEqual(merged, {"a": [3], "b": {"c": 1, "d": 2}})

    def test_save_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "saved.json"
            cfg = RAGConfig.load(use_env=False)
            cfg.save(path)
            restored = RAGConfig.load(path=path, use_env=False)
            self.assertEqual(cfg.to_dict(), restored.to_dict())


if __name__ == "__main__":
    unittest.main()
