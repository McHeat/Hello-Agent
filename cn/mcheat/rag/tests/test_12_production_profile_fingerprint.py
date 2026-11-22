"""P0-2：中文生产配置档、query/passage 前缀、模型指纹与旧产物兼容。"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cn.mcheat.rag import RAGConfig, RAGSystem
from cn.mcheat.rag.config import load_profile
from cn.mcheat.rag.embeddings.adapters import LangChainEmbedder
from cn.mcheat.rag.embeddings.hashing import HashingEmbedder
from cn.mcheat.rag.exceptions import (
    DependencyMissingError,
    IndexFingerprintMismatchError,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


class _FakeEmbeddings:
    """记录每次调用的文本，避免任何真实模型依赖。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def embed_documents(self, texts):
        self.calls.append(("documents", list(texts)))
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.calls.append(("query", text))
        return [1.0, 0.0]


def _embedder(model: str = "BAAI/bge-m3",
              preset: str = "bge") -> LangChainEmbedder:
    return LangChainEmbedder(
        embeddings=_FakeEmbeddings(), model=model, prefix_preset=preset
    )


class TestProductionProfile(unittest.TestCase):
    def test_production_profile_writes_full_params_for_switched_components(self):
        cfg = load_profile("production", use_env=False)
        self.assertEqual(cfg.profile, "production")
        self.assertTrue(cfg.fallback.threshold_provisional)
        self.assertEqual(cfg.component("embedder").name, "langchain")
        embedder_params = cfg.component("embedder").params
        self.assertEqual(embedder_params["model"], "BAAI/bge-m3")
        self.assertEqual(embedder_params["prefix_preset"], "bge")
        self.assertIn("query_prefix", embedder_params)
        self.assertIn("document_prefix", embedder_params)
        self.assertEqual(
            cfg.component("reranker").params["model_name"],
            "BAAI/bge-reranker-large",
        )
        self.assertEqual(cfg.component("vector_store").name, "sqlite")
        self.assertEqual(cfg.component("parent_store").name, "sqlite")
        self.assertTrue(cfg.retrieval.parallel_recall)
        self.assertGreater(cfg.retrieval.min_best_score, 0.0)

    def test_rag_profile_env_loads_production_patch(self):
        with patch.dict(os.environ, {"RAG_PROFILE": "production"}, clear=False):
            cfg = RAGConfig.load()
        self.assertEqual(cfg.profile, "production")
        self.assertEqual(cfg.component("embedder").name, "langchain")

    def test_missing_optional_dependency_has_install_hint(self):
        # 显式模拟「可选依赖未安装」，避免用例结果取决于当前环境是否恰好装了
        # langchain-huggingface（装了就会真的去下载模型，离线时抛 OSError）。
        with patch.dict(sys.modules, {"langchain_huggingface": None}):
            with self.assertRaises(DependencyMissingError) as ctx:
                RAGSystem(config=load_profile("production", use_env=False))
        self.assertIn("pip install", str(ctx.exception))

    def test_provisional_threshold_warning_is_emitted(self):
        with self.assertLogs("cn.mcheat.rag", level="WARNING"):
            from cn.mcheat.rag.config import FallbackConfig
            from cn.mcheat.rag.retrieval.fallback import FallbackChain
            FallbackChain.from_config(FallbackConfig(threshold_provisional=True))


class TestPrefixesAndFingerprint(unittest.TestCase):
    def test_bge_query_prefix_is_injected_only_on_query_side(self):
        embedder = _embedder(preset="bge")
        embedder.embed_query("问题")
        texts = embedder.embed_documents(["文档"])
        self.assertTrue(embedder._embeddings.calls[0][1].startswith(
            embedder.query_prefix
        ))
        self.assertTrue(embedder.query_prefix)
        self.assertEqual(len(texts), 1)

    def test_e5_prefixes_are_distinct_for_query_and_passage(self):
        embedder = _embedder(preset="e5")
        embedder.embed_query("question")
        embedder.embed_documents(["passage"])
        query_text = embedder._embeddings.calls[-2][1]
        document_text = embedder._embeddings.calls[-1][1][0]
        self.assertEqual(query_text, "query: question")
        self.assertEqual(document_text, "passage: passage")

    def test_fingerprint_is_stable_and_sensitive_to_model_and_prefix(self):
        self.assertEqual(
            HashingEmbedder(dimension=256).fingerprint(),
            HashingEmbedder(dimension=256).fingerprint(),
        )
        self.assertNotEqual(
            HashingEmbedder(dimension=256).fingerprint(),
            HashingEmbedder(dimension=128).fingerprint(),
        )
        base = _embedder(model="BAAI/bge-m3", preset="bge")
        self.assertEqual(base.fingerprint(),
                         _embedder(model="BAAI/bge-m3", preset="bge").fingerprint())
        self.assertNotEqual(
            base.fingerprint(),
            _embedder(model="BAAI/bge-large-zh-v1.5", preset="bge").fingerprint(),
        )
        self.assertNotEqual(
            base.fingerprint(),
            _embedder(model="BAAI/bge-m3", preset="e5").fingerprint(),
        )


class TestArtifactCompatibility(unittest.TestCase):
    def _saved_artifact(self, tmp: str) -> Path:
        system = RAGSystem(config=RAGConfig.load(use_env=False))
        system.index([DATA_DIR])
        return system.save(tmp)

    def test_old_artifact_without_new_fields_only_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._saved_artifact(tmp)
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key in ("embedding_fingerprint", "index_version", "manifest"):
                payload.pop(key, None)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertLogs("cn.mcheat.rag", level="WARNING"):
                restored = RAGSystem.load(tmp)
            self.assertGreater(restored.component("dense_retriever").count(), 0)

    def test_same_dimension_model_swap_is_blocked_with_rebuild_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._saved_artifact(tmp)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["embedding_fingerprint"] = "swapped-model-fingerprint"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(IndexFingerprintMismatchError) as ctx:
                RAGSystem.load(tmp)
            self.assertIn("重建索引", str(ctx.exception))

    def test_roundtrip_artifact_keeps_fingerprint_and_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._saved_artifact(tmp)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(payload["embedding_fingerprint"])
            self.assertGreaterEqual(payload["index_version"], 1)
            self.assertIn("manifest", payload)
            restored = RAGSystem.load(tmp)
            self.assertGreaterEqual(restored.indexer.index_version, 1)


if __name__ == "__main__":
    unittest.main()
