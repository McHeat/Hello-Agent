"""召回通路实现：把低层 ``Embedder + VectorStore`` 封装成统一的 ``RecallRetriever``。

``HybridRetriever`` 只依赖 ``RecallRetriever`` 接口，因此稠密 / 稀疏通路可以
自由替换。本模块提供：

* :class:`EmbeddingDenseRetriever` —— 默认稠密通路（本框架 Embedder + VectorStore）；
* :class:`LangChainDenseRetriever` —— 适配器，直接复用 LangChain 的
  ``VectorStore``（Chroma / FAISS / ...），证明向量库可整体替换；
* :class:`NullSparseRetriever` —— 空通路，用于单通路消融实验。
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from .._utils import batched, get_logger
from ..core.interfaces import Embedder, RecallRetriever, VectorStore
from ..core.registry import DENSE_RETRIEVER_REGISTRY, SPARSE_RETRIEVER_REGISTRY
from ..exceptions import ConfigError, DependencyMissingError
from ..models import ChildChunk, Hit, Vector

logger = get_logger(__name__)

# 在 LangChain Document.metadata 里回填完整子块，便于 search 时无损还原
_CHUNK_PAYLOAD_KEY = "__rag_child_chunk"


@DENSE_RETRIEVER_REGISTRY.register("embedding")
class EmbeddingDenseRetriever(RecallRetriever):
    """默认稠密通路：``query -> Embedder -> VectorStore.search``。

    Args:
        embedder: 向量化器（索引侧与查询侧共用同一实例）。
        vector_store: 可插拔向量库。
        batch_size: 离线批量 embedding 的大小。
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore,
                 batch_size: int = 32) -> None:
        if not isinstance(embedder, Embedder):
            raise ConfigError("embedder 必须实现 core.interfaces.Embedder")
        if not isinstance(vector_store, VectorStore):
            raise ConfigError("vector_store 必须实现 core.interfaces.VectorStore")
        if batch_size <= 0:
            raise ConfigError("batch_size 必须为正整数")
        self.embedder = embedder
        self.vector_store = vector_store
        self.batch_size = batch_size

    def index(self, chunks: Sequence[ChildChunk]) -> int:
        if not chunks:
            return 0
        vectors: list[Vector] = []
        for batch in batched(list(chunks), self.batch_size):
            vectors.extend(self.embedder.embed_documents([c.text for c in batch]))
        return self.vector_store.upsert(list(chunks), vectors)

    def search(self, query: str, k: int,
               metadata_filter: dict[str, Any] | None = None) -> list[Hit]:
        vector = self.embedder.embed_query(query)
        hits = self.vector_store.search(vector, k, metadata_filter)
        for hit in hits:
            hit.source = "dense"
        return hits

    def delete(self, ids: Sequence[str]) -> int:
        return self.vector_store.delete(ids)

    def ids(self) -> set[str]:
        return self.vector_store.ids()

    def count(self) -> int:
        return self.vector_store.count()

    def export_state(self) -> dict[str, Any]:
        return {"version": 1, "vector_store": self.vector_store.export_state()}

    def import_state(self, state: dict[str, Any]) -> None:
        self.vector_store.import_state(state.get("vector_store", {}))


@DENSE_RETRIEVER_REGISTRY.register("langchain_chroma")
class LangChainDenseRetriever(RecallRetriever):
    """适配器：用 LangChain 的 VectorStore 作为稠密通路。

    推荐直接注入已构造好的 ``vector_store``（任何实现 ``add_texts`` /
    ``similarity_search`` / ``delete`` 的对象），这样连 LangChain 都不强依赖，
    也便于用测试替身验证。未注入时按参数懒加载 Chroma + Embeddings。

    注意：多数 LangChain VectorStore 不返回相似度分数，因此 :meth:`search`
    用「名次倒数」作为近似分数交给融合器（RRF 本就只吃名次，影响很小）。
    """

    def __init__(self, vector_store: Any | None = None,
                 collection_name: str = "rag",
                 persist_directory: str | None = None,
                 embedding_provider: str = "huggingface",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 **kwargs: Any) -> None:
        self.batch_size = batch_size
        self._known_ids: set[str] = set()
        self._kwargs = kwargs
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        if vector_store is not None:
            self._vs = vector_store
        else:
            self._vs = _build_chroma(
                collection_name, persist_directory,
                embedding_provider, embedding_model, kwargs,
            )
        self._validate_store()

    def _validate_store(self) -> None:
        for method in ("add_texts", "similarity_search"):
            if not callable(getattr(self._vs, method, None)):
                raise ConfigError(
                    f"注入的 vector_store 缺少 {method}()；"
                    f"需要兼容 LangChain VectorStore 接口。"
                )

    def index(self, chunks: Sequence[ChildChunk]) -> int:
        if not chunks:
            return 0
        items = list(chunks)
        for batch in batched(items, self.batch_size):
            self._vs.add_texts(
                texts=[c.text for c in batch],
                metadatas=[
                    {**c.metadata,
                     _CHUNK_PAYLOAD_KEY: json.dumps(c.to_dict(), ensure_ascii=False)}
                    for c in batch
                ],
                ids=[c.id for c in batch],
            )
        self._known_ids.update(c.id for c in items)
        return len(items)

    def search(self, query: str, k: int,
               metadata_filter: dict[str, Any] | None = None) -> list[Hit]:
        try:
            docs = self._vs.similarity_search(query, k=k, **(
                {"filter": metadata_filter} if metadata_filter else {}
            ))
        except TypeError:
            # 部分 VectorStore 不支持 filter 关键字，降级为全库检索并本地过滤
            logger.warning("底层 VectorStore 不支持 filter 参数，改为本地元数据过滤")
            docs = self._vs.similarity_search(query, k=k)
            if metadata_filter:
                docs = [
                    doc for doc in docs
                    if all(doc.metadata.get(key) == value
                           for key, value in metadata_filter.items())
                ]

        hits: list[Hit] = []
        for rank, doc in enumerate(docs):
            payload = doc.metadata.get(_CHUNK_PAYLOAD_KEY)
            if not payload:
                logger.warning("LangChain 文档缺少 %s，跳过该命中", _CHUNK_PAYLOAD_KEY)
                continue
            chunk = ChildChunk.from_dict(json.loads(payload))
            # 名次倒数近似分数：RRF 只关心顺序，不影响融合结果
            hits.append(Hit(chunk=chunk, score=1.0 / (rank + 1), source="dense"))
        return hits

    def delete(self, ids: Sequence[str]) -> int:
        deleter = getattr(self._vs, "delete", None)
        if callable(deleter):
            try:
                deleter(ids=list(ids))
            except TypeError:
                deleter(list(ids))
        self._known_ids.difference_update(ids)
        return len(ids)

    def ids(self) -> set[str]:
        return set(self._known_ids)

    def count(self) -> int:
        return len(self._known_ids)

    def export_state(self) -> dict[str, Any]:
        return {
            "external": True,
            "backend": "langchain",
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "known_ids": sorted(self._known_ids),
        }

    def import_state(self, state: dict[str, Any]) -> None:
        self._known_ids = set(state.get("known_ids", []))


def _build_chroma(collection_name: str, persist_directory: str | None,
                  embedding_provider: str, embedding_model: str,
                  kwargs: dict[str, Any]) -> Any:
    """懒加载 Chroma + LangChain Embeddings。"""
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError as exc:
        raise DependencyMissingError(
            "langchain-community chromadb", "Chroma 向量库",
            "或直接注入 vector_store 对象",
        ) from exc
    from ..embeddings.adapters import _build_embeddings

    embeddings = _build_embeddings(embedding_provider, embedding_model, kwargs)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


@SPARSE_RETRIEVER_REGISTRY.register("null")
class NullSparseRetriever(RecallRetriever):
    """空稀疏通路：用于「只开向量召回」的消融实验与测试。"""

    def index(self, chunks: Sequence[ChildChunk]) -> int:
        return 0

    def search(self, query: str, k: int,
               metadata_filter: dict[str, Any] | None = None) -> list[Hit]:
        return []

    def delete(self, ids: Sequence[str]) -> int:
        return 0

    def ids(self) -> set[str]:
        return set()

    def count(self) -> int:
        return 0

    def export_state(self) -> dict[str, Any]:
        return {"version": 1}

    def import_state(self, state: dict[str, Any]) -> None:
        return None


__all__: Sequence[str] = [
    "EmbeddingDenseRetriever", "LangChainDenseRetriever", "NullSparseRetriever",
]
