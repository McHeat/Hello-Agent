"""可插拔向量库的两种内置实现（均零第三方依赖）。

* :class:`InMemoryVectorStore` —— 精确暴力余弦检索，随 ``RAGSystem.save()``
  整体序列化；适合中小语料、演示与单元测试。
* :class:`FileBackedVectorStore` —— 在上者基础上每次写操作后原子落盘到
  JSON，适合「离线进程建索引、在线进程直接读文件」的部署方式。

接 FAISS / Milvus / pgvector 时，实现 ``core.interfaces.VectorStore`` 即可，
上层 ``EmbeddingDenseRetriever`` 与在线检索链路完全不用改。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from .._utils import cosine_similarity, get_logger, l2_normalize
from ..core.interfaces import VectorStore
from ..core.registry import VECTOR_STORE_REGISTRY
from ..exceptions import ConfigError, DimensionMismatchError, VectorStoreError
from ..models import ChildChunk, Hit, Vector

logger = get_logger(__name__)


@VECTOR_STORE_REGISTRY.register("memory")
class InMemoryVectorStore(VectorStore):
    """内存向量库：``dict[id] = (vector, chunk)`` + 线性扫描。

    Args:
        metric: ``"cosine"``（写入时归一化）或 ``"dot"``（原样存储）。
        dimension: 可选；不传则由第一批写入的向量推断，并据此校验后续写入。
            **换 embedding 模型会触发** ``DimensionMismatchError``，需重建索引。
    """

    def __init__(self, metric: str = "cosine",
                 dimension: int | None = None) -> None:
        if metric not in {"cosine", "dot"}:
            raise ConfigError(f"未知的向量相似度度量 {metric!r}；可选 cosine/dot")
        if dimension is not None and dimension <= 0:
            raise ConfigError("dimension 必须为正整数")
        self.metric = metric
        self._dimension = dimension
        self._vectors: dict[str, Vector] = {}
        self._chunks: dict[str, ChildChunk] = {}

    # -- 写入 ---------------------------------------------------------------

    def upsert(self, chunks: Sequence[ChildChunk],
               vectors: Sequence[Vector]) -> int:
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"子块数({len(chunks)})与向量数({len(vectors)})不一致"
            )
        for chunk, raw_vector in zip(chunks, vectors):
            vector = tuple(float(x) for x in raw_vector)
            if not vector:
                raise VectorStoreError(f"子块 {chunk.id} 的向量为空")
            if self._dimension is None:
                self._dimension = len(vector)
            elif len(vector) != self._dimension:
                raise DimensionMismatchError(
                    f"子块 {chunk.id} 向量维度 {len(vector)} 与库维度 "
                    f"{self._dimension} 不符 —— 通常意味着换了 embedding 模型，"
                    f"请清空并重建索引（或把 vector_store.dimension 与模型对齐）。"
                )
            self._vectors[chunk.id] = (
                l2_normalize(vector) if self.metric == "cosine" else vector
            )
            self._chunks[chunk.id] = chunk
        logger.debug("向量库 upsert %d 条，当前 %d 条", len(chunks), self.count())
        return len(chunks)

    # -- 查询 ---------------------------------------------------------------

    def search(self, query_vector: Vector, k: int,
               metadata_filter: dict[str, Any] | None = None) -> list[Hit]:
        if k <= 0:
            return []
        query = tuple(float(x) for x in query_vector)
        if self._dimension is not None and len(query) != self._dimension:
            raise DimensionMismatchError(
                f"查询向量维度 {len(query)} 与库维度 {self._dimension} 不符"
            )
        if self.metric == "cosine":
            query = l2_normalize(query)

        scored: list[tuple[float, str]] = []
        for chunk_id, vector in self._vectors.items():
            chunk = self._chunks[chunk_id]
            if metadata_filter and not self._matches(chunk, metadata_filter):
                continue
            scored.append((cosine_similarity(query, vector), chunk_id))

        # 分数降序；分数相同按 ID 排序，保证结果可复现（评测/回归不抖动）
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            Hit(chunk=self._chunks[chunk_id], score=float(score), source="dense")
            for score, chunk_id in scored[:k]
        ]

    @staticmethod
    def _matches(chunk: ChildChunk, where: dict[str, Any]) -> bool:
        """元数据精确匹配；所有条件都满足才保留。"""
        return all(chunk.metadata.get(key) == value for key, value in where.items())

    def delete(self, ids: Sequence[str]) -> int:
        removed = 0
        for chunk_id in ids:
            if chunk_id in self._vectors:
                self._vectors.pop(chunk_id, None)
                self._chunks.pop(chunk_id, None)
                removed += 1
        return removed

    def ids(self) -> set[str]:
        return set(self._vectors)

    def count(self) -> int:
        return len(self._vectors)

    @property
    def dimension(self) -> int | None:
        return self._dimension

    # -- 持久化 -------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "metric": self.metric,
            "dimension": self._dimension,
            "records": [
                [chunk_id, list(self._vectors[chunk_id]),
                 self._chunks[chunk_id].to_dict()]
                for chunk_id in sorted(self._vectors)
            ],
        }

    def import_state(self, state: dict[str, Any]) -> None:
        if state.get("external"):
            # 由子类（FileBackedVectorStore）自行从磁盘恢复
            return
        self.metric = state.get("metric", "cosine")
        self._dimension = state.get("dimension")
        self._vectors.clear()
        self._chunks.clear()
        for chunk_id, vector, payload in state.get("records", []):
            self._vectors[chunk_id] = tuple(float(x) for x in vector)
            self._chunks[chunk_id] = ChildChunk.from_dict(payload)


@VECTOR_STORE_REGISTRY.register("file")
class FileBackedVectorStore(InMemoryVectorStore):
    """每次写操作后原子落盘的向量库。

    ``export_state`` 只返回文件路径（``external=True``），避免 ``RAGSystem.save``
    把同一份向量复制两遍；``import_state`` 会触发从磁盘重新加载。

    Args:
        path: JSON 索引文件路径。
        auto_save: ``False`` 时需手动调用 :meth:`flush`（批量导入更快）。
    """

    def __init__(self, path: str | Path, metric: str = "cosine",
                 dimension: int | None = None, auto_save: bool = True) -> None:
        super().__init__(metric=metric, dimension=dimension)
        self.path = Path(path)
        self.auto_save = auto_save
        if self.path.is_file():
            self._load_from_disk()
            logger.info("从 %s 恢复 %d 条向量", self.path, self.count())

    def upsert(self, chunks: Sequence[ChildChunk],
               vectors: Sequence[Vector]) -> int:
        count = super().upsert(chunks, vectors)
        if self.auto_save:
            self.flush()
        return count

    def delete(self, ids: Sequence[str]) -> int:
        count = super().delete(ids)
        if self.auto_save and count:
            self.flush()
        return count

    def flush(self) -> None:
        """把当前状态原子写入磁盘。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.path, super().export_state())

    def _load_from_disk(self) -> None:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"向量索引文件损坏：{self.path}：{exc}") from exc
        InMemoryVectorStore.import_state(self, state)

    def export_state(self) -> dict[str, Any]:
        return {"external": True, "backend": "file", "path": str(self.path)}

    def import_state(self, state: dict[str, Any]) -> None:
        expected = state.get("path")
        if expected and Path(expected) != self.path:
            logger.warning(
                "索引产物记录的向量库路径 %s 与当前 %s 不同，继续使用当前路径",
                expected, self.path,
            )
        if self.path.is_file():
            self._load_from_disk()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """临时文件 + ``os.replace``，避免在线进程读到写了一半的索引。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


__all__: Sequence[str] = ["InMemoryVectorStore", "FileBackedVectorStore"]
