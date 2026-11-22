"""基于标准库 ``sqlite3`` 的二进制向量库（P0-3 可选持久化升级）。

复用 ``SqliteParentStore`` 的 ``CREATE TABLE`` + ``INSERT OR REPLACE`` +
原子事务模式：向量用二进制 blob 存，避免 JSON 每次全量重写的写放大；
``export_state`` 只返回 ``external=True`` 路径，数据由 SQLite 自己管理。

连接设置 ``check_same_thread=False`` 并配 ``RLock``，供并行召回 / 服务化
场景安全并发读写。检索仍是精确暴力余弦（ANN 留待 P2）。
"""
from __future__ import annotations

import json
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any, Sequence

from .._utils import cosine_similarity, get_logger, l2_normalize
from ..core.interfaces import VectorStore
from ..core.registry import VECTOR_STORE_REGISTRY
from ..exceptions import (
    ConfigError,
    DimensionMismatchError,
    PersistenceError,
    VectorStoreError,
)
from ..models import ChildChunk, Hit, Vector

logger = get_logger(__name__)


@VECTOR_STORE_REGISTRY.register("sqlite")
class SqliteVectorStore(VectorStore):
    """把向量以二进制 blob 存入 SQLite 的可插拔向量库。

    Args:
        path: SQLite 数据库文件路径。
        metric: ``"cosine"``（写入时归一化）或 ``"dot"``。
        dimension: 可选，显式维度；不传时由第一批写入 / 已有数据推断。
    """

    def __init__(self, path: str | Path = "vectors.sqlite3",
                 metric: str = "cosine",
                 dimension: int | None = None) -> None:
        if metric not in {"cosine", "dot"}:
            raise ConfigError(f"未知的向量相似度度量 {metric!r}；可选 cosine/dot")
        if dimension is not None and dimension <= 0:
            raise ConfigError("dimension 必须为正整数")
        self.path = str(Path(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self._dimension = dimension
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS vectors ("
                "  id TEXT PRIMARY KEY,"
                "  vector BLOB NOT NULL,"
                "  chunk TEXT NOT NULL"
                ")"
            )
            self._conn.commit()
            if self._dimension is None:
                row = self._conn.execute(
                    "SELECT vector FROM vectors LIMIT 1"
                ).fetchone()
                if row is not None:
                    self._dimension = len(row[0]) // 8
        except sqlite3.Error as exc:
            raise PersistenceError(
                f"初始化向量数据库 {self.path} 失败：{exc}"
            ) from exc

    # -- 写入 ---------------------------------------------------------------

    def upsert(self, chunks: Sequence[ChildChunk],
               vectors: Sequence[Vector]) -> int:
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"子块数({len(chunks)})与向量数({len(vectors)})不一致"
            )
        with self._lock:
            records: list[tuple[str, bytes, str]] = []
            for chunk, raw_vector in zip(chunks, vectors):
                vector = tuple(float(x) for x in raw_vector)
                if not vector:
                    raise VectorStoreError(f"子块 {chunk.id} 的向量为空")
                if self._dimension is None:
                    self._dimension = len(vector)
                elif len(vector) != self._dimension:
                    raise DimensionMismatchError(
                        f"子块 {chunk.id} 向量维度 {len(vector)} 与库维度 "
                        f"{self._dimension} 不符 —— 换了 embedding 模型，请重建索引。"
                    )
                stored = (l2_normalize(vector) if self.metric == "cosine"
                          else vector)
                records.append((
                    chunk.id,
                    struct.pack(f"{len(stored)}d", *stored),
                    json.dumps(chunk.to_dict(), ensure_ascii=False),
                ))
            try:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO vectors (id, vector, chunk) "
                    "VALUES (?, ?, ?)",
                    records,
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                raise PersistenceError(f"写入向量数据库失败：{exc}") from exc
            logger.debug("SQLite 向量库 upsert %d 条", len(records))
            return len(records)

    # -- 查询 ---------------------------------------------------------------

    def search(self, query_vector: Vector, k: int,
               metadata_filter: dict[str, Any] | None = None) -> list[Hit]:
        if k <= 0:
            return []
        query = tuple(float(x) for x in query_vector)
        with self._lock:
            if self._dimension is not None and len(query) != self._dimension:
                raise DimensionMismatchError(
                    f"查询向量维度 {len(query)} 与库维度 {self._dimension} 不符"
                )
            if self.metric == "cosine":
                query = l2_normalize(query)
            try:
                rows = self._conn.execute(
                    "SELECT id, vector, chunk FROM vectors"
                ).fetchall()
            except sqlite3.Error as exc:
                raise PersistenceError(f"读取向量数据库失败：{exc}") from exc

            scored: list[tuple[float, str, ChildChunk]] = []
            for chunk_id, blob, payload in rows:
                chunk = ChildChunk.from_dict(json.loads(payload))
                if metadata_filter and not self._matches(chunk, metadata_filter):
                    continue
                stored = struct.unpack(f"{len(blob) // 8}d", blob)
                scored.append((cosine_similarity(query, stored), chunk_id, chunk))
            scored.sort(key=lambda item: (-item[0], item[1]))
            return [
                Hit(chunk=chunk, score=float(score), source="dense")
                for score, _, chunk in scored[:k]
            ]

    @staticmethod
    def _matches(chunk: ChildChunk, where: dict[str, Any]) -> bool:
        return all(chunk.metadata.get(key) == value for key, value in where.items())

    def get_many(self, ids: Sequence[str]) -> dict[str, ChildChunk]:
        unique = list(dict.fromkeys(ids))
        if not unique:
            return {}
        placeholders = ",".join("?" for _ in unique)
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT id, chunk FROM vectors "
                    f"WHERE id IN ({placeholders})", unique,
                ).fetchall()
            except sqlite3.Error as exc:
                raise PersistenceError(f"读取向量数据库失败：{exc}") from exc
        return {row[0]: ChildChunk.from_dict(json.loads(row[1])) for row in rows}

    def delete(self, ids: Sequence[str]) -> int:
        unique = list(dict.fromkeys(ids))
        if not unique:
            return 0
        placeholders = ",".join("?" for _ in unique)
        with self._lock:
            try:
                cursor = self._conn.execute(
                    f"DELETE FROM vectors WHERE id IN ({placeholders})", unique
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                raise PersistenceError(f"删除向量失败：{exc}") from exc
        return cursor.rowcount

    def delete_by_source(self, source: str) -> int:
        with self._lock:
            matched = [
                chunk_id for chunk_id, chunk in
                self.get_many(sorted(self.ids())).items()
                if source in {chunk.metadata.get("source"),
                              chunk.metadata.get("filename")}
            ]
            return self.delete(matched)

    def ids(self) -> set[str]:
        with self._lock:
            try:
                rows = self._conn.execute("SELECT id FROM vectors").fetchall()
            except sqlite3.Error as exc:
                raise PersistenceError(f"枚举向量 ID 失败：{exc}") from exc
        return {row[0] for row in rows}

    def count(self) -> int:
        with self._lock:
            try:
                return int(self._conn.execute(
                    "SELECT COUNT(*) FROM vectors"
                ).fetchone()[0])
            except sqlite3.Error as exc:
                raise PersistenceError(f"统计向量数量失败：{exc}") from exc

    # -- 持久化 -------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        """外部持久化：只记录路径，数据由 SQLite 自己管理。"""
        return {"external": True, "backend": "sqlite", "path": self.path}

    def import_state(self, state: dict[str, Any]) -> None:
        expected = state.get("path")
        if expected and expected != self.path:
            logger.warning(
                "索引产物记录的向量库路径 %s 与当前 %s 不同，继续使用当前路径",
                expected, self.path,
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__: Sequence[str] = ["SqliteVectorStore"]
