"""ParentStore：父块的「离线写 / 在线读」桥梁。

父子切分（Small-to-Big）里，向量库只存子块，父块必须另存一份，检索命中
后按 ``parent_id`` 取回。因此 ParentStore 的持久化能力决定了「离线建索引」
与「在线检索」能否分成两个进程 / 两次部署。

两种实现，体现可插拔：

* :class:`InMemoryParentStore` —— 零依赖，随 ``RAGSystem.save()`` 整体导出；
* :class:`SqliteParentStore` —— 标准库 ``sqlite3``，父块量大时独立落盘，
  在线服务可直接 ``open()`` 读取，不必把全量父块加载进内存。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from .._utils import get_logger
from ..core.interfaces import ParentStore
from ..core.registry import PARENT_STORE_REGISTRY
from ..exceptions import PersistenceError
from ..models import ParentChunk

logger = get_logger(__name__)


def _parent_to_dict(parent: ParentChunk) -> dict[str, Any]:
    return {
        "id": parent.id, "doc_id": parent.doc_id, "text": parent.text,
        "metadata": parent.metadata, "index": parent.index,
    }


def _parent_from_dict(data: dict[str, Any]) -> ParentChunk:
    return ParentChunk(
        id=data["id"], doc_id=data["doc_id"], text=data["text"],
        metadata=data.get("metadata", {}), index=data.get("index", 0),
    )


@PARENT_STORE_REGISTRY.register("memory")
class InMemoryParentStore(ParentStore):
    """基于 dict 的父块存储（默认）。"""

    def __init__(self) -> None:
        self._data: dict[str, ParentChunk] = {}

    def put(self, parents: Sequence[ParentChunk]) -> int:
        for parent in parents:
            self._data[parent.id] = parent
        return len(parents)

    def get(self, parent_id: str) -> ParentChunk | None:
        return self._data.get(parent_id)

    def get_many(self, ids: Sequence[str]) -> dict[str, ParentChunk]:
        return {pid: self._data[pid] for pid in ids if pid in self._data}

    def delete(self, ids: Sequence[str]) -> int:
        removed = 0
        for pid in ids:
            if self._data.pop(pid, None) is not None:
                removed += 1
        return removed

    def ids(self) -> set[str]:
        return set(self._data)

    def count(self) -> int:
        return len(self._data)

    def export_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "parents": [_parent_to_dict(p) for p in self._data.values()],
        }

    def import_state(self, state: dict[str, Any]) -> None:
        self._data = {
            item["id"]: _parent_from_dict(item) for item in state.get("parents", [])
        }

    def save(self, path: str | Path) -> None:
        """单独落盘（不依赖 RAGSystem.save），便于在线服务直接读取。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(target, self.export_state())

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryParentStore":
        """从 :meth:`save` 的产物恢复。"""
        store = cls()
        store.import_state(json.loads(Path(path).read_text(encoding="utf-8")))
        return store


@PARENT_STORE_REGISTRY.register("sqlite")
class SqliteParentStore(ParentStore):
    """基于 ``sqlite3`` 的父块存储（标准库，无第三方依赖）。

    Args:
        path: 数据库文件路径。父块直接落盘，``RAGSystem.save()`` 只记录路径，
            不会把全量文本复制进索引产物 JSON。
    """

    def __init__(self, path: str | Path = "parents.sqlite3") -> None:
        self.path = str(Path(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        try:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS parents ("
                "  id TEXT PRIMARY KEY,"
                "  payload TEXT NOT NULL"
                ")"
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise PersistenceError(f"初始化父块数据库 {self.path} 失败：{exc}") from exc

    def put(self, parents: Sequence[ParentChunk]) -> int:
        try:
            self._conn.executemany(
                "INSERT OR REPLACE INTO parents (id, payload) VALUES (?, ?)",
                [
                    (p.id, json.dumps(_parent_to_dict(p), ensure_ascii=False))
                    for p in parents
                ],
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise PersistenceError(f"写入父块数据库失败：{exc}") from exc
        return len(parents)

    def get(self, parent_id: str) -> ParentChunk | None:
        return self.get_many([parent_id]).get(parent_id)

    def get_many(self, ids: Sequence[str]) -> dict[str, ParentChunk]:
        unique = list(dict.fromkeys(ids))
        if not unique:
            return {}
        placeholders = ",".join("?" for _ in unique)
        try:
            rows = self._conn.execute(
                f"SELECT id, payload FROM parents WHERE id IN ({placeholders})",
                unique,
            ).fetchall()
        except sqlite3.Error as exc:
            raise PersistenceError(f"读取父块数据库失败：{exc}") from exc
        return {row[0]: _parent_from_dict(json.loads(row[1])) for row in rows}

    def delete(self, ids: Sequence[str]) -> int:
        unique = list(dict.fromkeys(ids))
        if not unique:
            return 0
        placeholders = ",".join("?" for _ in unique)
        try:
            cursor = self._conn.execute(
                f"DELETE FROM parents WHERE id IN ({placeholders})", unique
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise PersistenceError(f"删除父块失败：{exc}") from exc
        return cursor.rowcount

    def ids(self) -> set[str]:
        try:
            rows = self._conn.execute("SELECT id FROM parents").fetchall()
        except sqlite3.Error as exc:
            raise PersistenceError(f"枚举父块 ID 失败：{exc}") from exc
        return {row[0] for row in rows}

    def count(self) -> int:
        try:
            return int(self._conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0])
        except sqlite3.Error as exc:
            raise PersistenceError(f"统计父块数量失败：{exc}") from exc

    def export_state(self) -> dict[str, Any]:
        """外部持久化：只记录路径，数据由 SQLite 自己管理。"""
        return {"external": True, "backend": "sqlite", "path": self.path}

    def import_state(self, state: dict[str, Any]) -> None:
        """无需导入数据；仅校验路径一致，避免误连到另一个库。"""
        expected = state.get("path")
        if expected and expected != self.path:
            logger.warning(
                "索引产物记录的父块库路径 %s 与当前 %s 不同，继续使用当前路径",
                expected, self.path,
            )

    def close(self) -> None:
        """关闭连接（在线服务退出时调用）。"""
        self._conn.close()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """临时文件 + ``os.replace`` 原子写，避免写一半被读到损坏的索引。"""
    import os

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


__all__: Sequence[str] = ["InMemoryParentStore", "SqliteParentStore"]
