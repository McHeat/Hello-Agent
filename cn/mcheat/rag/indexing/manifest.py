"""文档级增量索引的准绳：来源内容哈希 → 子块 / 父块 ID 的清单。

``SourceManifest`` 记录每个来源（文件路径）的 ``content_hash``、由它切出的
``child_ids`` 与 ``parent_ids``。增量构建时先用 ``_utils.content_hash`` 做
变更检测，只重切分变更来源，删除旧 ID 后写入新块；``prune_stale`` 仍只
在**全量模式**生效。

``index_version`` 同时驱动缓存失效与增量对账：每次成功 ``index()`` 递增。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .._utils import get_logger

logger = get_logger(__name__)


@dataclass
class SourceRecord:
    """单个来源的快照。"""

    content_hash: str
    child_ids: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "child_ids": list(self.child_ids),
            "parent_ids": list(self.parent_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceRecord":
        return cls(
            content_hash=str(data.get("content_hash", "")),
            child_ids=list(data.get("child_ids", [])),
            parent_ids=list(data.get("parent_ids", [])),
        )


@dataclass
class SourceManifest:
    """一次索引构建后所有来源的快照集合。"""

    index_version: int = 0
    embedding_fingerprint: str = ""
    sources: dict[str, SourceRecord] = field(default_factory=dict)

    def changed_sources(self, current_hashes: Mapping[str, str]) -> list[str]:
        """返回新增 / 内容变更 / 已删除的来源（按名称排序，结果可复现）。"""
        keys = sorted(set(self.sources) | set(current_hashes))
        changed: list[str] = []
        for source in keys:
            record = self.sources.get(source)
            if record is None or record.content_hash != current_hashes.get(source):
                changed.append(source)
        return changed

    def bump(self) -> "SourceManifest":
        """返回 ``index_version`` 递增后的新快照（不修改自身）。"""
        return SourceManifest(
            index_version=self.index_version + 1,
            embedding_fingerprint=self.embedding_fingerprint,
            sources={key: SourceRecord.from_dict(record.as_dict())
                     for key, record in self.sources.items()},
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "index_version": self.index_version,
            "embedding_fingerprint": self.embedding_fingerprint,
            "sources": {
                key: record.as_dict() for key, record in sorted(self.sources.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceManifest":
        raw_sources = data.get("sources", {})
        sources = {
            str(key): (value if isinstance(value, SourceRecord)
                       else SourceRecord.from_dict(value))
            for key, value in raw_sources.items()
        } if isinstance(raw_sources, Mapping) else {}
        return cls(
            index_version=int(data.get("index_version", 0)),
            embedding_fingerprint=str(data.get("embedding_fingerprint", "")),
            sources=sources,
        )


__all__ = ["SourceRecord", "SourceManifest"]
