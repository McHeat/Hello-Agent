"""离线索引流水线 + 索引产物持久化。

离线阶段（可单独部署为一个批处理任务）::

    sources → LoaderRegistry → CleaningPipeline → ParentChildSplitter
            → ParentStore(父块) + DenseRetriever(子块向量) + SparseRetriever(BM25)
            → 裁剪过期块 → IndexArtifact

设计约束：

* **幂等**：所有 ID 由内容稳定哈希生成，重复建索引走 upsert 而非累加；
  ``prune_stale=True`` 时删除本次未出现的旧块，保证索引与语料一致。
* **先清洗再切分**：清洗会改变文本长度，反过来会让 ``start/end`` 偏移失真，
  破坏 Small-to-Big 的子串不变量。
* **错误可定位**：每个阶段出错都带阶段名与文档来源，不会丢一条裸异常。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .._utils import get_logger
from ..cleaning.pipeline import CleaningPipeline, CleaningReport
from ..config import RAGConfig
from ..core.interfaces import (
    ParentChildSplitter,
    ParentStore,
    RecallRetriever,
    Source,
)
from ..exceptions import IndexingError, PersistenceError, RAGError
from ..loaders.base import LoaderRegistry
from ..models import IndexStats

logger = get_logger(__name__)

ARTIFACT_FILENAME = "index.json"


# ---------------------------------------------------------------------------
# 索引产物
# ---------------------------------------------------------------------------


@dataclass
class IndexArtifact:
    """一次离线构建的完整快照：配置 + 各组件状态 + 统计。

    把**配置一起归档**很重要：换 embedding 模型 / 切分参数后，旧索引与新代码
    不兼容；归档配置能让回归评测明确知道某个基线是用哪套参数产出的。
    """

    version: int = 1
    config: dict[str, Any] = field(default_factory=dict)
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    index_stats: dict[str, Any] = field(default_factory=dict)
    cleaning_report: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        from datetime import datetime, timezone
        return {
            "version": self.version,
            "created_at": self.created_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": self.config,
            "components": self.components,
            "index_stats": self.index_stats,
            "cleaning_report": self.cleaning_report,
        }

    def save(self, directory: str | Path) -> Path:
        """原子写入 ``<directory>/index.json``。"""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / ARTIFACT_FILENAME
        tmp = target.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, target)
        except OSError as exc:
            raise PersistenceError(f"写入索引产物失败：{target}：{exc}") from exc
        return target

    @classmethod
    def load(cls, directory: str | Path) -> "IndexArtifact":
        """读取索引产物。

        Raises:
            PersistenceError: 文件不存在、JSON 非法、版本不兼容。
        """
        target = Path(directory) / ARTIFACT_FILENAME
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PersistenceError(
                f"索引产物不存在：{target}；请先运行离线索引（RAGSystem.index + save）。"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise PersistenceError(f"索引产物损坏：{target}：{exc}") from exc
        if payload.get("version") != 1:
            raise PersistenceError(
                f"索引产物版本 {payload.get('version')!r} 不受支持（当前仅支持 1）"
            )
        return cls(
            version=payload.get("version", 1),
            config=payload.get("config", {}),
            components=payload.get("components", {}),
            index_stats=payload.get("index_stats", {}),
            cleaning_report=payload.get("cleaning_report", {}),
            created_at=payload.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# 离线索引器
# ---------------------------------------------------------------------------


class OfflineIndexer:
    """把一批来源构建成可检索索引。

    所有依赖均为接口/组合对象，由 ``RAGSystem`` 注入，便于单独替换测试。
    """

    def __init__(
        self,
        loader_registry: LoaderRegistry,
        cleaning_pipeline: CleaningPipeline,
        splitter: ParentChildSplitter,
        dense: RecallRetriever,
        sparse: RecallRetriever,
        parent_store: ParentStore,
        config: RAGConfig,
    ) -> None:
        self.loader_registry = loader_registry
        self.cleaning_pipeline = cleaning_pipeline
        self.splitter = splitter
        self.dense = dense
        self.sparse = sparse
        self.parent_store = parent_store
        self.config = config
        self.last_cleaning_report: CleaningReport | None = None
        self.last_stats: IndexStats | None = None

    def build(self, sources: Sequence[Source]) -> IndexStats:
        """执行完整离线构建。

        Args:
            sources: 文件路径 / 目录路径列表（``DirectoryLoader`` 会递归展开）。

        Returns:
            :class:`IndexStats`，含各阶段计数与跳过明细。

        Raises:
            IndexingError: 任一阶段失败（``RAGError`` 子类会原样透传，
                便于调用方针对 ``LoaderError`` 等做细分处理）。
        """
        started = time.perf_counter()
        stats = IndexStats()
        if not sources:
            logger.warning("未提供任何来源，跳过索引构建")
            return stats

        try:
            # --- 1) 加载 ----------------------------------------------------
            raw_documents = self.loader_registry.load_many(sources)
            stats.documents_loaded = len(raw_documents)
            logger.info("阶段 1/5 加载：%d 篇文档", len(raw_documents))

            # --- 2) 清洗（必须在切分之前）----------------------------------
            documents, cleaning_report = self.cleaning_pipeline.run(raw_documents)
            self.last_cleaning_report = cleaning_report
            stats.documents_kept = len(documents)
            stats.documents_dropped = cleaning_report.dropped
            for error in cleaning_report.errors:
                stats.skipped.append({
                    "stage": "cleaning", "source": error["source"],
                    "reason": f"{error['step']}: {error['error']}",
                })
            logger.info("阶段 2/5 清洗：保留 %d / %d 篇", len(documents), len(raw_documents))

            if not documents:
                logger.warning("清洗后无文档，索引保持为空（旧数据将被裁剪）")

            # --- 3) 父子切分 ------------------------------------------------
            split = self.splitter.split(documents)
            stats.parents = len(split.parents)
            stats.children = len(split.children)
            logger.info("阶段 3/5 切分：%d 父块 / %d 子块",
                        len(split.parents), len(split.children))

            # --- 4) 三处写入 ------------------------------------------------
            self.parent_store.put(split.parents)
            if self.config.retrieval.enable_dense:
                stats.vectors_upserted = self.dense.index(split.children)
            if self.config.retrieval.enable_sparse:
                self.sparse.index(split.children)
            logger.info("阶段 4/5 写入：父块库 / 向量库 / BM25 索引")

            # --- 5) 裁剪过期块（保证幂等、语料删除能生效）-------------------
            if self.config.retrieval.prune_stale:
                stats.pruned = self._prune(split)
                logger.info("阶段 5/5 裁剪：删除 %d 个过期 ID", stats.pruned)

        except RAGError:
            raise
        except Exception as exc:
            raise IndexingError(
                f"离线索引构建失败：{type(exc).__name__}: {exc}"
            ) from exc

        stats.duration_ms = (time.perf_counter() - started) * 1000.0
        self.last_stats = stats
        logger.info("索引构建完成：%s", stats.as_dict())
        return stats

    def _prune(self, split) -> int:
        """删除本次构建未产出的旧子块 / 父块 ID。"""
        current_children = {child.id for child in split.children}
        current_parents = {parent.id for parent in split.parents}
        removed = 0
        for retriever in (self.dense, self.sparse):
            stale = retriever.ids() - current_children
            if stale:
                removed += retriever.delete(sorted(stale))
        stale_parents = self.parent_store.ids() - current_parents
        if stale_parents:
            self.parent_store.delete(sorted(stale_parents))
        return removed


__all__: Sequence[str] = ["OfflineIndexer", "IndexArtifact", "ARTIFACT_FILENAME"]
