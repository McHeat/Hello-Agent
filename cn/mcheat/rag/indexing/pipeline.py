"""离线索引流水线 + 索引产物持久化。

离线阶段（可单独部署为一个批处理任务）::

    sources → LoaderRegistry → CleaningPipeline → ParentChildSplitter
            → ParentStore(父块) + DenseRetriever(子块向量) + SparseRetriever(BM25)
            → 裁剪过期块（仅全量模式）→ IndexArtifact

设计约束：

* **幂等**：所有 ID 由内容稳定哈希生成，重复建索引走 upsert 而非累加；
  全量模式 ``prune_stale=True`` 时删除本次未出现的旧块；增量模式
  ``mode="auto"`` 基于 ``SourceManifest`` 只重建变更来源。
* **单次落盘**：构建期关闭 ``auto_save``，结束时统一 ``flush()`` 一次，
  把近似 O(N²) 的全量 JSON 重写降为 O(N)。
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

from .._utils import content_hash, get_logger
from ..cleaning.pipeline import CleaningPipeline, CleaningReport
from ..config import RAGConfig
from ..core.interfaces import (
    ParentChildSplitter,
    ParentStore,
    RecallRetriever,
    Source,
)
from ..exceptions import ConfigError, IndexingError, PersistenceError, RAGError
from ..loaders.base import LoaderRegistry
from ..models import Document, IndexStats, SplitResult
from .manifest import SourceManifest, SourceRecord

logger = get_logger(__name__)

ARTIFACT_FILENAME = "index.json"


# ---------------------------------------------------------------------------
# 索引产物
# ---------------------------------------------------------------------------


@dataclass
class IndexArtifact:
    """一次离线构建的完整快照：配置 + 各组件状态 + 统计 + 指纹 / 版本 / 清单。

    把**配置一起归档**很重要：换 embedding 模型 / 切分参数后，旧索引与新代码
    不兼容；归档配置能让回归评测明确知道某个基线是用哪套参数产出的。
    旧产物缺少 ``embedding_fingerprint`` / ``index_version`` / ``manifest``
    时**仅 WARN 不报错**，保证向后兼容。
    """

    version: int = 1
    config: dict[str, Any] = field(default_factory=dict)
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    index_stats: dict[str, Any] = field(default_factory=dict)
    cleaning_report: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    embedding_fingerprint: str = ""   # 同维度换模型也能拦截
    index_version: int = 0            # 缓存失效键
    manifest: dict[str, Any] = field(default_factory=dict)  # 旧产物缺失时仅 WARN

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
            "embedding_fingerprint": self.embedding_fingerprint,
            "index_version": self.index_version,
            "manifest": self.manifest,
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
        """读取索引产物；旧产物缺新字段仅 WARN。

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
        missing = [key for key in
                   ("embedding_fingerprint", "index_version", "manifest")
                   if key not in payload]
        if missing:
            logger.warning(
                "索引产物缺少字段 %s（旧格式产物）：已按兼容模式加载，"
                "无法校验模型指纹 / 缓存版本 / 增量清单；建议用当前配置重建索引。",
                missing,
            )
        return cls(
            version=payload.get("version", 1),
            config=payload.get("config", {}),
            components=payload.get("components", {}),
            index_stats=payload.get("index_stats", {}),
            cleaning_report=payload.get("cleaning_report", {}),
            created_at=payload.get("created_at", ""),
            embedding_fingerprint=str(payload.get("embedding_fingerprint", "")),
            index_version=int(payload.get("index_version", 0)),
            manifest=dict(payload.get("manifest", {})),
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
        self.index_version = 0
        self.manifest: SourceManifest | None = None

    def build(self, sources: Sequence[Source],
              mode: str | None = None) -> IndexStats:
        """执行离线构建（全量 / 增量）。

        Args:
            sources: 文件路径 / 目录路径列表（``DirectoryLoader`` 会递归展开）。
            mode: ``None`` 时取 ``indexing.mode``；``auto`` 表示无 manifest
                时全量、有则增量；``full`` 强制全量；``incremental`` 强制增量。

        Returns:
            :class:`IndexStats`，含各阶段计数、模式、版本与增量明细。

        Raises:
            IndexingError: 任一阶段失败（``RAGError`` 子类会原样透传）。
        """
        started = time.perf_counter()
        stats = IndexStats()
        if not sources:
            logger.warning("未提供任何来源，跳过索引构建")
            return stats

        try:
            raw_documents = self.loader_registry.load_many(sources)
            stats.documents_loaded = len(raw_documents)
            logger.info("阶段 1/5 加载：%d 篇文档", len(raw_documents))
            current_hashes = self._source_hashes(raw_documents)
            effective_mode = self._effective_mode(mode)
            stats.mode = effective_mode

            if effective_mode == "incremental":
                self._build_incremental(raw_documents, current_hashes, stats)
            else:
                self._build_full(raw_documents, current_hashes, stats)

            self.index_version += 1
            if self.manifest is not None:
                self.manifest.index_version = self.index_version
            stats.index_version = self.index_version
            self._flush_once()
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

    # -- 模式与来源哈希 -----------------------------------------------------

    def _effective_mode(self, mode: str | None) -> str:
        requested = mode or self.config.indexing.mode
        if requested not in {"auto", "full", "incremental"}:
            raise ConfigError(
                f"未知的索引模式 {requested!r}；可选 auto/full/incremental"
            )
        if requested == "auto":
            if self.manifest is None or not self.manifest.sources:
                return "full"
            # 指纹校验：同维度换模型也不能在旧向量上做增量，必须全量重建
            saved = self.manifest.embedding_fingerprint
            current = self._fingerprint()
            if saved and current and saved != current:
                logger.warning(
                    "embedding 指纹不一致（%s → %s）：旧向量与新模型语义空间不同，"
                    "强制按全量模式重建索引并清除增量清单。", saved, current,
                )
                return "full"
            return "incremental"
        return requested

    @staticmethod
    def _source_hashes(documents: Sequence[Document]) -> dict[str, str]:
        """按文档来源（文件路径）分组算内容哈希，用于变更检测。"""
        grouped: dict[str, list[str]] = {}
        for document in documents:
            source = str(document.metadata.get("source", document.id))
            grouped.setdefault(source, []).append(document.page_content)
        return {
            source: content_hash("\x1f".join(sorted(texts)))
            for source, texts in grouped.items()
        }

    @staticmethod
    def _documents_by_source(
        documents: Sequence[Document],
    ) -> dict[str, list[Document]]:
        grouped: dict[str, list[Document]] = {}
        for document in documents:
            source = str(document.metadata.get("source", document.id))
            grouped.setdefault(source, []).append(document)
        return grouped

    def _fingerprint(self) -> str:
        embedder = getattr(self.dense, "embedder", None)
        if embedder is not None and hasattr(embedder, "fingerprint"):
            return str(embedder.fingerprint())
        return ""

    @staticmethod
    def _records(split: SplitResult, hashes: dict[str, str]) -> dict[str, SourceRecord]:
        """把切分结果按来源归集成 manifest 记录。"""
        records: dict[str, SourceRecord] = {}
        for source in sorted(hashes):
            records[source] = SourceRecord(
                content_hash=hashes[source],
                child_ids=sorted(
                    child.id for child in split.children
                    if str(child.metadata.get("source", "")) == source
                ),
                parent_ids=sorted(
                    parent.id for parent in split.parents
                    if str(parent.metadata.get("source", "")) == source
                ),
            )
        return records

    # -- 全量 / 增量 ---------------------------------------------------------

    def _build_full(self, raw_documents: Sequence[Document],
                    hashes: dict[str, str], stats: IndexStats) -> None:
        documents, cleaning_report = self.cleaning_pipeline.run(raw_documents)
        self.last_cleaning_report = cleaning_report
        stats.documents_kept = len(documents)
        stats.documents_dropped = cleaning_report.dropped
        self._collect_cleaning_errors(cleaning_report, stats)
        logger.info("阶段 2/5 清洗：保留 %d / %d 篇", len(documents),
                    len(raw_documents))
        if not documents:
            logger.warning("清洗后无文档，索引保持为空（旧数据将被裁剪）")

        split = self.splitter.split(documents)
        stats.parents = len(split.parents)
        stats.children = len(split.children)
        logger.info("阶段 3/5 切分：%d 父块 / %d 子块",
                    len(split.parents), len(split.children))

        self.parent_store.put(split.parents)
        if self.config.retrieval.enable_dense:
            stats.vectors_upserted = self.dense.index(split.children)
        if self.config.retrieval.enable_sparse:
            self.sparse.index(split.children)
        logger.info("阶段 4/5 写入：父块库 / 向量库 / BM25 索引")

        if self.config.retrieval.prune_stale:
            stats.pruned = self._prune(split)
            logger.info("阶段 5/5 裁剪：删除 %d 个过期 ID", stats.pruned)

        self.manifest = SourceManifest(
            index_version=self.index_version,
            embedding_fingerprint=self._fingerprint(),
            sources=self._records(split, hashes),
        )

    def _build_incremental(self, raw_documents: Sequence[Document],
                           hashes: dict[str, str],
                           stats: IndexStats) -> None:
        manifest = self.manifest or SourceManifest()
        changed = manifest.changed_sources(hashes)
        stats.changed_sources = len(changed)
        by_source = self._documents_by_source(raw_documents)
        changed_present = [source for source in changed if source in by_source]
        selected = [doc for source in changed_present
                    for doc in by_source[source]]
        unchanged_count = len(raw_documents) - len(selected)

        documents, cleaning_report = self.cleaning_pipeline.run(selected)
        self.last_cleaning_report = cleaning_report
        stats.documents_kept = len(documents) + unchanged_count
        stats.documents_dropped = len(raw_documents) - stats.documents_kept
        self._collect_cleaning_errors(cleaning_report, stats)
        logger.info("增量构建：变更来源 %d 个，重切分 %d 篇文档",
                    len(changed), len(selected))

        old_child_ids: set[str] = set()
        old_parent_ids: set[str] = set()
        for source in changed:
            record = manifest.sources.get(source)
            if record is None:
                continue
            old_child_ids.update(record.child_ids)
            old_parent_ids.update(record.parent_ids)
        removed = 0
        for retriever in (self.dense, self.sparse):
            if old_child_ids:
                removed += retriever.delete(sorted(old_child_ids))
        if old_parent_ids:
            self.parent_store.delete(sorted(old_parent_ids))
        stats.deleted_chunks = removed

        split = self.splitter.split(documents)
        self.parent_store.put(split.parents)
        if self.config.retrieval.enable_dense:
            stats.vectors_upserted = self.dense.index(split.children)
        if self.config.retrieval.enable_sparse:
            self.sparse.index(split.children)

        new_sources = dict(manifest.sources)
        for source in changed:
            new_sources.pop(source, None)
        changed_hashes = {source: hashes[source] for source in changed_present}
        new_sources.update(self._records(split, changed_hashes))
        self.manifest = SourceManifest(
            index_version=self.index_version,
            embedding_fingerprint=self._fingerprint(),
            sources=new_sources,
        )

        stats.parents = self.parent_store.count()
        stats.children = (len(self.dense.ids()) if self.config.retrieval.enable_dense
                          else len(self.sparse.ids()))
        stats.pruned = 0  # prune_stale 只在全量模式生效

    @staticmethod
    def _collect_cleaning_errors(report: CleaningReport,
                                 stats: IndexStats) -> None:
        for error in report.errors:
            stats.skipped.append({
                "stage": "cleaning", "source": error["source"],
                "reason": f"{error['step']}: {error['error']}",
            })

    # -- 落盘与裁剪 ---------------------------------------------------------

    def _flush_once(self) -> None:
        """构建结束统一落盘一次；只对真正实现 ``flush()`` 的组件调用。"""
        seen: set[int] = set()
        for target in (self.dense, self.sparse, self.parent_store):
            for obj in (target, getattr(target, "vector_store", None)):
                if obj is None or id(obj) in seen:
                    continue
                seen.add(id(obj))
                flush = getattr(obj, "flush", None)
                if callable(flush):
                    flush()

    def _prune(self, split: SplitResult) -> int:
        """删除本次构建未产出的旧子块 / 父块 ID（仅全量模式）。"""
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
