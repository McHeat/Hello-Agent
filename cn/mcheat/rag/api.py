""":class:`RAGSystem` —— 整个框架的顶层门面。

把「离线索引」与「在线检索」两个模块封装成四个动作::

    system = RAGSystem()                      # 配置驱动装配所有可插拔组件
    stats  = system.index(["data/"])          # 离线：加载→清洗→切分→建索引
    system.save("./.rag_index")               # 离线：落盘（含配置与统计）
    system = RAGSystem.load("./.rag_index")   # 在线：另一进程直接加载
    result = system.retrieve("问题")           # 在线：召回→精排→上下文→兜底
    report = system.evaluate("golden.json")   # 量化：Recall / Precision 回归

所有组件既可由 ``configs/defaults.json`` 按名字替换，也可用 ``overrides``
直接注入实例，两套方式都经过接口契约校验。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ._utils import get_logger
from .cleaning.pipeline import CleaningPipeline
from .config import RAGConfig
from .core.bootstrap import bootstrap
from .core.interfaces import Source
from .core.registry import (
    ALL_REGISTRIES,
    BUILD_ORDER,
    resolve_references,
)
from .evaluation.golden import GoldenSet
from .evaluation.runner import EvalReport, EvalRunner, RegressionVerdict
from .exceptions import ConfigError, PersistenceError
from .indexing.pipeline import IndexArtifact, OfflineIndexer
from .loaders.base import build_default_loader_registry
from .models import IndexStats, RetrievalResult
from .retrieval.context import SmallToBigExpander
from .retrieval.fallback import FallbackChain
from .retrieval.hybrid import HybridRetriever
from .retrieval.pipeline import OnlineRetriever

logger = get_logger(__name__)


class RAGSystem:
    """配置驱动的 RAG 系统门面。

    Args:
        config: :class:`RAGConfig`；不传则加载内置默认配置（含环境变量覆盖）。
        overrides: 组件实例覆盖字典，键取自 ``BUILD_ORDER``（如
            ``"vector_store"`` / ``"reranker"``）。注入的对象必须实现对应
            ABC，否则抛 :class:`ConfigError`。这是不使用注册表时最直接的
            扩展方式。
    """

    def __init__(self, config: RAGConfig | None = None,
                 overrides: Mapping[str, Any] | None = None) -> None:
        bootstrap()
        if config is None:
            config = RAGConfig.load()
        if not isinstance(config, RAGConfig):
            raise ConfigError("config 必须是 RAGConfig 实例")
        config.validate()
        self.config = config
        self._components = self._build_components(dict(overrides or {}))
        self._wire()

    # -- 装配 ---------------------------------------------------------------

    def _build_components(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """按 BUILD_ORDER 构造组件，解析 ``"@xxx"`` 依赖引用。"""
        unknown = set(overrides) - set(BUILD_ORDER)
        if unknown:
            raise ConfigError(
                f"overrides 含未知组件键 {sorted(unknown)}；可用：{list(BUILD_ORDER)}"
            )
        built: dict[str, Any] = {}
        for key in BUILD_ORDER:
            registry = ALL_REGISTRIES[key]
            if key in overrides:
                instance = overrides[key]
                if registry.base is not None and not isinstance(instance, registry.base):
                    raise ConfigError(
                        f"overrides[{key!r}] 的 {type(instance).__name__} "
                        f"未实现 {registry.base.__name__} 接口"
                    )
                built[key] = instance
                logger.info("组件 %s 由 overrides 注入：%s", key, type(instance).__name__)
                continue

            spec = self.config.component(key)
            params = resolve_references(spec.params, built)
            # 离线 embedding 批大小由 retrieval 配置统一控制
            if key == "dense_retriever" and spec.name == "embedding":
                params.setdefault("batch_size", self.config.retrieval.batch_size)
            built[key] = registry.create(spec.name, **params)
            logger.debug("组件 %s = %s(%s)", key, spec.name,
                         type(built[key]).__name__)
        return built

    def _wire(self) -> None:
        """把叶子组件拼成离线 / 在线两条流水线。"""
        c = self._components
        self.cleaning_pipeline = CleaningPipeline.from_config(self.config.cleaning)
        self.loader_registry = build_default_loader_registry(self.config.loader)

        self.hybrid = HybridRetriever(
            dense=c["dense_retriever"],
            sparse=c["sparse_retriever"],
            fusion=c["fusion"],
            enable_dense=self.config.retrieval.enable_dense,
            enable_sparse=self.config.retrieval.enable_sparse,
        )
        expand_to_parent = bool(
            self.config.component("context_builder").params.get("expand_to_parent", True)
        )
        self.expander = SmallToBigExpander(
            c["parent_store"], enabled=expand_to_parent
        )
        self.fallback_chain = FallbackChain.from_config(self.config.fallback)

        self.indexer = OfflineIndexer(
            loader_registry=self.loader_registry,
            cleaning_pipeline=self.cleaning_pipeline,
            splitter=c["splitter"],
            dense=c["dense_retriever"],
            sparse=c["sparse_retriever"],
            parent_store=c["parent_store"],
            config=self.config,
        )
        self.retriever = OnlineRetriever(
            hybrid=self.hybrid,
            reranker=c["reranker"],
            expander=self.expander,
            context_builder=c["context_builder"],
            fallback_chain=self.fallback_chain,
            config=self.config,
        )

    def component(self, key: str) -> Any:
        """取已装配的组件实例（便于自定义扩展与调试）。"""
        try:
            return self._components[key]
        except KeyError as exc:
            raise ConfigError(f"未知组件 {key!r}；可用：{list(BUILD_ORDER)}") from exc

    # -- 离线索引 -----------------------------------------------------------

    def index(self, sources: Sequence[Source] | Source) -> IndexStats:
        """构建索引。

        Args:
            sources: 单个路径或路径列表；目录会被递归展开，5 种格式自动分发。

        Returns:
            :class:`IndexStats`。
        """
        if isinstance(sources, (str, Path)):
            sources = [sources]
        return self.indexer.build(list(sources))

    # -- 在线检索 -----------------------------------------------------------

    def retrieve(self, query: str,
                 metadata_filter: dict[str, Any] | None = None) -> RetrievalResult:
        """检索并返回完整结果（含 prompt 与兜底结论）。

        调用方必须遵守::

            if result.answer_ready:
                llm.invoke(result.prompt)
            else:
                return result.fallback.message   # 转人工 / 如实说没找到
        """
        return self.retriever.retrieve(query, metadata_filter=metadata_filter)

    # -- 评测 ---------------------------------------------------------------

    def evaluate(self, golden_set: GoldenSet | str | Path) -> EvalReport:
        """在 Golden Set 上跑回归评测。

        Args:
            golden_set: :class:`GoldenSet` 或 JSON 文件路径。

        Returns:
            :class:`EvalReport`（Recall 在粗召回算、Precision 在精排后算）。
        """
        loaded = (
            GoldenSet.from_json(golden_set)
            if isinstance(golden_set, (str, Path)) else golden_set
        )
        return EvalRunner(self, loaded).run()

    def evaluate_with_gate(self, golden_set: GoldenSet | str | Path,
                           baseline_path: str | Path,
                           max_drop: float | None = None) -> tuple[EvalReport, RegressionVerdict]:
        """跑评测并直接与基线比对，返回 ``(报告, 回归结论)``。

        适合放进 CI：``verdict.passed`` 为 ``False`` 时让流水线失败，
        防止一次「优化」悄悄拖垮另一项指标（注意事项第 4 条）。
        """
        report = self.evaluate(golden_set)
        baseline = EvalReport.load(baseline_path)
        threshold = (
            self.config.evaluation.max_regression_drop
            if max_drop is None else max_drop
        )
        return report, report.regression_check(baseline, max_drop=threshold)

    # -- 持久化 -------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """把配置 + 三份索引状态 + 统计整体落盘。

        只持久化 **dense_retriever / sparse_retriever / parent_store**：
        ``vector_store`` 的状态已包含在 dense_retriever 的导出里，其余组件
        由配置重建，避免重复存储。

        Raises:
            PersistenceNotSupportedError: 注入的自定义组件未实现 ``Stateful``。
        """
        c = self._components
        artifact = IndexArtifact(
            config=self.config.to_dict(),
            components={
                "dense_retriever": c["dense_retriever"].export_state(),
                "sparse_retriever": c["sparse_retriever"].export_state(),
                "parent_store": c["parent_store"].export_state(),
            },
            index_stats=(
                self.indexer.last_stats.as_dict()
                if self.indexer.last_stats else {}
            ),
            cleaning_report=(
                self.indexer.last_cleaning_report.as_dict()
                if self.indexer.last_cleaning_report else {}
            ),
        )
        path = artifact.save(directory)
        logger.info("索引产物已保存：%s", path)
        return path

    @classmethod
    def load(cls, directory: str | Path, config: RAGConfig | None = None,
             overrides: Mapping[str, Any] | None = None) -> "RAGSystem":
        """从索引产物恢复系统（在线服务入口）。

        Args:
            directory: :meth:`save` 使用的目录。
            config: 覆盖归档配置；**换 embedding 模型/切分参数时不要这么做**，
                否则向量维度与父子 ID 都可能对不上。
            overrides: 组件实例覆盖，语义同构造函数。
        """
        artifact = IndexArtifact.load(directory)
        effective_config = config or RAGConfig.from_dict(artifact.config)
        effective_config.validate()
        system = cls(config=effective_config, overrides=overrides)

        states = artifact.components
        required = {"dense_retriever", "sparse_retriever", "parent_store"}
        missing = required - set(states)
        if missing:
            raise PersistenceError(
                f"索引产物缺少组件状态：{sorted(missing)}，文件可能已损坏"
            )
        for key in sorted(required):
            system.component(key).import_state(states[key])

        if artifact.index_stats:
            try:
                system.indexer.last_stats = IndexStats(**artifact.index_stats)
            except TypeError as exc:
                logger.warning("索引统计字段不兼容，已忽略：%s", exc)
        logger.info(
            "索引产物已加载：%s（dense=%d, sparse=%d, parent=%d）",
            directory,
            system.component("dense_retriever").count(),
            system.component("sparse_retriever").count(),
            system.component("parent_store").count(),
        )
        return system


__all__: Sequence[str] = ["RAGSystem"]
