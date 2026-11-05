"""组件注册表：把「配置里的字符串」解析成「实现了接口的对象」。

这是可插拔设计的运行时枢纽。用法::

    from cn.mcheat.rag.core.registry import VECTOR_STORE_REGISTRY

    @VECTOR_STORE_REGISTRY.register("milvus")
    class MilvusVectorStore(VectorStore):
        ...

    # 配置 {"components": {"vector_store": {"name": "milvus", "params": {...}}}}
    store = VECTOR_STORE_REGISTRY.create("milvus", uri="...")

注册表会在 ``create`` 后做 ``isinstance`` 校验：即使绕过继承、用鸭子类型
实现了同名方法，也会被拦下并给出明确的契约提示，避免线上静默出错。
"""
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

from ..exceptions import ConfigError
from .._utils import get_logger
from .interfaces import (
    Cleaner,
    ContextBuilder,
    Embedder,
    FallbackStrategy,
    FusionStrategy,
    ParentChildSplitter,
    ParentStore,
    Reranker,
    RecallRetriever,
    VectorStore,
)

T = TypeVar("T")
Factory = Callable[..., T]
logger = get_logger(__name__)


class Registry(Generic[T]):
    """具名组件注册表（线程不安全的轻量实现，构建期使用）。

    Args:
        kind: 注册表用途名，仅用于报错信息，如 ``"vector_store"``。
        base: 期望的抽象基类；``create`` 会校验产物类型。
    """

    def __init__(self, kind: str, base: type[T] | None = None) -> None:
        self.kind = kind
        self.base = base
        self._factories: dict[str, Factory[T]] = {}

    def register(
        self, key: str, factory: Factory[T] | None = None
    ) -> Any:
        """注册组件，同时支持装饰器与直接调用两种写法。

        ``@REG.register("memory")`` 装饰类时，工厂就是类本身；
        ``REG.register("memory", lambda **kw: ...)`` 可注册更复杂的构造逻辑。

        Raises:
            ConfigError: 同名重复注册（防止静默覆盖，避免难以排查的行为漂移）。
        """
        if key in self._factories:
            raise ConfigError(
                f"{self.kind} 注册表里已存在 name={key!r}，禁止重复注册；"
                f"如需覆盖请先调用 unregister({key!r})。"
            )

        def _do_register(target: Factory[T]) -> Factory[T]:
            self._factories[key] = target
            logger.debug("注册 %s 组件：%s -> %s", self.kind, key, target)
            return target

        if factory is not None:
            return _do_register(factory)
        return _do_register

    def unregister(self, key: str) -> None:
        """注销组件；不存在时静默返回。"""
        self._factories.pop(key, None)

    def create(self, key: str, **kwargs: Any) -> T:
        """按名字实例化组件。

        Raises:
            ConfigError: 名字未注册、构造抛错、或产物不满足接口契约。
        """
        factory = self._factories.get(key)
        if factory is None:
            raise ConfigError(
                f"未注册的 {self.kind} 组件 name={key!r}。"
                f"可用值：{sorted(self._factories)}。"
                f"检查 configs/defaults.json，或先用 {self.kind} 注册表注册自定义实现。"
            )
        try:
            instance = factory(**kwargs)
        except ConfigError:
            raise
        except Exception as exc:  # 把构造期异常包成统一的可诊断错误
            raise ConfigError(
                f"构造 {self.kind} 组件 {key!r} 失败：{type(exc).__name__}: {exc}"
            ) from exc
        if self.base is not None and not isinstance(instance, self.base):
            raise ConfigError(
                f"{self.kind} 组件 {key!r} 的实例 {type(instance).__name__} "
                f"未实现接口 {self.base.__name__}；请继承对应 ABC。"
            )
        return instance

    def keys(self) -> list[str]:
        """已注册的名字（按字典序）。"""
        return sorted(self._factories)

    def __contains__(self, key: str) -> bool:
        return key in self._factories

    def __len__(self) -> int:
        return len(self._factories)


# --- 全局注册表（实现模块在 core/bootstrap.py 中统一导入完成注册）------------

EMBEDDER_REGISTRY: Registry[Embedder] = Registry("embedder", Embedder)
VECTOR_STORE_REGISTRY: Registry[VectorStore] = Registry("vector_store", VectorStore)
DENSE_RETRIEVER_REGISTRY: Registry[RecallRetriever] = Registry(
    "dense_retriever", RecallRetriever
)
SPARSE_RETRIEVER_REGISTRY: Registry[RecallRetriever] = Registry(
    "sparse_retriever", RecallRetriever
)
SPLITTER_REGISTRY: Registry[ParentChildSplitter] = Registry(
    "splitter", ParentChildSplitter
)
PARENT_STORE_REGISTRY: Registry[ParentStore] = Registry("parent_store", ParentStore)
FUSION_REGISTRY: Registry[FusionStrategy] = Registry("fusion", FusionStrategy)
RERANKER_REGISTRY: Registry[Reranker] = Registry("reranker", Reranker)
CONTEXT_BUILDER_REGISTRY: Registry[ContextBuilder] = Registry(
    "context_builder", ContextBuilder
)

# CLEANER_REGISTRY 的 value 是 Cleaner 类/工厂，配置里按名字开关各清洗步骤
CLEANER_REGISTRY: Registry[Cleaner] = Registry("cleaner", Cleaner)

# 兜底策略按 ``fallback.strategies`` 里的名字顺序组合，不参与 BUILD_ORDER
FALLBACK_STRATEGY_REGISTRY: Registry[FallbackStrategy] = Registry(
    "fallback_strategy", FallbackStrategy
)

ALL_REGISTRIES: dict[str, Registry[Any]] = {
    "embedder": EMBEDDER_REGISTRY,
    "vector_store": VECTOR_STORE_REGISTRY,
    "dense_retriever": DENSE_RETRIEVER_REGISTRY,
    "sparse_retriever": SPARSE_RETRIEVER_REGISTRY,
    "splitter": SPLITTER_REGISTRY,
    "parent_store": PARENT_STORE_REGISTRY,
    "fusion": FUSION_REGISTRY,
    "reranker": RERANKER_REGISTRY,
    "context_builder": CONTEXT_BUILDER_REGISTRY,
}

# 组件构建顺序：被 ``@component`` 引用依赖的必须先构造。
# 见 config.ComponentSpec.params 中的 "@embedder" / "@vector_store" 引用语法。
BUILD_ORDER: tuple[str, ...] = (
    "embedder",
    "vector_store",
    "splitter",
    "parent_store",
    "fusion",
    "reranker",
    "context_builder",
    "dense_retriever",
    "sparse_retriever",
)

REQUIRED_COMPONENTS: frozenset[str] = frozenset(BUILD_ORDER)


def resolve_references(params: dict[str, Any], built: dict[str, Any]) -> dict[str, Any]:
    """把配置参数里 ``"@name"`` 形式的引用替换成已构建好的组件实例。

    这样 JSON 配置也能表达依赖注入，例如::

        "dense_retriever": {
            "name": "embedding",
            "params": {"embedder": "@embedder", "vector_store": "@vector_store"}
        }

    Args:
        params: 原始参数 dict（不会被修改）。
        built: 已构建组件字典。

    Raises:
        ConfigError: 引用了尚未构建 / 不存在的组件。
    """

    def _resolve(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("@"):
            key = value[1:]
            if key not in built:
                raise ConfigError(
                    f"配置参数引用了组件 {value!r}，但它尚未构建。"
                    f"当前已构建：{sorted(built)}；请检查 BUILD_ORDER 或改写配置。"
                )
            return built[key]
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_resolve(v) for v in value]
        return value

    return {k: _resolve(v) for k, v in params.items()}
