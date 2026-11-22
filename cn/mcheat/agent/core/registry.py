"""组件注册表：把「配置里的字符串」解析成「实现了接口的对象」。

与 ``cn.mcheat.rag.core.registry`` 同构：``Registry.register`` 同时支持
装饰器与直接调用两种写法，``create`` 后做 ``isinstance`` 契约校验，
``resolve_references`` 支持 ``"@组件名"`` 依赖注入（如 skill 引用 ``@llm``）。
"""
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

from langgraph.checkpoint.base import BaseCheckpointSaver

from .._utils import get_logger
from ..exceptions import ConfigError
from .interfaces import (
    EnvProbe,
    IntentClassifier,
    LLMClient,
    Skill,
    SyntaxKnowledgeBase,
)

T = TypeVar("T")
Factory = Callable[..., T]
logger = get_logger(__name__)


class Registry(Generic[T]):
    """具名组件注册表（线程不安全的轻量实现，构建期使用）。

    Args:
        kind: 注册表用途名，仅用于报错信息，如 ``"llm"``。
        base: 期望的抽象基类；``create`` 会校验产物类型。
    """

    def __init__(self, kind: str, base: type[T] | None = None) -> None:
        self.kind = kind
        self.base = base
        self._factories: dict[str, Factory[T]] = {}

    def register(self, key: str, factory: Factory[T] | None = None) -> Any:
        """注册组件，同时支持装饰器与直接调用两种写法。

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

LLM_REGISTRY: Registry[LLMClient] = Registry("llm", LLMClient)
INTENT_CLASSIFIER_REGISTRY: Registry[IntentClassifier] = Registry(
    "intent_classifier", IntentClassifier
)
KNOWLEDGE_BASE_REGISTRY: Registry[SyntaxKnowledgeBase] = Registry(
    "knowledge_base", SyntaxKnowledgeBase
)
ENV_PROBE_REGISTRY: Registry[EnvProbe] = Registry("env_probe", EnvProbe)
SKILL_REGISTRY: Registry[Skill] = Registry("skill", Skill)
CHECKPOINTER_REGISTRY: Registry[BaseCheckpointSaver] = Registry(
    "checkpointer", BaseCheckpointSaver
)

#: 组件键 -> 注册表。三个技能键共用 SKILL_REGISTRY，
#: 由 defaults.json 中各自的 name（syntax / environment / review）区分实现。
ALL_REGISTRIES: dict[str, Registry[Any]] = {
    "llm": LLM_REGISTRY,
    "intent_classifier": INTENT_CLASSIFIER_REGISTRY,
    "knowledge_base": KNOWLEDGE_BASE_REGISTRY,
    "env_probe": ENV_PROBE_REGISTRY,
    "skill_syntax": SKILL_REGISTRY,
    "skill_env": SKILL_REGISTRY,
    "skill_review": SKILL_REGISTRY,
    "checkpointer": CHECKPOINTER_REGISTRY,
}

# 组件构建顺序：被 ``@component`` 引用依赖的必须先构造。
# 见 config.ComponentSpec.params 中的 "@llm" / "@knowledge_base" 引用语法。
BUILD_ORDER: tuple[str, ...] = (
    "llm",
    "intent_classifier",
    "knowledge_base",
    "env_probe",
    "skill_syntax",
    "skill_env",
    "skill_review",
    "checkpointer",
)

REQUIRED_COMPONENTS: frozenset[str] = frozenset(BUILD_ORDER)


def resolve_references(params: dict[str, Any], built: dict[str, Any]) -> dict[str, Any]:
    """把配置参数里 ``"@name"`` 形式的引用替换成已构建好的组件实例。

    这样 JSON 配置也能表达依赖注入，例如::

        "skill_syntax": {
          "name": "syntax",
          "params": {"llm": "@llm", "knowledge_base": "@knowledge_base"}
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
