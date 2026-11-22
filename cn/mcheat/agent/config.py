"""配置管理：JSON 默认值 → 用户配置 → 环境变量 → 代码覆盖，逐级合并 + 校验。

配置优先级（后者覆盖前者）::

    configs/defaults.json  <  用户 JSON  <  AGENT_* 环境变量  <  overrides 字典

与 ``cn.mcheat.rag.config`` 同构：所有可调参数集中一处，``validate()``
做硬约束校验（不满足直接抛 :class:`ConfigError`）+ 软告警（偏离建议
区间打 WARNING，允许做实验但不会被无声忽略）。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from ._utils import get_logger
from .core.registry import REQUIRED_COMPONENTS
from .exceptions import ConfigError

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "defaults.json"


# ---------------------------------------------------------------------------
# 配置片段
# ---------------------------------------------------------------------------


@dataclass
class ComponentSpec:
    """一个可插拔组件的构造说明。

    Attributes:
        name: 注册表中的名字，如 ``"syntax"``（SyntaxSkill）。
        params: 传给构造函数的关键字参数。值若为 ``"@xxx"`` 字符串，
            会在构建时被替换为已构建好的 ``xxx`` 组件（依赖注入），
            见 ``core.registry.resolve_references``。
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComponentSpec":
        if not isinstance(data, Mapping):
            raise ConfigError(
                f"components 的每一项必须是对象（含 name / params），收到：{data!r}"
            )
        if "name" not in data:
            raise ConfigError(f"components 里的每一项都必须有 'name'，收到：{dict(data)!r}")
        params = data.get("params", {})
        if not isinstance(params, dict):
            raise ConfigError(
                f"组件 {data['name']!r} 的 params 必须是对象，收到 {type(params).__name__}"
            )
        unknown = {k for k in data if not k.startswith("_")} - {"name", "params"}
        if unknown:
            raise ConfigError(f"组件 {data['name']!r} 含未知键：{sorted(unknown)}")
        return cls(name=str(data["name"]), params=dict(params))


@dataclass
class GraphConfig:
    """LangGraph 图执行参数。

    ``max_clarify_turns`` 是 clarify↔intent 循环的熔断阀：超过轮数后
    不再向用户发问，直接如实说明并结束本轮，防止耗尽 recursion_limit。
    """

    recursion_limit: int = 25       # LangGraph 递归上限（一轮对话 ≈ 4~6 个超步）
    max_clarify_turns: int = 2      # 单轮最多向用户发起澄清的次数（>=1）


@dataclass
class IntentConfig:
    """意图分类参数。"""

    system_prompt: str = ""         # 留空用内置默认（core/adapters.py）
    history_window: int = 6         # 分类时带上的最近消息条数


@dataclass
class ClarifyConfig:
    """澄清链路文案与熔断回复。"""

    question_template: str = (
        "我不太确定你想让我做什么。你是想要：\n{candidates}\n"
        "请回复序号（1/2/3），或直接补充说明你的需求。"
    )
    exhausted_reply: str = (
        "抱歉，多轮澄清后仍无法确定你的意图。你可以：\n"
        "1) 换个更具体的问法，例如「列表推导式怎么写」；\n"
        "2) 直接粘贴一段 Python 代码，我会给出修改指导；\n"
        "3) 询问「检测一下我的 Python 环境」，我来检查安装状态。"
    )


@dataclass
class EnvProbeConfig:
    """环境探针参数（对应 ``env_probe`` 组件的默认构造参数）。"""

    timeout: float = 10.0           # subprocess 超时（秒），防止 pip 挂起阻塞图
    include_pip_list: bool = True   # 是否执行 pip list（较慢，可关）
    pip_list_limit: int = 30        # pip list 截断行数，防 token 膨胀


# ---------------------------------------------------------------------------
# 顶层配置
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """整个框架的配置根对象。"""

    version: int = 1
    components: dict[str, ComponentSpec] = field(default_factory=dict)
    graph: GraphConfig = field(default_factory=GraphConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    clarify: ClarifyConfig = field(default_factory=ClarifyConfig)
    env_probe: EnvProbeConfig = field(default_factory=EnvProbeConfig)

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentConfig":
        """从「已合并完成」的配置字典构造，缺失字段保留 dataclass 默认值。

        Raises:
            ConfigError: 出现未知配置键或类型明显不符。
        """
        unknown = _visible_keys(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"配置项含未知顶层键：{sorted(unknown)}")

        raw_components = data.get("components", {})
        if not isinstance(raw_components, dict):
            raise ConfigError("'components' 必须是对象")
        components = {
            str(key): ComponentSpec.from_dict(value)
            for key, value in raw_components.items()
        }
        return cls(
            version=int(data.get("version", 1)),
            components=components,
            graph=_build(GraphConfig, data.get("graph", {})),
            intent=_build(IntentConfig, data.get("intent", {})),
            clarify=_build(ClarifyConfig, data.get("clarify", {})),
            env_probe=_build(EnvProbeConfig, data.get("env_probe", {})),
        )

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
        use_env: bool = True,
    ) -> "AgentConfig":
        """按优先级加载配置并完成校验。

        Args:
            path: 用户 JSON 配置路径；``None`` 表示只用内置默认值。
            overrides: 代码级覆盖（字典，会深度合并）。
            use_env: 是否读取 ``AGENT_*`` 环境变量。

        Returns:
            已校验的 :class:`AgentConfig`。

        Raises:
            ConfigError: 文件不存在、JSON 非法、或校验不通过。
        """
        data = _read_json(DEFAULT_CONFIG_PATH)
        if path is not None:
            user_path = Path(path)
            if not user_path.is_file():
                raise ConfigError(f"用户配置文件不存在：{user_path}")
            data = _deep_merge(data, _read_json(user_path))
        if overrides:
            data = _deep_merge(data, dict(overrides))
        if use_env:
            data = _apply_env_overrides(data)
        config = cls.from_dict(data)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        """导出为普通 dict（可用于归档与复现）。"""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """把当前配置写入 JSON。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def component(self, key: str) -> ComponentSpec:
        """取某个组件规格；不存在时给出可操作的报错。"""
        try:
            return self.components[key]
        except KeyError as exc:
            raise ConfigError(
                f"配置缺少组件 {key!r}；必须包含：{sorted(REQUIRED_COMPONENTS)}"
            ) from exc

    # -- 校验 ---------------------------------------------------------------

    def validate(self) -> None:
        """硬约束校验 + 建议区间软告警。

        Raises:
            ConfigError: 任一硬约束不满足。
        """
        missing = REQUIRED_COMPONENTS - set(self.components)
        if missing:
            raise ConfigError(f"配置缺少必需组件：{sorted(missing)}")

        g = self.graph
        if g.max_clarify_turns < 1:
            raise ConfigError("graph.max_clarify_turns 必须 >= 1（0 轮澄清等于禁用 human-in-the-loop）")
        if not 8 <= g.recursion_limit <= 100:
            raise ConfigError("graph.recursion_limit 需落在 [8, 100]。")
        # 每轮澄清大约消耗 clarify -> intent -> skill -> answer 4 个超步，
        # 预算不足会在运行期触发 GraphRecursionError，这里提前拦下。
        if g.recursion_limit < 4 * (g.max_clarify_turns + 1):
            raise ConfigError(
                f"graph.recursion_limit({g.recursion_limit}) 不足以支撑 "
                f"max_clarify_turns({g.max_clarify_turns}) 的澄清循环；"
                f"至少需要 {4 * (g.max_clarify_turns + 1)}。"
            )

        if self.intent.history_window < 0:
            raise ConfigError("intent.history_window 不能为负。")

        if not self.clarify.question_template or "{candidates}" not in self.clarify.question_template:
            raise ConfigError("clarify.question_template 必须包含 {candidates} 占位符。")
        if not self.clarify.exhausted_reply:
            raise ConfigError("clarify.exhausted_reply 不能为空。")

        e = self.env_probe
        if e.timeout <= 0:
            raise ConfigError("env_probe.timeout 必须为正数（秒）。")
        if not 0 <= e.pip_list_limit <= 200:
            raise ConfigError("env_probe.pip_list_limit 需落在 [0, 200]。")

        # 软告警：偏离经验值时提示，但不阻断（允许做实验）
        if g.max_clarify_turns > 3:
            logger.warning(
                "graph.max_clarify_turns=%d 超过建议值 3：澄清轮数过多会让用户疲劳，"
                "请确认确实需要。", g.max_clarify_turns,
            )
        if e.pip_list_limit > 50:
            logger.warning(
                "env_probe.pip_list_limit=%d 偏大：pip list 全量输出会显著增加 token 消耗。",
                e.pip_list_limit,
            )


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _build(cls: type, data: Any):
    """用 dict 覆盖 dataclass 默认值，并拒绝未知键。"""
    if not isinstance(data, Mapping):
        raise ConfigError(f"'{cls.__name__}' 配置必须是对象，收到 {type(data).__name__}")
    names = {f.name for f in fields(cls)}
    unknown = _visible_keys(data) - names
    if unknown:
        raise ConfigError(
            f"{cls.__name__} 含未知配置键：{sorted(unknown)}；可用：{sorted(names)}"
        )
    return cls(**{k: v for k, v in data.items() if k in names})


def _visible_keys(data: Mapping[str, Any]) -> set[str]:
    """返回非注释键（``_`` 前缀视为 JSON 注释，因为 JSON 不支持原生注释）。"""
    return {key for key in data if not str(key).startswith("_")}


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 配置，附带有用的上下文信息。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"配置文件不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"配置文件 {path} 不是合法 JSON：第 {exc.lineno} 行 {exc.msg}") from exc
    except OSError as exc:
        raise ConfigError(f"读取配置文件 {path} 失败：{exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"配置文件 {path} 的顶层必须是对象")
    return payload


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并字典；``override`` 中的标量/列表直接替换，dict 继续下钻。"""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# 只暴露少量高频参数给环境变量；全部参数请用 JSON 配置，避免隐式魔法。
# 说明：LLM 依赖运行期注入的模型实例，刻意不提供环境变量切换。
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "AGENT_RECURSION_LIMIT": ("graph", "recursion_limit"),
    "AGENT_MAX_CLARIFY_TURNS": ("graph", "max_clarify_turns"),
    "AGENT_HISTORY_WINDOW": ("intent", "history_window"),
    "AGENT_ENV_TIMEOUT": ("env_probe", "timeout"),
    "AGENT_PIP_LIST_LIMIT": ("env_probe", "pip_list_limit"),
    "AGENT_ENV_PROBE": ("components", "env_probe", "name"),
    "AGENT_CHECKPOINTER": ("components", "checkpointer", "name"),
}


def _get_path(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = data
    for part in path[:-1]:
        current = current.setdefault(part, {})
        if not isinstance(current, dict):
            raise ConfigError(f"环境变量路径 {'.'.join(path)} 与配置结构冲突")
    current[path[-1]] = value


def _coerce_env(raw: str, current: Any) -> Any:
    """按当前默认值的类型解析环境变量字符串。"""
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"环境变量值 {raw!r} 需要是整数") from exc
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"环境变量值 {raw!r} 需要是浮点数") from exc
    return raw


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """把 ``AGENT_*`` 环境变量写入配置字典；未设置的变量不影响结果。

    注意：``AGENT_ENV_PROBE`` / ``AGENT_CHECKPOINTER`` 只切组件名时，
    新实现的 params 用 defaults 里的原值（内置实现的参数签名稳定）；
    自定义实现请在 JSON / overrides 里完整声明。
    """
    for env_name, path in ENV_OVERRIDES.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        current = _get_path(data, path)
        value = _coerce_env(raw, current)
        _set_path(data, path, value)
        logger.debug("环境变量覆盖 %s -> %s = %r", env_name, ".".join(path), value)

    level = os.getenv("AGENT_LOG_LEVEL")
    if level:
        numeric = getattr(logging, level.strip().upper(), None)
        if isinstance(numeric, int):
            logging.getLogger("cn.mcheat.agent").setLevel(numeric)
        else:
            logger.warning("忽略非法的 AGENT_LOG_LEVEL=%r", level)
    return data
