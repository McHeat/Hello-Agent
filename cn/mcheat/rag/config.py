"""配置管理：JSON 默认值 → 用户配置 → 环境变量 → 代码覆盖，逐级合并 + 校验。

配置优先级（后者覆盖前者）::

    configs/defaults.json  <  用户 JSON  <  RAG_* 环境变量  <  overrides 字典

所有可调参数集中在一处，代码里不出现散落的魔法数字。每次 ``RAGConfig``
构建都会执行 :meth:`RAGConfig.validate`，把《rag编码注意事项》里的建议区间
（粗召回 30~50、精排 3~5）做成**软告警**：允许偏离以便做消融实验，但会在
日志里明确提示，避免「拍脑袋调参」而不自知（注意事项第 4 条）。
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
        name: 注册表中的名字，如 ``"memory"`` / ``"cross_encoder"``。
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
            raise ConfigError(f"组件 {data['name']!r} 的 params 必须是对象，收到 {type(params).__name__}")
        unknown = {k for k in data if not k.startswith("_")} - {"name", "params"}
        if unknown:
            raise ConfigError(f"组件 {data['name']!r} 含未知键：{sorted(unknown)}")
        return cls(name=str(data["name"]), params=dict(params))


@dataclass
class LoaderConfig:
    """DocumentLoader 的通用选项。"""

    recursive: bool = True                 # 目录加载时是否递归
    encoding: str = "utf-8"               # 首选编码
    fallback_encodings: list[str] = field(default_factory=lambda: ["gb18030", "latin-1"])
    max_file_size_mb: float = 20.0        # 单文件大小上限，防止误读巨型文件


@dataclass
class CleaningConfig:
    """清洗流水线选项。

    Attributes:
        steps: 按顺序执行的 Cleaner 名字。前几个是内置步骤，也可以填
            ``CLEANER_REGISTRY`` 里注册的自定义名字。
        on_error: ``"raise"`` 立即失败；``"skip"`` 丢弃出错文档并继续。
        step_params: 自定义步骤的构造参数，按步骤名索引。
        min_length: 清洗后正文短于该字符数即丢弃。
        boilerplate_patterns: 需要剔除的样板行正则（页码、版权声明等）。
    """

    steps: list[str] = field(default_factory=lambda: [
        "unicode_normalize", "control_chars", "html_strip",
        "whitespace", "boilerplate", "drop_empty", "deduplicate",
    ])
    on_error: str = "raise"
    step_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    min_length: int = 10
    boilerplate_patterns: list[str] = field(default_factory=lambda: [
        r"^\s*第?\s*\d+\s*页(\s*/\s*共\s*\d+\s*页)?\s*$",
        r"^\s*(copyright|©|版权所有|all rights reserved).*$",
    ])


@dataclass
class RetrievalConfig:
    """在线检索的粗召回 / 精排参数。

    ``coarse_top_k`` 与 ``rerank_top_n`` 直接对应《注意事项》第 1 条：
    粗召回 30~50 条，**只把精排后的 3~5 条**放进 prompt。
    """

    coarse_top_k: int = 50          # 融合后保留的粗候选上限（建议 30~50）
    rerank_top_n: int = 5           # 精排后进入 prompt 的条数（建议 3~5）
    dense_k: int = 50               # 稠密通路单路召回条数
    sparse_k: int = 50              # BM25 通路单路召回条数
    min_recall_hits: int = 3        # 低于该数量视为「召回不足」，触发兜底
    min_best_score: float = 0.0     # 最高精排分低于该值触发兜底
    enable_dense: bool = True       # 消融开关：关闭稠密通路
    enable_sparse: bool = True      # 消融开关：关闭稀疏通路
    batch_size: int = 32            # 离线 embedding 批大小
    prune_stale: bool = True        # 重建索引时删除本次未出现的旧块


@dataclass
class FallbackConfig:
    """兜底链路参数（《注意事项》第 5 条）。"""

    enabled: bool = True
    widen_factor: float = 2.0       # 扩 Top-K 重试的倍率
    max_retries: int = 2            # 最多重试次数（防止无限放大）
    max_coarse_top_k: int = 200     # 扩 Top-K 的硬上限
    handoff_enabled: bool = True    # 是否允许「转人工」这一档
    no_answer_text: str = "没有找到相关信息，无法基于现有知识库回答该问题。"
    handoff_text: str = "检索到的资料不足以可靠回答，已转人工处理。"
    strategies: list[str] = field(default_factory=lambda: [
        "widen_top_k", "human_handoff", "honest_no_answer",
    ])


@dataclass
class PromptConfig:
    """把 ``BuiltContext`` 包成最终 prompt 的选项。"""

    system_instruction: str = (
        "你是知识库问答助手。请只依据下面提供的资料回答问题，"
        "并在句末用 [编号] 标注引用。资料中没有的信息不要编造；"
        "若资料不足，请直接说明没有找到相关信息。"
    )
    repeat_question_at_tail: bool = True  # 末尾重复问题，强化近因效应
    context_header: str = "以下是检索到的资料（按重要性排列，最关键内容位于开头与结尾）："
    question_template: str = "问题：{query}"


@dataclass
class EvaluationConfig:
    """Golden Set 回归评测选项（《注意事项》第 4 条）。"""

    id_field: str = "source"        # 用哪个字段判定「相关」：source/parent_id/doc_id/chunk_id
    recall_stage: str = "coarse"    # Context Recall 在粗召回阶段计算
    precision_stage: str = "final"  # Context Precision 在精排后（真正进 prompt）计算
    max_regression_drop: float = 0.02  # 相对基线允许的最大指标跌幅


# ---------------------------------------------------------------------------
# 顶层配置
# ---------------------------------------------------------------------------


@dataclass
class RAGConfig:
    """整个框架的配置根对象。"""

    version: int = 1
    components: dict[str, ComponentSpec] = field(default_factory=dict)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RAGConfig":
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
            loader=_build(LoaderConfig, data.get("loader", {})),
            cleaning=_build(CleaningConfig, data.get("cleaning", {})),
            retrieval=_build(RetrievalConfig, data.get("retrieval", {})),
            fallback=_build(FallbackConfig, data.get("fallback", {})),
            prompt=_build(PromptConfig, data.get("prompt", {})),
            evaluation=_build(EvaluationConfig, data.get("evaluation", {})),
        )

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
        use_env: bool = True,
    ) -> "RAGConfig":
        """按优先级加载配置并完成校验。

        Args:
            path: 用户 JSON 配置路径；``None`` 表示只用内置默认值。
            overrides: 代码级覆盖（字典，会深度合并）。
            use_env: 是否读取 ``RAG_*`` 环境变量。

        Returns:
            已校验的 :class:`RAGConfig`。

        Raises:
            ConfigError: 文件不存在、JSON 非法、或校验不通过。
        """
        data = _read_json(DEFAULT_CONFIG_PATH)
        default_names = {
            key: value.get("name")
            for key, value in data.get("components", {}).items()
            if isinstance(value, dict)
        }
        patches: list[Mapping[str, Any]] = []

        if path is not None:
            user_path = Path(path)
            if not user_path.is_file():
                raise ConfigError(f"用户配置文件不存在：{user_path}")
            user_patch = _read_json(user_path)
            patches.append(user_patch)
            data = _deep_merge(data, user_patch)
        if overrides:
            override_patch = dict(overrides)
            patches.append(override_patch)
            data = _deep_merge(data, override_patch)
        if use_env:
            # 环境变量只改组件名 / 标量，也视为一次 patch 参与下面的参数对账
            patches.append(_env_component_patch())
            data = _apply_env_overrides(data)

        _reconcile_component_params(data, default_names, patches)
        config = cls.from_dict(data)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        """导出为普通 dict（可用于 ``save`` 或与基线一起归档）。"""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """把当前配置写入 JSON（与索引产物一起归档，保证可复现）。"""
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
        """执行硬约束校验 + 对「注意事项」建议区间做软告警。

        Raises:
            ConfigError: 任一硬约束不满足。
        """
        missing = REQUIRED_COMPONENTS - set(self.components)
        if missing:
            raise ConfigError(f"配置缺少必需组件：{sorted(missing)}")

        r = self.retrieval
        f = self.fallback
        _positive("retrieval.coarse_top_k", r.coarse_top_k)
        _positive("retrieval.rerank_top_n", r.rerank_top_n)
        _positive("retrieval.dense_k", r.dense_k)
        _positive("retrieval.sparse_k", r.sparse_k)
        _positive("retrieval.batch_size", r.batch_size)
        if r.rerank_top_n > r.coarse_top_k:
            raise ConfigError(
                f"rerank_top_n({r.rerank_top_n}) 不能大于 coarse_top_k"
                f"({r.coarse_top_k})：精排无法凭空产出更多候选。"
            )
        if not (r.enable_dense or r.enable_sparse):
            raise ConfigError("粗召回两条通路都被关闭，检索永远无结果；至少启用一条。")
        if not 0.0 <= r.min_best_score <= 1.0:
            raise ConfigError("retrieval.min_best_score 需落在 [0, 1]。")

        # 父子切分不变量（注意事项第 2 条）
        sp = self.component("splitter").params
        parent_size = int(sp.get("parent_size", 1200))
        child_size = int(sp.get("child_size", 240))
        overlap = int(sp.get("child_overlap", 48))
        if child_size <= 0 or parent_size <= 0:
            raise ConfigError("parent_size / child_size 必须为正整数。")
        if child_size > parent_size:
            raise ConfigError(
                f"child_size({child_size}) 不能大于 parent_size({parent_size})："
                f"子块必须是父块的子串，Small-to-Big 才成立。"
            )
        if not 0 <= overlap < child_size:
            raise ConfigError("child_overlap 需满足 0 <= overlap < child_size。")

        # 上下文编排
        cb = self.component("context_builder").params
        placement = str(cb.get("placement", "edges"))
        if placement not in {"edges", "sequential", "head", "tail"}:
            raise ConfigError(
                f"未知的上下文位置策略 {placement!r}；可选 edges/sequential/head/tail。"
            )
        if int(cb.get("max_context_chars", 4000)) <= 0:
            raise ConfigError("max_context_chars 必须为正整数。")

        if self.cleaning.on_error not in {"raise", "skip"}:
            raise ConfigError("cleaning.on_error 只能是 'raise' 或 'skip'。")
        if f.max_retries < 0:
            raise ConfigError("fallback.max_retries 不能为负。")
        if f.widen_factor <= 1.0:
            raise ConfigError("fallback.widen_factor 必须大于 1，否则扩 Top-K 没有意义。")
        if f.max_coarse_top_k < r.coarse_top_k:
            raise ConfigError("fallback.max_coarse_top_k 不能小于 retrieval.coarse_top_k。")
        if self.evaluation.recall_stage not in {"coarse", "final"} or \
                self.evaluation.precision_stage not in {"coarse", "final"}:
            raise ConfigError("评测阶段只能是 'coarse' 或 'final'。")

        # 软告警：把文档里的经验值显式写进代码，调参时不会被无声忽略
        _warn_range("coarse_top_k", r.coarse_top_k, 30, 50,
                    "《注意事项》第 1 条：混合召回建议 Top-30~50")
        _warn_range("rerank_top_n", r.rerank_top_n, 3, 5,
                    "《注意事项》第 1 条：精排后建议只把 Top-3~5 放进 prompt")


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
        raise ConfigError(f"{cls.__name__} 含未知配置键：{sorted(unknown)}；可用：{sorted(names)}")
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


# 会把某个组件「换实现」的环境变量
_ENV_COMPONENT_NAMES: dict[str, str] = {
    "RAG_RERANKER": "reranker",
    "RAG_EMBEDDER": "embedder",
    "RAG_VECTOR_STORE": "vector_store",
}


def _env_component_patch() -> dict[str, Any]:
    """收集「仅通过环境变量切换组件名」的 patch（用于参数对账）。"""
    components = {
        key: {"name": os.getenv(env_name)}
        for env_name, key in _ENV_COMPONENT_NAMES.items()
        if os.getenv(env_name)
    }
    return {"components": components} if components else {}


def _reconcile_component_params(
    data: dict[str, Any],
    default_names: dict[str, Any],
    patches: list[Mapping[str, Any]],
) -> None:
    """换实现时**不继承旧实现的 params**。

    深度合并会把默认实现的参数（如 ``lexical`` 精排器的
    ``coverage_weight``）残留给新实现（如 ``noop``），导致构造期
    ``TypeError``。组件规格在语义上是原子的：只调 ``name`` 时应当丢弃
    不兼容的旧参数；调用方为新实现显式写的 ``params`` 则完整保留。

    Args:
        data: 已合并、将被就地修正的配置字典。
        default_names: 内置默认配置里各组件的名字。
        patches: 用户提供过的各级 patch（文件 / 代码 / 环境变量）。
    """
    explicit: dict[str, dict[str, Any]] = {}
    for patch in patches:
        raw_components = patch.get("components")
        if not isinstance(raw_components, Mapping):
            continue
        for key, spec in raw_components.items():
            if isinstance(spec, Mapping) and isinstance(spec.get("params"), Mapping):
                explicit.setdefault(str(key), {})
                explicit[str(key)] = _deep_merge(explicit[str(key)], spec["params"])

    components = data.get("components")
    if not isinstance(components, dict):
        return
    for key, default_name in default_names.items():
        spec = components.get(key)
        if isinstance(spec, dict) and spec.get("name") != default_name:
            spec["params"] = dict(explicit.get(key, {}))
            logger.debug("组件 %s 切换实现为 %r，重置 params 为 %s",
                         key, spec.get("name"), spec["params"])


# 只暴露少量高频参数给环境变量；全部参数请用 JSON 配置，避免隐式魔法
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "RAG_COARSE_TOP_K": ("retrieval", "coarse_top_k"),
    "RAG_RERANK_TOP_N": ("retrieval", "rerank_top_n"),
    "RAG_MIN_RECALL_HITS": ("retrieval", "min_recall_hits"),
    "RAG_MIN_BEST_SCORE": ("retrieval", "min_best_score"),
    "RAG_FALLBACK_ENABLED": ("fallback", "enabled"),
    "RAG_MAX_RETRIES": ("fallback", "max_retries"),
    "RAG_RERANKER": ("components", "reranker", "name"),
    "RAG_EMBEDDER": ("components", "embedder", "name"),
    "RAG_VECTOR_STORE": ("components", "vector_store", "name"),
    "RAG_CONTEXT_PLACEMENT": ("components", "context_builder", "params", "placement"),
    "RAG_MAX_CONTEXT_CHARS": ("components", "context_builder", "params", "max_context_chars"),
    "RAG_PARENT_SIZE": ("components", "splitter", "params", "parent_size"),
    "RAG_CHILD_SIZE": ("components", "splitter", "params", "child_size"),
    "RAG_CHILD_OVERLAP": ("components", "splitter", "params", "child_overlap"),
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
    """把 ``RAG_*`` 环境变量写入配置字典；未设置的变量不影响结果。"""
    for env_name, path in ENV_OVERRIDES.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        current = _get_path(data, path)
        value = _coerce_env(raw, current)
        _set_path(data, path, value)
        logger.debug("环境变量覆盖 %s -> %s = %r", env_name, ".".join(path), value)

    level = os.getenv("RAG_LOG_LEVEL")
    if level:
        numeric = getattr(logging, level.strip().upper(), None)
        if isinstance(numeric, int):
            logging.getLogger("cn.mcheat.rag").setLevel(numeric)
        else:
            logger.warning("忽略非法的 RAG_LOG_LEVEL=%r", level)
    return data


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ConfigError(f"{name} 必须为正整数，当前为 {value}")


def _warn_range(name: str, value: int, low: int, high: int, reason: str) -> None:
    if not (low <= value <= high):
        logger.warning(
            "%s=%d 偏离建议区间 [%d, %d]（%s）。允许用于消融实验，"
            "但请配合 Golden Set 回归评测确认收益。",
            name, value, low, high, reason,
        )
