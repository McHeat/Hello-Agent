"""RAG 框架的异常体系。

设计原则：

* 所有异常继承 :class:`RAGError`，调用方可以 ``except RAGError`` 一网打尽；
* 按**阶段**分层（配置 / 加载 / 清洗 / 索引 / 检索 / 评测 / 持久化），
  便于上层做差异化重试与告警；
* 报错信息必须包含「发生了什么 + 上下文 + 怎么修」，不允许裸 ``raise``。
"""
from __future__ import annotations


class RAGError(Exception):
    """本框架所有异常的基类。"""


# --- 配置 -------------------------------------------------------------------


class ConfigError(RAGError):
    """配置文件 / 环境变量非法，或组件名未在注册表中注册。"""


class DependencyMissingError(RAGError):
    """可选的第三方依赖未安装。

    核心链路只用标准库；仅在用户主动切换到某个适配器时才需要额外依赖。
    """

    def __init__(self, package: str, feature: str, extra: str = "") -> None:
        self.package = package
        self.feature = feature
        hint = f"请执行：pip install {package}"
        if extra:
            hint += f"（{extra}）"
        super().__init__(
            f"{feature} 需要可选依赖 '{package}'，但当前环境未安装。{hint}；"
            f"或改用零依赖的默认实现（见 configs/defaults.json 的 components）。"
        )


# --- 加载与清洗 --------------------------------------------------------------


class LoaderError(RAGError):
    """文档读取失败（IO 错误、编码错误、格式非法等）。"""


class UnsupportedSourceError(LoaderError):
    """没有找到能处理该文件后缀的 DocumentLoader。"""


class CleaningError(RAGError):
    """清洗流水线中某个 Cleaner 执行失败。"""


# --- 离线索引 ----------------------------------------------------------------


class IndexingError(RAGError):
    """离线建索引过程中的错误。"""


class ParentChunkNotFoundError(IndexingError):
    """子块引用的父块在 ParentStore 中不存在（索引文件不一致 / 被裁剪）。"""


# --- 向量库 ------------------------------------------------------------------


class VectorStoreError(RAGError):
    """向量库读写错误。"""


class DimensionMismatchError(VectorStoreError):
    """写入向量维度与库中已有维度不一致（通常是换了 embedding 模型）。"""


# --- 在线检索 ----------------------------------------------------------------


class RetrievalError(RAGError):
    """在线检索链路错误。"""


class EmptyRecallError(RetrievalError):
    """粗召回两条通路都没有返回任何候选。

    注意：该异常**不直接抛给终端用户**，而是作为触发兜底链路的内部信号，
    见 ``retrieval/fallback.py``。
    """


# --- 持久化 ------------------------------------------------------------------


class PersistenceError(RAGError):
    """索引产物保存 / 加载失败。"""


class PersistenceNotSupportedError(PersistenceError):
    """自定义组件没有实现 ``export_state`` / ``import_state``。"""

    def __init__(self, component: str, operation: str) -> None:
        super().__init__(
            f"组件 {component!r} 未实现 {operation}()，无法随 RAGSystem 一起持久化。"
            f"请实现 Stateful 接口，或通过 RAGSystem(overrides=...) 注入自带持久化的组件。"
        )


# --- 评测 --------------------------------------------------------------------


class EvaluationError(RAGError):
    """Golden Set 格式非法或回归评测执行失败。"""
