"""内置实现自注册（幂等）。

import 本包即完成注册，用户代码可以直接用配置名替换组件。
重复调用安全：已注册时直接返回，不会触发「禁止重复注册」。
"""
from __future__ import annotations

import sqlite3

from .._utils import get_logger
from ..exceptions import DependencyMissingError
from .adapters import LangChainLLMClient, LlmIntentClassifier
from .registry import (
    CHECKPOINTER_REGISTRY,
    ENV_PROBE_REGISTRY,
    INTENT_CLASSIFIER_REGISTRY,
    KNOWLEDGE_BASE_REGISTRY,
    LLM_REGISTRY,
    SKILL_REGISTRY,
)
from ..skills import EnvSkill, ReviewSkill, SyntaxSkill
from ..tools import InMemorySyntaxKnowledgeBase, LocalEnvProbe

logger = get_logger(__name__)

_BOOTSTRAPPED = False


def _sqlite_checkpointer(db_path: str = "agent_checkpoints.db"):
    """SqliteSaver 工厂：懒加载可选依赖，缺包时给出可行动的报错。"""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise DependencyMissingError(
            "使用 sqlite checkpointer 需要 langgraph-checkpoint-sqlite",
            hint="pip install langgraph-checkpoint-sqlite",
        ) from exc
    return SqliteSaver(sqlite3.connect(db_path))


def bootstrap() -> None:
    """注册全部内置实现（幂等，import 包时自动执行一次）。"""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    LLM_REGISTRY.register("langchain", LangChainLLMClient)
    INTENT_CLASSIFIER_REGISTRY.register("llm", LlmIntentClassifier)
    KNOWLEDGE_BASE_REGISTRY.register("memory", InMemorySyntaxKnowledgeBase)
    ENV_PROBE_REGISTRY.register("local", LocalEnvProbe)

    SKILL_REGISTRY.register("syntax", SyntaxSkill)
    SKILL_REGISTRY.register("environment", EnvSkill)
    SKILL_REGISTRY.register("review", ReviewSkill)

    CHECKPOINTER_REGISTRY.register("memory", _memory_checkpointer)
    CHECKPOINTER_REGISTRY.register("sqlite", _sqlite_checkpointer)

    _BOOTSTRAPPED = True
    logger.debug("agent 框架内置组件注册完成")


def _memory_checkpointer():
    """默认内存检查点（进程内多轮记忆；重启即失，生产换 sqlite/postgres）。"""
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()
