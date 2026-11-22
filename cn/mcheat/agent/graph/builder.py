"""StateGraph 组装：节点 / 条件边 / 编译（纯编排，不含业务逻辑）。

图结构::

    START -> intent ─┬─(syntax)────> syntax        ──┐
                     ├─(env)──────> environment    ──┤
                     ├─(review)───> review         ──┼─> answer -> END
                     └─(unclear)──> clarify ─┬─(熔断)─┘
                                           └─(resume 回复)─> intent（循环）

关键约束：``interrupt`` / ``Command(resume=...)`` 依赖 checkpointer 保存
暂停点，``compile`` 必须传入 checkpointer 且 invoke 配置带 ``thread_id``。
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .._utils import get_logger
from ..config import AgentConfig
from ..exceptions import GraphError
from .nodes import build_node_functions, route_after_clarify, route_after_intent
from .state import TutorState

logger = get_logger(__name__)


def build_graph(
    components: dict[str, Any],
    config: AgentConfig,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """组装并编译 Python 学习 Agent 的 LangGraph 图。

    Args:
        components: 已装配组件字典（须含 skill_syntax / skill_env / skill_review /
            intent_classifier 等，见 ``core.registry.BUILD_ORDER``）。
        config: 已校验配置（max_clarify_turns 等运行参数）。
        checkpointer: 检查点存储。**不传则 clarify 的 interrupt 无法恢复**，
            生产上必须提供（门面默认注入 MemorySaver）。

    Returns:
        编译好的 ``CompiledStateGraph``。

    Raises:
        GraphError: 图构建或编译失败。
    """
    try:
        nodes = build_node_functions(components, config)

        builder = StateGraph(TutorState)
        for name, fn in nodes.items():
            builder.add_node(name, fn)

        builder.add_edge(START, "intent")
        builder.add_conditional_edges(
            "intent",
            route_after_intent,
            {
                "syntax": "syntax",
                "environment": "environment",
                "review": "review",
                "clarify": "clarify",
            },
        )
        builder.add_edge("syntax", "answer")
        builder.add_edge("environment", "answer")
        builder.add_edge("review", "answer")
        builder.add_conditional_edges(
            "clarify",
            route_after_clarify,
            {"answer": "answer", "intent": "intent"},
        )
        builder.add_edge("answer", END)

        graph = builder.compile(checkpointer=checkpointer)
    except GraphError:
        raise
    except Exception as exc:
        raise GraphError(
            f"LangGraph 图构建失败：{type(exc).__name__}: {exc}",
            hint="检查组件是否实现接口契约、config 是否通过 validate()",
        ) from exc

    logger.debug("图构建完成：节点=%s", sorted(nodes))
    return graph
