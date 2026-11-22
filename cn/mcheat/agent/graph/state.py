"""TutorState：图里流动的数据结构。

设计说明：

* ``messages`` 用 ``add_messages`` 做**追加**语义，配合 checkpointer 支撑
  多轮记忆；其余字段每轮由门面 ``ask()`` 重置，避免上一轮的 intent /
  answer 残留污染本轮路由；
* ``skill_output`` 直接存 :class:`SkillOutput` dataclass（MemorySaver 原生
  支持；sqlite/postgres checkpointer 的序列化器同样可处理 dataclass）。
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from ..models import SkillOutput


class TutorState(TypedDict):
    """Python 学习 Agent 的图状态。"""

    messages: Annotated[list[AnyMessage], add_messages]  # 对话历史（追加）
    question: str                       # 本轮用户问题
    intent: str                         # 意图分类结果（Intent.value）
    clarify_turns: int                  # 本轮已发起的澄清次数（熔断计数器）
    skill_output: SkillOutput | None    # 技能分支产出（未执行为 None）
    answer: str                         # 最终答复（answer 节点写入）
