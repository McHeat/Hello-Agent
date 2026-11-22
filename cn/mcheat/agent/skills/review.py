"""代码修改指导技能：LLM 三段式评审（问题定位 / 修改建议 / 改进代码）。"""
from __future__ import annotations

from typing import Any, Sequence

from .._utils import get_logger
from ..core.interfaces import LLMClient, Skill
from ..exceptions import LlmError, SkillError
from ..models import Intent, SkillOutput

logger = get_logger(__name__)

REVIEW_SYSTEM_PROMPT = """\
你是资深 Python 代码评审。请对用户提供的代码按固定三段式输出：

## 1. 问题定位
逐条列出 bug、坏味道与隐患（引用具体行号或代码片段）；没有问题就如实说明。

## 2. 修改建议
针对每个问题给出具体做法与理由（为什么这样改）。

## 3. 改进后的代码
给出完整可运行的改进版；如果原代码本身没有问题，只给可选的优化建议。

要求：只点评用户给出的代码；不确定的猜测要标注「猜测」。"""


class ReviewSkill(Skill):
    """意图 ``review`` 对应的技能分支（纯 LLM 分析，无工具）。"""

    name = "review"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, question: str, history: Sequence[Any] = ()) -> SkillOutput:
        try:
            content = self.llm.invoke(
                question, system=REVIEW_SYSTEM_PROMPT, history=history
            )
        except LlmError:
            raise  # LLM 层错误保持原类型透出，调用方可统一按 LlmError 处理
        except Exception as exc:
            raise SkillError(
                f"代码评审生成失败：{type(exc).__name__}: {exc}",
                hint="检查 LLM 连接后重试",
            ) from exc
        return SkillOutput(intent=Intent.REVIEW, content=content)
