"""语法查询技能：本地知识库检索 + LLM 组织讲解。

知识未命中时**如实说明并推荐可问主题**，不让 LLM 凭空编造 ——
与 rag 框架第 5 条「兜底链路：如实说没找到」同一哲学。
"""
from __future__ import annotations

from typing import Any, Sequence

from .._utils import get_logger
from ..core.interfaces import LLMClient, Skill, SyntaxKnowledgeBase
from ..exceptions import LlmError, SkillError
from ..models import Intent, SkillOutput

logger = get_logger(__name__)

SYNTAX_SYSTEM_PROMPT = """\
你是耐心的 Python 语法讲师。请只依据下方提供的【知识库资料】回答用户问题：
- 资料没有覆盖的内容如实说明，不要编造；
- 讲解顺序：一句话结论 -> 代码示例 -> 常见坑；
- 代码示例必须可以直接复制运行。"""


class SyntaxSkill(Skill):
    """意图 ``syntax`` 对应的技能分支。

    Args:
        llm: LLM 客户端（组织讲解语言）。
        knowledge_base: 语法知识库（默认为内置本地实现，可替换向量检索）。
    """

    name = "syntax"

    def __init__(self, llm: LLMClient, knowledge_base: SyntaxKnowledgeBase) -> None:
        self.llm = llm
        self.knowledge_base = knowledge_base

    def run(self, question: str, history: Sequence[Any] = ()) -> SkillOutput:
        hit = self.knowledge_base.search(question)

        if not hit.matched:
            topics = "、".join(hit.suggestions[:8]) + (
                " 等" if len(hit.suggestions) > 8 else ""
            )
            content = (
                f"本地语法知识库中没有找到与你的问题直接匹配的主题，"
                f"这里如实说明而不是猜测。\n\n"
                f"你可以换个说法重问，或从这些主题中挑选：{topics}。"
            )
            logger.info("语法知识库未命中，已如实告知（推荐 %d 个主题）", len(hit.suggestions))
            return SkillOutput(
                intent=Intent.SYNTAX,
                content=content,
                matched_topic=None,
                tool="search_python_syntax",
                details={"suggestions": hit.suggestions},
            )

        prompt = (
            f"【知识库资料：{hit.title}】\n{hit.content}\n\n"
            f"【用户问题】\n{question}"
        )
        try:
            content = self.llm.invoke(prompt, system=SYNTAX_SYSTEM_PROMPT, history=history)
        except LlmError:
            raise  # LLM 层错误保持原类型透出，调用方可统一按 LlmError 处理
        except Exception as exc:
            raise SkillError(
                f"语法讲解生成失败：{type(exc).__name__}: {exc}",
                hint="检查 LLM 连接；知识库已命中，可稍后重试",
            ) from exc
        return SkillOutput(
            intent=Intent.SYNTAX,
            content=content,
            matched_topic=hit.title,
            tool="search_python_syntax",
            details={"topic": hit.topic},
        )
