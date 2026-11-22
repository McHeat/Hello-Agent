"""领域模型：意图、知识命中、环境报告、技能产出与交互结果。

本模块只依赖标准库（``enum`` / ``dataclasses``），不 import langgraph /
langchain，保证被任何层安全引用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exceptions import ClarifyError


# ---------------------------------------------------------------------------
# 意图
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    """用户意图的四分类（值即图路由键）。"""

    SYNTAX = "syntax"      # 查询 Python 语法知识
    ENV = "env"            # 检测 Python 安装环境
    REVIEW = "review"      # 代码修改指导
    UNCLEAR = "unclear"    # 意图不明确，需要澄清

    @classmethod
    def from_value(cls, raw: Any) -> "Intent":
        """容错解析：LLM 输出、配置字符串、枚举实例都能安全转换。

        解析失败不抛异常，一律返回 :attr:`UNCLEAR` —— 意图分类属于
        「错了可以澄清」的环节，绝不应该让整轮对话崩掉。
        """
        if isinstance(raw, cls):
            return raw
        text = str(raw or "").strip().lower()
        if not text:
            return cls.UNCLEAR
        text = text.splitlines()[0]  # 只看第一行，忽略模型可能附带的解释
        if "syntax" in text or "语法" in text:
            return cls.SYNTAX
        if "env" in text or "环境" in text:
            return cls.ENV
        if "review" in text or "代码" in text or "code" in text:
            return cls.REVIEW
        return cls.UNCLEAR


#: 澄清时展示给用户的候选意图说明（key 与 Intent.value 对应）
INTENT_CANDIDATES: dict[str, str] = {
    "syntax": "查询 Python 语法知识 —— 例如「列表推导式怎么写」「装饰器的原理」",
    "env": "检测 Python 安装环境 —— 版本、解释器路径、pip、虚拟环境、已装包",
    "review": "代码修改指导 —— 粘贴你的代码，我来定位问题、给修改建议和改进版",
}


# ---------------------------------------------------------------------------
# 语法知识库
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeHit:
    """一次知识库查询的结果。

    Attributes:
        matched: 是否命中主题。未命中时框架**如实说明**而不是编造答案。
        topic: 命中的主题 key（未命中为 ``None``）。
        title: 命中主题的展示名（未命中为 ``None``）。
        content: 命中主题的讲解正文（未命中为空串）。
        suggestions: 未命中时推荐的可问主题列表。
    """

    matched: bool
    topic: str | None = None
    title: str | None = None
    content: str = ""
    suggestions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 环境检测
# ---------------------------------------------------------------------------


@dataclass
class EnvReport:
    """Python 安装环境的一次探测快照。

    所有字段都是纯文本 / 列表，保证可序列化、可断言；单项探测失败
    不会让整个报告失败，而是记入 ``issues`` 交给上层如实呈现。
    """

    python_version: str = ""        # 如 "3.13.1"
    executable: str = ""            # sys.executable
    platform_info: str = ""         # platform.platform()
    virtual_env: bool = False       # 是否运行在虚拟环境中
    venv_detail: str = ""           # 虚拟环境路径说明（未启用为空）
    pip_available: bool = False
    pip_version: str = ""
    packages: list[str] = field(default_factory=list)   # pip list 截断后的行
    issues: list[str] = field(default_factory=list)     # 探测过程中的失败项

    def to_display(self) -> str:
        """渲染成给 LLM / 用户看的纯文本报告。"""
        lines = [
            f"Python 版本：{self.python_version or '未知'}",
            f"解释器路径：{self.executable or '未知'}",
            f"操作系统：{self.platform_info or '未知'}",
            f"虚拟环境：{'是' + (f'（{self.venv_detail}）' if self.venv_detail else '') if self.virtual_env else '否（系统/全局环境）'}",
            f"pip：{self.pip_version if self.pip_available else '不可用'}",
        ]
        if self.packages:
            lines.append(f"已安装包（截断至 {len(self.packages)} 行）：")
            lines.extend(f"  {row}" for row in self.packages)
        if self.issues:
            lines.append("探测告警：")
            lines.extend(f"  - {issue}" for issue in self.issues)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 技能产出与交互结果
# ---------------------------------------------------------------------------


@dataclass
class SkillOutput:
    """一个技能分支（syntax / environment / review）的执行结果。"""

    intent: Intent
    content: str                                   # 技能产出的最终内容
    matched_topic: str | None = None               # syntax：命中的知识主题
    tool: str | None = None                        # 使用的工具名（如 search_python_syntax）
    details: dict[str, Any] = field(default_factory=dict)  # 附加上下文（如环境摘要）


@dataclass
class ClarifyRequest:
    """clarify 节点经 ``interrupt()`` 发出的澄清请求（payload）。"""

    question: str            # 向用户发出的澄清问题
    candidates: list[str]    # 候选意图说明
    turns_used: int          # 本轮已用澄清轮数（含本次）

    def as_dict(self) -> dict[str, Any]:
        """转成普通 dict —— interrupt payload 保持可序列化，兼容任意 checkpointer。"""
        return {
            "question": self.question,
            "candidates": list(self.candidates),
            "turns_used": self.turns_used,
        }


@dataclass
class ClarifyInterrupt:
    """门面返回给调用方的「需要人工澄清」信号。

    调用方应把 ``question`` 与 ``candidates`` 展示给用户，收集回复后调用
    ``PythonTutorAgent.resume(reply, thread_id)`` 继续。
    """

    question: str
    candidates: list[str]
    turns_used: int

    @classmethod
    def from_payload(cls, payload: Any) -> "ClarifyInterrupt":
        """从 interrupt payload（dict 或 ClarifyRequest）容错构造。"""
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, ClarifyRequest):
            return cls(payload.question, payload.candidates, payload.turns_used)
        if isinstance(payload, dict):
            return cls(
                question=str(payload.get("question", "请补充说明你的需求")),
                candidates=[str(c) for c in payload.get("candidates", [])],
                turns_used=int(payload.get("turns_used", 0)),
            )
        raise ClarifyError(
            f"无法解析澄清请求 payload：{type(payload).__name__}",
            hint="payload 应是 ClarifyRequest 或等价 dict",
        )


@dataclass
class AgentAnswer:
    """门面返回给调用方的最终答案。"""

    content: str                        # 最终答复文本
    intent: Intent                      # 实际服务的意图（unclear 表示澄清超限）
    skill_output: SkillOutput | None = None  # 技能原始产出（便于程序化消费）
