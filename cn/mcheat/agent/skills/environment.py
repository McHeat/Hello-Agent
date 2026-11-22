"""环境检测技能：探针收集事实 + LLM 总结判断。

职责边界刻意清晰：**探针负责事实（可断言、无幻觉），LLM 只负责解读**。
"""
from __future__ import annotations

from typing import Any, Sequence

from .._utils import get_logger
from ..core.interfaces import EnvProbe, LLMClient, Skill
from ..exceptions import LlmError, SkillError
from ..models import Intent, SkillOutput

logger = get_logger(__name__)

ENV_SYSTEM_PROMPT = """\
你是 Python 环境诊断助手。请只依据下方【探针收集的环境信息】总结当前环境状态：
- 先给一句话总体结论（是否健康、有无需要注意的问题）；
- 再分点列出关键信息（版本 / 解释器 / 虚拟环境 / pip / 关键包）；
- 「探测告警」里的问题要逐条解释成因与解决办法；
- 信息里没有的内容不要编造。"""


class EnvSkill(Skill):
    """意图 ``env`` 对应的技能分支。

    Args:
        llm: LLM 客户端（总结判断）。
        env_probe: 环境探针（默认为本机 subprocess 实现，可替换为远程探测）。
    """

    name = "env"

    def __init__(self, llm: LLMClient, env_probe: EnvProbe) -> None:
        self.llm = llm
        self.env_probe = env_probe

    def run(self, question: str, history: Sequence[Any] = ()) -> SkillOutput:
        report = self.env_probe.probe()
        prompt = (
            f"【探针收集的环境信息】\n{report.to_display()}\n\n"
            f"【用户问题】\n{question}"
        )
        try:
            content = self.llm.invoke(prompt, system=ENV_SYSTEM_PROMPT, history=history)
        except LlmError:
            raise  # LLM 层错误保持原类型透出，调用方可统一按 LlmError 处理
        except Exception as exc:
            raise SkillError(
                f"环境总结生成失败：{type(exc).__name__}: {exc}",
                hint="探针数据已收集成功，可查看 details 中的原始报告",
            ) from exc
        return SkillOutput(
            intent=Intent.ENV,
            content=content,
            tool="check_python_env",
            details={
                "python_version": report.python_version,
                "executable": report.executable,
                "virtual_env": report.virtual_env,
                "pip_available": report.pip_available,
                "issues": list(report.issues),
            },
        )
