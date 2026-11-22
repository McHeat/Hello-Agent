"""测试替身（不被 discover 收集）：FakeLLM、Stub 知识库 / 探针。"""
from __future__ import annotations

from typing import Any, Sequence

from cn.mcheat.agent.core.interfaces import EnvProbe, LLMClient, SyntaxKnowledgeBase
from cn.mcheat.agent.exceptions import LlmError
from cn.mcheat.agent.models import EnvReport, KnowledgeHit


class FakeLLM(LLMClient):
    """脚本化 LLM：按队列依次返回预设回复；耗尽后回显 prompt 或抛错。

    Args:
        responses: 预设回复队列（每次 invoke 弹出一个）。
        error_when_exhausted: 队列耗尽后再被调用时抛 :class:`LlmError`
            （用于验证错误透传链路）。
    """

    def __init__(self, responses: Sequence[str] | None = None,
                 error_when_exhausted: bool = False) -> None:
        self.responses: list[str] = list(responses or [])
        self.error_when_exhausted = error_when_exhausted
        self.calls: list[dict[str, Any]] = []

    def invoke(self, prompt: str, *, system: str | None = None,
               history: Sequence[Any] | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system,
                           "history": list(history or [])})
        if self.responses:
            return self.responses.pop(0)
        if self.error_when_exhausted:
            raise LlmError("FakeLLM：预设回复已耗尽（模拟 LLM 故障）")
        return f"echo: {prompt}"


class StubKnowledgeBase(SyntaxKnowledgeBase):
    """可控命中的知识库：hit=True 时永远命中固定主题。"""

    def __init__(self, hit: bool = True) -> None:
        self.hit = hit

    def search(self, query: str) -> KnowledgeHit:
        if self.hit:
            return KnowledgeHit(
                matched=True, topic="stub_topic", title="桩主题",
                content="桩知识内容：这段内容应出现在 LLM 的 prompt 里。",
            )
        return KnowledgeHit(matched=False, suggestions=["主题A", "主题B"])


class StubEnvProbe(EnvProbe):
    """固定返回的探针，避免测试依赖真实 subprocess。"""

    def __init__(self, report: EnvReport | None = None) -> None:
        self.report = report or EnvReport(
            python_version="3.13.1", executable="/tmp/venv/bin/python",
            platform_info="macOS-15-arm64", virtual_env=True,
            venv_detail="/tmp/venv", pip_available=True,
            pip_version="pip 25.0 (from /tmp/venv)", packages=["pip", "setuptools"],
        )
        self.call_count = 0

    def probe(self) -> EnvReport:
        self.call_count += 1
        return self.report
