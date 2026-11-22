"""分阶段异常体系：每个报错都带上下文与修复建议。

继承树（根为 :class:`AgentError`）::

    AgentError
    ├── ConfigError              配置加载 / 校验 / 组件装配失败
    ├── DependencyMissingError   可选依赖未安装（附 pip install 建议）
    ├── LlmError                 LLM 调用失败（key 缺失 / 超时 / 网络）
    ├── SkillError               技能分支执行失败
    ├── ClarifyError             澄清链路异常（payload 非法 / 无待恢复的暂停点）
    └── GraphError               LangGraph 图构建 / 执行失败（含递归超限）
"""
from __future__ import annotations


class AgentError(Exception):
    """框架根异常。

    Args:
        message: 面向人的错误描述（发生了什么）。
        hint: 可选的修复建议（怎么办）。生产环境报错必须「可行动」，
            禁止只抛一个裸异常名。
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}（修复建议：{self.hint}）"
        return self.message


class ConfigError(AgentError):
    """配置或组件装配失败。"""


class DependencyMissingError(AgentError):
    """可选依赖未安装（如 langgraph-checkpoint-sqlite）。"""


class LlmError(AgentError):
    """LLM 调用失败。"""


class SkillError(AgentError):
    """技能分支执行失败。"""


class ClarifyError(AgentError):
    """澄清链路异常。"""


class GraphError(AgentError):
    """LangGraph 图构建 / 执行失败。"""
