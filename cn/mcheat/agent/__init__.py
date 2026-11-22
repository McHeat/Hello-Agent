"""生产级 Python 学习 Agent 框架（基于 LangGraph）。

三类能力 + human-in-the-loop 意图澄清，图结构::

    START -> intent ─┬─(syntax)────> syntax      ──┐
                     ├─(env)──────> environment  ──┼─> answer -> END
                     ├─(review)───> review       ──┤
                     └─(unclear)──> clarify ─┬─(熔断)─┘
                                           └─(resume)─> intent

设计要点（与 ``cn/mcheat.rag`` 同一套工程范式）：

* 组件全部 ABC 接口 + 具名注册表，改配置名 / 注册自定义实现 / 注入实例
  三种方式可替换；LLM 通过 ``llm=`` 参数注入（默认适配 LangChain
  ``BaseChatModel``），框架零厂商 SDK 依赖；
* 配置优先级：``configs/defaults.json`` < 用户 JSON < ``AGENT_*`` 环境变量
  < overrides 字典，``validate()`` 硬校验 + 软告警；
* 澄清链路用 ``interrupt()`` / ``Command(resume=...)``，``max_clarify_turns``
  熔断防止 clarify↔intent 死循环；
* 异常分阶段（ConfigError / LlmError / SkillError / ClarifyError / GraphError），
  报错带修复建议；库代码只挂 NullHandler。

快速开始::

    from llm import get_model                    # 根目录模型工厂（.env 切换厂商）
    from cn.mcheat.agent import PythonTutorAgent, ClarifyInterrupt

    agent = PythonTutorAgent(llm=get_model())
    outcome = agent.ask("列表推导式怎么写？", thread_id="u1")
    while isinstance(outcome, ClarifyInterrupt):  # 意图不明 -> 问用户 -> 继续
        print(outcome.question)
        outcome = agent.resume(input("你："), thread_id="u1")
    print(outcome.content)

从项目根目录运行示例：``python -m cn.mcheat.agent.examples.cli``
（详见包内 ``README.md``）。
"""
from __future__ import annotations

from .core.bootstrap import bootstrap

# import 本包即完成内置组件注册，用户代码可以直接用配置名替换组件
bootstrap()

from .api import PythonTutorAgent  # noqa: E402  (必须在 bootstrap 之后)
from .config import AgentConfig  # noqa: E402
from .exceptions import (  # noqa: E402
    AgentError,
    ClarifyError,
    ConfigError,
    DependencyMissingError,
    GraphError,
    LlmError,
    SkillError,
)
from .models import (  # noqa: E402
    INTENT_CANDIDATES,
    AgentAnswer,
    ClarifyInterrupt,
    ClarifyRequest,
    EnvReport,
    Intent,
    KnowledgeHit,
    SkillOutput,
)

__version__ = "1.0.0"

__all__ = [
    # 门面 / 配置
    "PythonTutorAgent", "AgentConfig", "bootstrap", "__version__",
    # 交互结果
    "AgentAnswer", "ClarifyInterrupt", "ClarifyRequest", "Intent",
    "INTENT_CANDIDATES",
    # 领域模型
    "SkillOutput", "KnowledgeHit", "EnvReport",
    # 异常
    "AgentError", "ConfigError", "DependencyMissingError", "LlmError",
    "SkillError", "ClarifyError", "GraphError",
]
