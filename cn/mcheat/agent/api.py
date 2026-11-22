"""PythonTutorAgent 门面：四个动作覆盖完整交互闭环。

用法::

    from llm import get_model                        # 根目录模型工厂
    from cn.mcheat.agent import PythonTutorAgent

    agent = PythonTutorAgent(llm=get_model())
    outcome = agent.ask("列表推导式怎么写？", thread_id="user-1")

    if outcome is ClarifyInterrupt:                  # 意图不明确 -> 展示澄清问题
        reply = input(outcome.question)
        outcome = agent.resume(reply, thread_id="user-1")

    print(outcome.content)                           # 最终答案

所有组件可插拔：改配置名 / 注册自定义实现 / 构造时注入实例（同 rag 框架）。
"""
from __future__ import annotations

from typing import Any, Mapping

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from ._utils import get_logger
from .config import AgentConfig
from .core.adapters import LangChainLLMClient
from .core.bootstrap import bootstrap
from .core.interfaces import LLMClient
from .core.registry import ALL_REGISTRIES, BUILD_ORDER, resolve_references
from .exceptions import AgentError, ClarifyError, ConfigError, GraphError, LlmError
from .graph.builder import build_graph
from .graph.nodes import validate_clarify_reply
from .models import AgentAnswer, ClarifyInterrupt, Intent

logger = get_logger(__name__)


class PythonTutorAgent:
    """配置驱动的 Python 学习 Agent 门面。

    Args:
        config: :class:`AgentConfig`；不传则加载内置默认配置（含环境变量覆盖）。
        overrides: 组件实例覆盖字典，键取自 ``BUILD_ORDER``（如
            ``{"knowledge_base": MyKB()}``）。注入对象必须实现对应 ABC，
            否则抛 :class:`ConfigError`。
        llm: LLM 注入快捷方式 —— 接受 ``BaseChatModel``（自动包装成
            :class:`LangChainLLMClient`）或 ``LLMClient`` 实例，等价于
            ``overrides={"llm": ...}``。
        checkpointer: 检查点存储实例（覆盖配置里的 checkpointer 组件）。
            interrupt/resume 依赖它，默认为 MemorySaver。

    Raises:
        ConfigError: 配置校验失败或组件契约不满足。
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        overrides: Mapping[str, Any] | None = None,
        *,
        llm: BaseChatModel | LLMClient | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        bootstrap()
        if config is None:
            config = AgentConfig.load()
        if not isinstance(config, AgentConfig):
            raise ConfigError("config 必须是 AgentConfig 实例")
        config.validate()
        self.config = config

        merged = dict(overrides or {})
        if llm is not None:
            merged["llm"] = self._wrap_llm(llm)
        self._components = self._build_components(merged)

        if checkpointer is not None:
            self._components["checkpointer"] = checkpointer

        self.graph = build_graph(
            self._components, config, checkpointer=self._components["checkpointer"]
        )
        logger.info(
            "PythonTutorAgent 就绪（max_clarify_turns=%d, recursion_limit=%d）",
            config.graph.max_clarify_turns, config.graph.recursion_limit,
        )

    # -- 装配 ---------------------------------------------------------------

    @staticmethod
    def _wrap_llm(llm: Any) -> LLMClient:
        """把 ``BaseChatModel`` / ``LLMClient`` 统一成 ``LLMClient``。"""
        if isinstance(llm, LLMClient):
            return llm
        if isinstance(llm, BaseChatModel):
            return LangChainLLMClient(model=llm)
        raise ConfigError(
            f"llm 参数需要是 BaseChatModel 或 LLMClient 实例，"
            f"收到 {type(llm).__name__}",
            hint="例如 PythonTutorAgent(llm=get_model())（根目录 llm.py 工厂）",
        )

    def _build_components(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """按 BUILD_ORDER 构造组件，解析 ``"@xxx"`` 依赖引用。"""
        unknown = set(overrides) - set(BUILD_ORDER)
        if unknown:
            raise ConfigError(
                f"overrides 含未知组件键 {sorted(unknown)}；可用：{list(BUILD_ORDER)}"
            )

        built: dict[str, Any] = {}
        for key in BUILD_ORDER:
            registry = ALL_REGISTRIES[key]
            if key in overrides:
                instance = overrides[key]
                if registry.base is not None and not isinstance(instance, registry.base):
                    raise ConfigError(
                        f"overrides[{key!r}] 的 {type(instance).__name__} "
                        f"未实现 {registry.base.__name__} 接口"
                    )
                built[key] = instance
                logger.info("组件 %s 由 overrides 注入：%s", key, type(instance).__name__)
                continue

            spec = self.config.component(key)
            params = resolve_references(spec.params, built)
            params = self._inject_toplevel_params(key, spec.name, params)
            built[key] = registry.create(spec.name, **params)
            logger.debug("组件 %s = %s(%s)", key, spec.name, type(built[key]).__name__)
        return built

    def _inject_toplevel_params(
        self, key: str, name: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """把顶层配置段的值注入对应内置组件的构造参数（单一事实来源）。

        顶层段是文档化的参数入口（含 ``AGENT_*`` 环境变量映射），
        组件 params 里同名键以顶层为准。
        """
        if key == "intent_classifier" and name == "llm":
            params.setdefault("history_window", self.config.intent.history_window)
            if self.config.intent.system_prompt:
                params["system_prompt"] = self.config.intent.system_prompt
        elif key == "env_probe" and name == "local":
            params.update(
                timeout=self.config.env_probe.timeout,
                include_pip_list=self.config.env_probe.include_pip_list,
                pip_list_limit=self.config.env_probe.pip_list_limit,
            )
        return params

    def component(self, key: str) -> Any:
        """取已装配的组件实例（便于自定义扩展与调试）。"""
        try:
            return self._components[key]
        except KeyError as exc:
            raise ConfigError(f"未知组件 {key!r}；可用：{list(BUILD_ORDER)}") from exc

    # -- 在线交互 -----------------------------------------------------------

    def ask(self, question: str, thread_id: str = "default") -> AgentAnswer | ClarifyInterrupt:
        """提出问题，得到答案或澄清请求。

        Args:
            question: 用户问题 / 代码。
            thread_id: 会话线程 ID（同一 ID 共享多轮记忆）。

        Returns:
            :class:`AgentAnswer`（完成）或 :class:`ClarifyInterrupt`（需要人工澄清，
            调用 :meth:`resume` 继续）。

        Raises:
            AgentError: 问题为空。
            LlmError: LLM 调用失败。
            GraphError: 图执行失败（含递归超限）。
        """
        if not question or not str(question).strip():
            raise AgentError("question 不能为空", hint="请传入用户的问题文本")
        question = str(question).strip()

        # 每轮重置路由相关字段；messages 由 add_messages 追加保留多轮记忆
        payload = {
            "question": question,
            "messages": [HumanMessage(content=question)],
            "intent": "",
            "clarify_turns": 0,
            "skill_output": None,
            "answer": "",
        }
        return self._run(payload, thread_id)

    def resume(self, reply: str, thread_id: str = "default") -> AgentAnswer | ClarifyInterrupt:
        """用用户的澄清回复从 interrupt 暂停点继续执行。

        Args:
            reply: 用户对澄清问题的回复。
            thread_id: 必须与触发澄清的 :meth:`ask` 相同。

        Returns:
            同 :meth:`ask`（仍可能再次澄清，直到达到 max_clarify_turns）。

        Raises:
            ClarifyError: 回复为空。
            GraphError: 没有待恢复的暂停点，或图执行失败。
        """
        reply = validate_clarify_reply(reply)
        return self._run(Command(resume=reply), thread_id)

    def history(self, thread_id: str = "default") -> list[Any]:
        """取某个会话线程的完整消息历史（调试 / 展示用）。"""
        state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return list(state.values.get("messages", [])) if state.values else []

    # -- 内部 ---------------------------------------------------------------

    def _run_config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.config.graph.recursion_limit,
        }

    def _run(self, payload: Any, thread_id: str) -> AgentAnswer | ClarifyInterrupt:
        try:
            result = self.graph.invoke(payload, config=self._run_config(thread_id))
        except GraphRecursionError as exc:
            raise GraphError(
                f"图执行超过 recursion_limit={self.config.graph.recursion_limit}",
                hint="提高 graph.recursion_limit，或检查意图分类是否反复落入 unclear",
            ) from exc
        except (LlmError, AgentError):
            raise  # 业务异常原样透出，保证调用方可分类处理
        except Exception as exc:
            raise GraphError(
                f"图执行失败：{type(exc).__name__}: {exc}",
                hint="若为「无待恢复的暂停点」，请确认 resume 与 ask 使用相同 thread_id",
            ) from exc

        if not isinstance(result, dict):
            raise GraphError(f"图执行返回了意外类型：{type(result).__name__}")

        interrupts = result.get("__interrupt__")
        if interrupts:
            # human-in-the-loop 暂停：把 payload 转成 ClarifyInterrupt 交调用方
            payload_value = getattr(interrupts[0], "value", interrupts[0])
            return ClarifyInterrupt.from_payload(payload_value)

        return AgentAnswer(
            content=str(result.get("answer") or ""),
            intent=Intent.from_value(result.get("intent") or "unclear"),
            skill_output=result.get("skill_output"),
        )


__all__ = ["PythonTutorAgent"]
