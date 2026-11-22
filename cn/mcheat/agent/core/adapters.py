"""内置适配器：LangChain LLM 适配与基于 LLM 的意图分类。

框架业务代码只依赖 ``LLMClient`` / ``IntentClassifier`` 抽象；
本模块是两者的默认实现，langchain-core 是本框架的硬依赖
（LangGraph 本身也依赖它），不引入任何厂商 SDK。
"""
from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .._utils import get_logger
from ..exceptions import LlmError
from ..models import Intent
from .interfaces import IntentClassifier, LLMClient

logger = get_logger(__name__)


class LangChainLLMClient(LLMClient):
    """把 LangChain ``BaseChatModel`` 适配成 ``LLMClient``。

    Args:
        model: 任意 LangChain ChatModel（ChatAnthropic / ChatOpenAI /
            兼容接口……）。``None`` 表示「由配置装配但尚未注入模型」——
            构造不报错（保证 import 与装配链路可用），首次调用时才抛
            :class:`LlmError` 并给出注入指引，方便测试替换。

    用法::

        from llm import get_model                       # 根目录模型工厂
        agent = PythonTutorAgent(llm=get_model())       # 门面自动包装
    """

    def __init__(self, model: BaseChatModel | None = None) -> None:
        self.model = model

    def invoke(
        self,
        prompt: str,
        *,
        system: str | None = None,
        history: Sequence[Any] | None = None,
    ) -> str:
        if self.model is None:
            raise LlmError(
                "LLM 尚未注入：LangChainLLMClient 没有拿到 BaseChatModel 实例",
                hint="构造时传入 PythonTutorAgent(llm=get_model())，"
                     "或在 overrides 里提供实现了 LLMClient 的组件",
            )
        messages: list[Any] = []
        if system:
            messages.append(SystemMessage(content=system))
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=prompt))
        try:
            response = self.model.invoke(messages)
        except Exception as exc:  # 网络 / key / 限流等，统一包成可诊断错误
            raise LlmError(
                f"LLM 调用失败：{type(exc).__name__}: {exc}",
                hint="检查 .env 的 API key 与 LLM_PROVIDER 配置、网络连通性",
            ) from exc
        # 部分模型返回 thinking 块列表，.content 不是 str；优先用 .text
        text = getattr(response, "text", None)
        if not text and isinstance(response.content, str):
            text = response.content
        return text or str(response.content)


DEFAULT_INTENT_SYSTEM_PROMPT = """\
你是一个意图分类器。根据用户问题判断意图，只输出下面四个单词之一，不要任何解释：

- syntax：询问 Python 语法 / 语言特性 / 用法（如：列表推导式怎么写、装饰器是什么、with 语句原理）
- env：询问或要求检测 Python 安装环境（版本、解释器路径、pip、虚拟环境、已安装的包）
- review：希望对一段代码进行修改指导、找 bug、优化建议
- unclear：以上都不是、问题信息不足、或同时包含多种意图无法判断
"""


class LlmIntentClassifier(IntentClassifier):
    """默认意图分类器：一次轻量 LLM 调用 + 容错解析。

    设计约定（见 ``IntentClassifier`` 文档）：任何失败都返回
    :attr:`Intent.UNCLEAR`，交由 clarify 链路向用户确认，绝不中断对话。
    """

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str | None = None,
        history_window: int = 6,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt or DEFAULT_INTENT_SYSTEM_PROMPT
        self.history_window = history_window

    def classify(self, question: str, history: Sequence[Any] = ()) -> Intent:
        recent = list(history[-self.history_window:]) if history else []
        try:
            raw = self.llm.invoke(
                f"用户问题：{question}\n请输出意图类别。",
                system=self.system_prompt,
                history=recent,
            )
        except LlmError as exc:
            logger.warning("意图分类 LLM 调用失败，降级为 unclear：%s", exc)
            return Intent.UNCLEAR
        intent = Intent.from_value(raw)
        logger.debug("意图分类结果：%s（原始输出：%r）", intent.value, raw[:50])
        return intent
