"""全部抽象接口（扩展点）。

框架内置实现只是「默认值」，生产环境通过三种方式替换（见 README）：
改配置名 / 向注册表注册自定义实现 / 构造时直接注入实例。
所有接口都是 ``abc.ABC``：鸭子类型写错方法名会在装配期被拦下。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Sequence

from ..models import EnvReport, Intent, KnowledgeHit, SkillOutput


class LLMClient(ABC):
    """最小 LLM 抽象：文本进、文本出。

    刻意不绑定任何厂商 SDK / 消息协议：意图分类、技能 prompt 都只需要
    「system + 可选历史 + 用户输入 → 一段文本」。内置适配器
    :class:`~cn.mcheat.agent.core.adapters.LangChainLLMClient` 包装 LangChain
    的 ``BaseChatModel``，业务代码只依赖本接口。
    """

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        *,
        system: str | None = None,
        history: Sequence[Any] | None = None,
    ) -> str:
        """执行一次补全。

        Args:
            prompt: 本次用户输入 / 指令。
            system: 可选 system 指令。
            history: 可选的对话历史（原生消息对象或 ``(role, text)`` 元组）。

        Returns:
            模型输出的纯文本。

        Raises:
            LlmError: 调用失败（key 缺失 / 网络 / 超时）。
        """


class IntentClassifier(ABC):
    """意图分类器：把用户问题映射到 :class:`Intent` 四分类。"""

    @abstractmethod
    def classify(self, question: str, history: Sequence[Any] = ()) -> Intent:
        """分类意图。

        约定：**解析失败 / LLM 异常应返回** :attr:`Intent.UNCLEAR` **而不是抛异常**，
        由 clarify 链路向用户确认，保证对话不中断。
        """


class Skill(ABC):
    """一个能力分支（syntax / environment / review）。

    实现类在 ``SKILL_REGISTRY`` 按名注册，由 intent 节点路由分发；
    ``name`` 必须与对应 :class:`Intent` 的 value 一致。
    """

    name: ClassVar[str]

    @abstractmethod
    def run(self, question: str, history: Sequence[Any] = ()) -> SkillOutput:
        """执行技能并产出结构化结果。"""


class SyntaxKnowledgeBase(ABC):
    """Python 语法知识库（默认为内置本地实现，可替换为向量检索）。"""

    @abstractmethod
    def search(self, query: str) -> KnowledgeHit:
        """按用户问题检索讲解内容。

        未命中时返回 ``matched=False`` 的 :class:`KnowledgeHit`（附推荐主题），
        框架会如实告知用户而不是编造答案。
        """


class EnvProbe(ABC):
    """环境探针：收集本机 Python 安装环境信息。"""

    @abstractmethod
    def probe(self) -> EnvReport:
        """执行探测并返回快照。单项失败记入 ``issues``，不抛异常。"""
