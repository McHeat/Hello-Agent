"""节点函数：intent / clarify / 技能包装 / answer。

节点只做编排（读 state、调组件、返回部分更新），业务逻辑全部在
可插拔组件里 —— 这样图结构稳定，能力随组件替换演进。

节点清单（与 README 的状态流转图对应）::

    intent    意图分类（LLM） -> 条件边路由到四个分支
    syntax    语法查询（SyntaxSkill）
    environment 环境检测（EnvSkill）
    review    代码指导（ReviewSkill）
    clarify   意图澄清（interrupt 暂停 -> 人回复 -> 回 intent 重分类；超限熔断）
    answer    汇总输出（确定性：直接采用技能产出或熔断话术）
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from .._utils import get_logger
from ..config import AgentConfig
from ..core.interfaces import Skill
from ..exceptions import ClarifyError
from ..models import INTENT_CANDIDATES, ClarifyRequest, Intent, SkillOutput
from .state import TutorState

logger = get_logger(__name__)


def build_node_functions(
    components: dict[str, Any], config: AgentConfig
) -> dict[str, Callable[[TutorState], dict]]:
    """用已装配的组件 + 配置闭包出全部节点函数。

    Args:
        components: ``BUILD_ORDER`` 构建出的组件实例字典。
        config: 已校验的 :class:`AgentConfig`。

    Returns:
        节点名 -> 节点函数 的映射（key 与 builder 中的节点名一致）。
    """

    # -- intent：意图分类 ----------------------------------------------------
    def intent_node(state: TutorState) -> dict:
        history: Sequence[Any] = list(state["messages"])[:-1]  # 排除本轮问题本身
        intent = components["intent_classifier"].classify(state["question"], history=history)
        logger.info("意图分类完成：%s", intent.value)
        return {"intent": intent.value}

    # -- 技能分支：包装 Skill 成 LangGraph 节点 ------------------------------
    def _make_skill_node(skill: Skill) -> Callable[[TutorState], dict]:
        def node(state: TutorState) -> dict:
            history: Sequence[Any] = list(state["messages"])[:-1]
            output: SkillOutput = skill.run(state["question"], history=history)
            logger.info("技能 %s 执行完成（intent=%s）", skill.name, output.intent.value)
            return {"skill_output": output, "intent": output.intent.value}

        return node

    # -- clarify：human-in-the-loop 意图澄清 ---------------------------------
    def clarify_node(state: TutorState) -> dict:
        turns = int(state.get("clarify_turns") or 0)

        # 熔断：超过最大澄清轮数后不再打扰用户，如实说明并结束本轮
        if turns >= config.graph.max_clarify_turns:
            logger.warning("澄清轮数已达上限 %d，熔断并结束本轮", config.graph.max_clarify_turns)
            return {
                "answer": config.clarify.exhausted_reply,
                "intent": Intent.UNCLEAR.value,
            }

        candidates = [
            f"{i}. {INTENT_CANDIDATES[key]}" for i, key in enumerate(("syntax", "env", "review"), 1)
        ]
        question = config.clarify.question_template.format(
            candidates="\n".join(candidates)
        )
        payload = ClarifyRequest(
            question=question,
            candidates=candidates,
            turns_used=turns + 1,
        ).as_dict()

        # interrupt() 暂停图执行，payload 透传给调用方（门面转成 ClarifyInterrupt）；
        # 用户通过 Command(resume=回复) 恢复后，interrupt 的返回值即用户回复。
        reply = interrupt(payload)
        if isinstance(reply, dict):  # 容错：调用方可能传结构化回复
            reply = reply.get("reply", "")
        reply = str(reply or "").strip() or "（用户未提供补充信息）"

        # 澄清文本同时并入「问题」与「历史」：并入 question 让技能层的
        # 知识库检索 / prompt 直接看到补充信息；并入 messages 保留完整对话轨迹
        return {
            "question": f"{state['question']}\n（用户补充：{reply}）",
            "clarify_turns": turns + 1,
            "messages": [HumanMessage(content=f"（补充说明）{reply}")],
        }

    # -- answer：确定性汇总 ---------------------------------------------------
    def answer_node(state: TutorState) -> dict:
        output = state.get("skill_output")
        if output is not None:
            return {"answer": output.content}
        # 走到这里只剩一种情况：clarify 熔断（answer 已写入）或异常路径，兜底话术
        return {"answer": state.get("answer") or config.clarify.exhausted_reply}

    return {
        "intent": intent_node,
        "syntax": _make_skill_node(components["skill_syntax"]),
        "environment": _make_skill_node(components["skill_env"]),
        "review": _make_skill_node(components["skill_review"]),
        "clarify": clarify_node,
        "answer": answer_node,
    }


def route_after_intent(state: TutorState) -> str:
    """intent 之后的条件边：意图值 -> 节点名（未知意图一律进 clarify 兜底）。"""
    mapping = {
        Intent.SYNTAX.value: "syntax",
        Intent.ENV.value: "environment",
        Intent.REVIEW.value: "review",
        Intent.UNCLEAR.value: "clarify",
    }
    return mapping.get(str(state.get("intent") or ""), "clarify")


def route_after_clarify(state: TutorState) -> str:
    """clarify 之后的条件边：熔断（answer 已写入）去 answer，否则回 intent 重分类。"""
    if state.get("answer"):
        return "answer"
    return "intent"


def validate_clarify_reply(reply: Any) -> str:
    """门面侧对 resume 值的轻校验（供 api.py 使用）。"""
    if reply is None or (isinstance(reply, str) and not reply.strip()):
        raise ClarifyError(
            "resume 回复内容为空",
            hint="请传入用户的澄清回复文本，例如 agent.resume('我想问列表推导式')",
        )
    return str(reply)
