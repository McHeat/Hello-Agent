"""test_04：图路由四分支、clarify 循环与熔断、条件边函数。"""
from __future__ import annotations

import unittest

from cn.mcheat.agent.config import AgentConfig
from cn.mcheat.agent.graph.nodes import route_after_clarify, route_after_intent
from cn.mcheat.agent import PythonTutorAgent, AgentAnswer, ClarifyInterrupt, Intent

from ._fakes import FakeLLM


def _agent(max_clarify_turns: int = 2, llm: FakeLLM | None = None) -> PythonTutorAgent:
    cfg = AgentConfig.load(
        overrides={"graph": {"max_clarify_turns": max_clarify_turns}}
    )
    return PythonTutorAgent(config=cfg, llm=llm or FakeLLM())


class TestIntentRouting(unittest.TestCase):
    def test_syntax_route(self):
        llm = FakeLLM(responses=["syntax", "讲解：列表推导式……"])
        agent = _agent(llm=llm)
        out = agent.ask("列表推导式怎么写", thread_id="t-syntax")
        self.assertIsInstance(out, AgentAnswer)
        self.assertEqual(out.intent, Intent.SYNTAX)
        self.assertIn("讲解", out.content)
        self.assertEqual(out.skill_output.matched_topic, "列表推导式")

    def test_env_route(self):
        llm = FakeLLM(responses=["env", "环境总结……"])
        agent = _agent(llm=llm)
        out = agent.ask("看看我的 Python 环境配置", thread_id="t-env")
        self.assertIsInstance(out, AgentAnswer)
        self.assertEqual(out.intent, Intent.ENV)
        self.assertEqual(out.skill_output.tool, "check_python_env")

    def test_review_route(self):
        llm = FakeLLM(responses=["review", "评审结果……"])
        agent = _agent(llm=llm)
        out = agent.ask("def add(a,b): return a+b\n 帮我看看", thread_id="t-review")
        self.assertIsInstance(out, AgentAnswer)
        self.assertEqual(out.intent, Intent.REVIEW)


class TestClarifyFlow(unittest.TestCase):
    def test_unclear_interrupts_then_recovers(self):
        # 第一次分类 unclear -> interrupt；用户澄清后重分类 syntax -> 讲解
        llm = FakeLLM(responses=["unclear", "syntax", "澄清后的讲解……"])
        agent = _agent(llm=llm)
        out = agent.ask("嗯？", thread_id="t-clarify")
        self.assertIsInstance(out, ClarifyInterrupt)
        self.assertEqual(out.turns_used, 1)
        self.assertEqual(len(out.candidates), 3)
        self.assertIn("语法", out.candidates[0])

        final = agent.resume("我想问列表推导式怎么写", thread_id="t-clarify")
        self.assertIsInstance(final, AgentAnswer)
        self.assertEqual(final.intent, Intent.SYNTAX)
        self.assertIn("澄清后的讲解", final.content)

    def test_clarify_exhaustion_fuses(self):
        # max_clarify_turns=1：一次澄清后仍 unclear，应熔断并如实结束
        llm = FakeLLM(responses=["unclear", "unclear"])
        agent = _agent(max_clarify_turns=1, llm=llm)
        out = agent.ask("嗯？", thread_id="t-exhaust")
        self.assertIsInstance(out, ClarifyInterrupt)

        final = agent.resume("还是说不清楚", thread_id="t-exhaust")
        self.assertIsInstance(final, AgentAnswer)
        self.assertEqual(final.intent, Intent.UNCLEAR)
        self.assertIn("无法确定", final.content)

    def test_empty_reply_rejected(self):
        llm = FakeLLM(responses=["unclear"])
        agent = _agent(llm=llm)
        agent.ask("嗯？", thread_id="t-empty")
        with self.assertRaises(Exception):
            agent.resume("   ", thread_id="t-empty")


class TestRouteFunctions(unittest.TestCase):
    def test_route_after_intent_mapping(self):
        self.assertEqual(route_after_intent({"intent": "syntax"}), "syntax")
        self.assertEqual(route_after_intent({"intent": "env"}), "environment")
        self.assertEqual(route_after_intent({"intent": "review"}), "review")
        self.assertEqual(route_after_intent({"intent": "unclear"}), "clarify")
        # 未知 / 缺失意图一律进 clarify 兜底，绝不路由到不存在的节点
        self.assertEqual(route_after_intent({"intent": "garbage"}), "clarify")
        self.assertEqual(route_after_intent({}), "clarify")

    def test_route_after_clarify(self):
        self.assertEqual(
            route_after_clarify({"answer": "已熔断的话术"}), "answer"
        )
        self.assertEqual(route_after_clarify({"answer": ""}), "intent")


if __name__ == "__main__":
    unittest.main()
