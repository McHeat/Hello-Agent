"""test_05：端到端 —— ask/interrupt/resume 全链路、多轮记忆与错误透传。"""
from __future__ import annotations

import unittest

from cn.mcheat.agent import AgentAnswer, PythonTutorAgent
from cn.mcheat.agent.exceptions import AgentError, LlmError

from ._fakes import FakeLLM


class TestEndToEnd(unittest.TestCase):
    def test_full_loop_syntax(self):
        llm = FakeLLM(responses=["syntax", "列表推导式讲解。"])
        agent = PythonTutorAgent(llm=llm)
        out = agent.ask("讲讲列表推导式", thread_id="e2e-1")
        self.assertIsInstance(out, AgentAnswer)
        self.assertEqual(out.content, "列表推导式讲解。")
        # 两次 LLM 调用：意图分类 + 技能生成
        self.assertEqual(len(llm.calls), 2)

    def test_multi_turn_memory_accumulates(self):
        llm = FakeLLM(responses=["syntax", "答1", "review", "答2"])
        agent = PythonTutorAgent(llm=llm)
        agent.ask("列表推导式", thread_id="e2e-mem")
        agent.ask("帮我看看这段代码", thread_id="e2e-mem")
        history = agent.history(thread_id="e2e-mem")
        self.assertEqual(len(history), 2, "同一 thread 的消息应跨轮累积")

    def test_independent_threads_isolated(self):
        llm = FakeLLM(responses=["syntax", "答A", "syntax", "答B"])
        agent = PythonTutorAgent(llm=llm)
        agent.ask("列表推导式", thread_id="e2e-a")
        agent.ask("装饰器", thread_id="e2e-b")
        self.assertEqual(len(agent.history("e2e-a")), 1)
        self.assertEqual(len(agent.history("e2e-b")), 1)

    def test_interrupt_resume_across_calls(self):
        # 完整 human-in-the-loop 链路：ask 暂停 -> 展示 -> resume 继续
        llm = FakeLLM(responses=["unclear", "env", "环境诊断完成。"])
        agent = PythonTutorAgent(llm=llm)
        outcome = agent.ask("呃", thread_id="e2e-hitl")
        if not isinstance(outcome, AgentAnswer):  # ClarifyInterrupt
            self.assertIn("想要", outcome.question)
            outcome = agent.resume("2", thread_id="e2e-hitl")
        self.assertIsInstance(outcome, AgentAnswer)
        self.assertEqual(outcome.intent.value, "env")
        self.assertIn("环境诊断完成", outcome.content)

    def test_llm_error_propagates(self):
        # 分类成功但技能执行时 LLM 故障：LlmError 应原类型透出（不被吞掉）
        llm = FakeLLM(responses=["syntax"], error_when_exhausted=True)
        agent = PythonTutorAgent(llm=llm)
        with self.assertRaises(LlmError):
            agent.ask("列表推导式怎么写", thread_id="e2e-err")

    def test_empty_question_rejected(self):
        agent = PythonTutorAgent(llm=FakeLLM())
        with self.assertRaises(AgentError):
            agent.ask("   ", thread_id="e2e-empty")


if __name__ == "__main__":
    unittest.main()
