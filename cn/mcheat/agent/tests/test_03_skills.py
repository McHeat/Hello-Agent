"""test_03：三个技能分支的输出结构、知识命中/未命中与探针降级。"""
from __future__ import annotations

import platform
import unittest

from cn.mcheat.agent.models import Intent
from cn.mcheat.agent.skills import EnvSkill, ReviewSkill, SyntaxSkill
from cn.mcheat.agent.tools import InMemorySyntaxKnowledgeBase, LocalEnvProbe
from cn.mcheat.agent.tools.env_probe import _run_cmd

from ._fakes import FakeLLM, StubEnvProbe, StubKnowledgeBase


class TestSyntaxSkill(unittest.TestCase):
    def test_hit_uses_llm_with_kb_content(self):
        llm = FakeLLM(responses=["这是讲解。"])
        skill = SyntaxSkill(llm=llm, knowledge_base=StubKnowledgeBase(hit=True))
        out = skill.run("列表推导式怎么写？")
        self.assertEqual(out.content, "这是讲解。")
        self.assertEqual(out.intent, Intent.SYNTAX)
        self.assertEqual(out.matched_topic, "桩主题")
        self.assertEqual(out.tool, "search_python_syntax")
        # 知识库内容必须进入 LLM 的 prompt（只依据资料作答）
        self.assertIn("桩知识内容", llm.calls[0]["prompt"])

    def test_miss_is_honest_without_llm_call(self):
        llm = FakeLLM()
        skill = SyntaxSkill(llm=llm, knowledge_base=StubKnowledgeBase(hit=False))
        out = skill.run("与主题无关的问题")
        self.assertFalse(out.matched_topic)
        self.assertIn("没有找到", out.content)
        self.assertIn("主题A", out.content)  # 推荐可问主题
        self.assertEqual(llm.calls, [], "未命中时不应调用 LLM 编造答案")


class TestEnvSkill(unittest.TestCase):
    def test_probe_facts_feed_llm(self):
        llm = FakeLLM(responses=["环境总结"])
        probe = StubEnvProbe()
        skill = EnvSkill(llm=llm, env_probe=probe)
        out = skill.run("帮我看看我的 Python 环境")
        self.assertEqual(out.content, "环境总结")
        self.assertEqual(out.intent, Intent.ENV)
        self.assertEqual(probe.call_count, 1)
        self.assertIn("3.13.1", llm.calls[0]["prompt"])  # 探针事实进入 prompt
        self.assertEqual(out.details["python_version"], "3.13.1")


class TestReviewSkill(unittest.TestCase):
    def test_three_part_review(self):
        llm = FakeLLM(responses=["## 1. 问题定位\n..."])
        skill = ReviewSkill(llm=llm)
        out = skill.run("def add(a, b):\n    return a + b\n 帮我看看")
        self.assertEqual(out.intent, Intent.REVIEW)
        self.assertEqual(out.content, "## 1. 问题定位\n...")


class TestLocalEnvProbe(unittest.TestCase):
    def test_probe_reflects_current_interpreter(self):
        probe = LocalEnvProbe(include_pip_list=False)
        report = probe.probe()
        self.assertEqual(report.python_version, platform.python_version())
        self.assertTrue(report.executable)
        # pip 探测：要么可用要么记入 issues，两种结果都不允许抛异常
        self.assertTrue(report.pip_available or report.issues)

    def test_missing_command_degrades_gracefully(self):
        out, err = _run_cmd(["no-such-command-xyz"], timeout=2.0)
        self.assertEqual(out, "")
        self.assertIn("命令不存在", err)

    def test_timeout_degrades_gracefully(self):
        out, err = _run_cmd(["sleep", "5"], timeout=0.1)
        self.assertEqual(out, "")
        self.assertIn("超时", err)


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = InMemorySyntaxKnowledgeBase()

    def test_exact_alias_match(self):
        hit = self.kb.search("列表推导式怎么写")
        self.assertTrue(hit.matched)
        self.assertEqual(hit.title, "列表推导式")

    def test_english_alias_match(self):
        hit = self.kb.search("how does list comprehension work?")
        self.assertTrue(hit.matched)
        self.assertEqual(hit.topic, "list_comprehension")

    def test_longest_alias_wins(self):
        hit = self.kb.search("字典推导式是什么")
        self.assertEqual(hit.topic, "dict_comprehension")

    def test_miss_returns_suggestions(self):
        hit = self.kb.search("今天天气怎么样")
        self.assertFalse(hit.matched)
        self.assertIn("装饰器", hit.suggestions)

    def test_default_kb_topic_count(self):
        self.assertEqual(len(self.kb.topics), 16)


if __name__ == "__main__":
    unittest.main()
