"""test_02：注册表契约、@依赖注入与 bootstrap 幂等。"""
from __future__ import annotations

import unittest

from cn.mcheat.agent.core import bootstrap as bootstrap_mod
from cn.mcheat.agent.core.bootstrap import bootstrap
from cn.mcheat.agent.core.interfaces import LLMClient, Skill
from cn.mcheat.agent.core.registry import (
    CHECKPOINTER_REGISTRY,
    ENV_PROBE_REGISTRY,
    KNOWLEDGE_BASE_REGISTRY,
    LLM_REGISTRY,
    Registry,
    resolve_references,
)
from cn.mcheat.agent.exceptions import ConfigError
from cn.mcheat.agent.models import Intent, SkillOutput


class _DummySkill(Skill):
    name = "dummy"

    def run(self, question, history=()):
        return SkillOutput(intent=Intent.SYNTAX, content="ok")


class TestRegistryContract(unittest.TestCase):
    def test_duplicate_registration_rejected(self):
        reg: Registry = Registry("test", None)
        reg.register("a", lambda: 1)
        with self.assertRaises(ConfigError):
            reg.register("a", lambda: 2)

    def test_unregistered_name_rejected(self):
        reg: Registry = Registry("test", None)
        with self.assertRaises(ConfigError):
            reg.create("nope")

    def test_contract_violation_rejected(self):
        reg: Registry[Skill] = Registry("skill", Skill)
        reg.register("not_a_skill", lambda: object())  # 工厂返回非 Skill
        with self.assertRaises(ConfigError):
            reg.create("not_a_skill")

    def test_valid_registration_and_create(self):
        reg: Registry[Skill] = Registry("skill", Skill)
        reg.register("dummy", _DummySkill)
        instance = reg.create("dummy")
        self.assertIsInstance(instance, _DummySkill)


class TestResolveReferences(unittest.TestCase):
    def test_reference_resolved(self):
        built = {"llm": object()}
        params = resolve_references({"llm": "@llm", "n": 3}, built)
        self.assertIs(params["llm"], built["llm"])
        self.assertEqual(params["n"], 3)

    def test_missing_reference_rejected(self):
        with self.assertRaises(ConfigError):
            resolve_references({"x": "@not_built"}, {})

    def test_nested_references(self):
        built = {"kb": object()}
        params = resolve_references(
            {"knowledge_base": "@kb", "opts": {"inner": ["@kb", 1]}}, built
        )
        self.assertIs(params["opts"]["inner"][0], built["kb"])


class TestBootstrap(unittest.TestCase):
    def test_builtin_components_registered(self):
        bootstrap()
        self.assertIn("langchain", LLM_REGISTRY)
        self.assertIn("memory", KNOWLEDGE_BASE_REGISTRY)
        self.assertIn("local", ENV_PROBE_REGISTRY)
        self.assertIn("memory", CHECKPOINTER_REGISTRY)

    def test_bootstrap_idempotent(self):
        bootstrap()
        self.assertTrue(bootstrap_mod._BOOTSTRAPPED)
        bootstrap()  # 第二次调用不应抛「重复注册」
        self.assertTrue(bootstrap_mod._BOOTSTRAPPED)


if __name__ == "__main__":
    unittest.main()
