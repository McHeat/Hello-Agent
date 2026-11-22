"""test_01：配置加载、多级合并、环境变量覆盖与 validate 硬校验。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cn.mcheat.agent.config import AgentConfig
from cn.mcheat.agent.exceptions import ConfigError
from cn.mcheat.agent.core.registry import REQUIRED_COMPONENTS


class TestConfigLoad(unittest.TestCase):
    def test_defaults_load(self):
        cfg = AgentConfig.load()
        self.assertEqual(
            set(cfg.components), set(REQUIRED_COMPONENTS),
            "默认配置必须覆盖全部必需组件",
        )
        self.assertEqual(cfg.graph.max_clarify_turns, 2)
        self.assertEqual(cfg.graph.recursion_limit, 25)
        self.assertEqual(cfg.components["llm"].name, "langchain")
        # @依赖引用语法存在于默认配置
        self.assertEqual(
            cfg.components["intent_classifier"].params["llm"], "@llm"
        )

    def test_user_json_deep_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user.json"
            path.write_text(
                json.dumps({"graph": {"recursion_limit": 40}}), encoding="utf-8"
            )
            cfg = AgentConfig.load(path=path)
        self.assertEqual(cfg.graph.recursion_limit, 40)
        self.assertEqual(cfg.graph.max_clarify_turns, 2)  # 未覆盖项保留默认

    def test_env_overrides(self):
        env = {"AGENT_MAX_CLARIFY_TURNS": "3", "AGENT_ENV_TIMEOUT": "5.5"}
        with patch.dict("os.environ", env):
            cfg = AgentConfig.load()
        self.assertEqual(cfg.graph.max_clarify_turns, 3)
        self.assertEqual(cfg.env_probe.timeout, 5.5)

    def test_env_overrides_do_not_leak(self):
        with patch.dict("os.environ", {"AGENT_MAX_CLARIFY_TURNS": "3"}):
            AgentConfig.load()
        cfg = AgentConfig.load()  # 环境变量已移除，回落默认值
        self.assertEqual(cfg.graph.max_clarify_turns, 2)


class TestConfigValidate(unittest.TestCase):
    def _base(self, **graph_kwargs) -> AgentConfig:
        cfg = AgentConfig.load()
        for key, value in graph_kwargs.items():
            setattr(cfg.graph, key, value)
        return cfg

    def test_missing_components_rejected(self):
        cfg = AgentConfig()  # 空 components
        with self.assertRaises(ConfigError):
            cfg.validate()

    def test_zero_clarify_turns_rejected(self):
        cfg = self._base(max_clarify_turns=0)
        with self.assertRaises(ConfigError):
            cfg.validate()

    def test_recursion_limit_too_small_rejected(self):
        # 4 * (max_clarify_turns + 1) = 12 > 10，应被硬校验拦下
        cfg = self._base(max_clarify_turns=2, recursion_limit=10)
        with self.assertRaises(ConfigError):
            cfg.validate()

    def test_recursion_limit_out_of_range_rejected(self):
        cfg = self._base(recursion_limit=4)
        with self.assertRaises(ConfigError):
            cfg.validate()

    def test_bad_clarify_template_rejected(self):
        cfg = AgentConfig.load()
        cfg.clarify.question_template = "没有占位符的模板"
        with self.assertRaises(ConfigError):
            cfg.validate()

    def test_unknown_toplevel_key_rejected(self):
        with self.assertRaises(ConfigError):
            AgentConfig.load(overrides={"no_such_section": {}})

    def test_bad_env_type_rejected(self):
        with patch.dict("os.environ", {"AGENT_MAX_CLARIFY_TURNS": "abc"}):
            with self.assertRaises(ConfigError):
                AgentConfig.load()


if __name__ == "__main__":
    unittest.main()
