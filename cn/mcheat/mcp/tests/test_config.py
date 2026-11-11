"""配置层（M1 / X1 / X2）单元测试：零 SDK 依赖。

覆盖：解析与字段、${VAR} 插值、四类校验失败、重复名、文件加载
（JSON / YAML / 不存在 / 坏后缀）、merge 优先级。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from cn.mcheat.mcp.config import (
    HttpServerConfig,
    MCPClientSettings,
    StdioServerConfig,
    expand_env,
    load_config,
)
from cn.mcheat.mcp.exceptions import ConfigError, DuplicateServerError

_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 解析与字段
# ---------------------------------------------------------------------------


def test_parse_stdio_and_http() -> None:
    s = MCPClientSettings.from_dict({
        "math": {"transport": "stdio", "command": "python", "args": ["a.py"], "env": {"K": "v"}},
        "remote": {"transport": "streamable-http", "url": "http://127.0.0.1:8000/mcp",
                   "headers": {"X": "y"}, "timeout": 5},
    })
    assert isinstance(s.servers["math"], StdioServerConfig)
    assert s.servers["math"].command == "python"
    assert s.servers["math"].env == {"K": "v"}
    assert isinstance(s.servers["remote"], HttpServerConfig)
    assert s.servers["remote"].timeout == 5.0


def test_top_level_servers_key() -> None:
    s = MCPClientSettings.from_dict({"servers": {"a": {"transport": "stdio", "command": "x"}}})
    assert "a" in s.servers


def test_env_expansion() -> None:
    import os

    os.environ["TEST_TOKEN"] = "tok123"
    out = expand_env({"h": {"Authorization": "Bearer ${TEST_TOKEN}"},
                      "u": "${TEST_URL:-http://fallback}"} )
    assert out["h"]["Authorization"] == "Bearer tok123"
    assert out["u"] == "http://fallback"


# ---------------------------------------------------------------------------
# 校验失败（每条都应带「哪个服务器 + 怎么修」）
# ---------------------------------------------------------------------------


def _expect_config_error(entry: dict) -> str:
    try:
        MCPClientSettings.from_dict({"bad": entry})
    except ConfigError as e:
        return str(e)
    raise AssertionError(f"应当抛 ConfigError：{entry}")


def test_validation_failures() -> None:
    assert "command" in _expect_config_error({"transport": "stdio"})
    assert "http://" in _expect_config_error({"transport": "streamable-http", "url": "ftp://x"})
    assert "transport" in _expect_config_error({"transport": "grpc"})
    assert "未知配置键" in _expect_config_error({"transport": "stdio", "command": "x", "typo": 1})
    assert "timeout" in _expect_config_error({"transport": "streamable-http",
                                              "url": "http://x/mcp", "timeout": -1})


def test_duplicate_name() -> None:
    s = MCPClientSettings.from_dict({"a": {"transport": "stdio", "command": "x"}})
    try:
        s.add_server("a", {"transport": "stdio", "command": "y"})
    except DuplicateServerError:
        pass
    else:
        raise AssertionError("应当抛 DuplicateServerError")


def test_merge_priority() -> None:
    base = MCPClientSettings.from_dict({"a": {"transport": "stdio", "command": "x"},
                                        "b": {"transport": "stdio", "command": "y"}})
    over = MCPClientSettings.from_dict({"a": {"transport": "stdio", "command": "z"},
                                        "c": {"transport": "stdio", "command": "w"}})
    base.merge(over)
    assert base.servers["a"].command == "z"   # 后者覆盖
    assert base.servers["c"].command == "w"   # 新增
    assert len(base) == 3


# ---------------------------------------------------------------------------
# 文件加载
# ---------------------------------------------------------------------------


def test_load_json_file() -> None:
    p = (_DIR.parent / "examples" / "config.example.json")
    s = load_config(p)
    assert set(s.servers) == {"math", "notes"}
    assert s.servers["notes"].headers["Authorization"].startswith("Bearer ")


def test_load_yaml_file() -> None:
    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as d:
        p = P(d) / "c.yaml"
        p.write_text("servers:\n  a:\n    transport: stdio\n    command: x\n", encoding="utf-8")
        s = load_config(p)
        assert s.servers["a"].command == "x"


def test_load_missing_file() -> None:
    try:
        load_config(_DIR / "nope.json")
    except ConfigError as e:
        assert "不存在" in str(e)
    else:
        raise AssertionError("应当抛 ConfigError")


def test_load_bad_suffix() -> None:
    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as d:
        p = P(d) / "c.toml"
        p.write_text("x = 1", encoding="utf-8")
        try:
            load_config(p)
        except ConfigError as e:
            assert "后缀" in str(e)
        else:
            raise AssertionError("应当抛 ConfigError")


def _run_all() -> int:
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
