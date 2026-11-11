"""聚合注册表（M7）单元测试：零依赖。

覆盖：覆盖式登记、限定名/短名路由、多义名消歧、资源模板匹配、清场。
"""
from __future__ import annotations

import sys

from cn.mcheat.mcp.exceptions import ToolNotFoundError
from cn.mcheat.mcp.models import (
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
)
from cn.mcheat.mcp.registry import AggregatedRegistry


def _tool(server: str, name: str) -> ToolDescriptor:
    return ToolDescriptor(server=server, name=name, description=f"{server}/{name}")


def _resource(server: str, uri: str, is_template: bool = False) -> ResourceDescriptor:
    return ResourceDescriptor(server=server, uri=uri, name=uri, is_template=is_template)


def test_register_and_lookup() -> None:
    r = AggregatedRegistry()
    r.register_tools("math", [_tool("math", "add"), _tool("math", "mul")])
    assert r.lookup_tool("math__add").name == "add"
    assert r.lookup_tool("add").name == "add"  # 唯一短名
    assert len(r.all_tools()) == 2


def test_register_overwrites() -> None:
    """toolsListChanged 后重新发现 = 覆盖式更新（旧条目失效）。"""
    r = AggregatedRegistry()
    r.register_tools("a", [_tool("a", "x")])
    r.register_tools("a", [_tool("a", "y")])
    assert [t.name for t in r.all_tools()] == ["y"]  # x 已失效


def test_ambiguous_short_name() -> None:
    r = AggregatedRegistry()
    r.register_tools("a", [_tool("a", "add")])
    r.register_tools("b", [_tool("b", "add")])
    try:
        r.lookup_tool("add")
    except ToolNotFoundError as e:
        assert "a__add" in str(e) and "b__add" in str(e)  # 错误列出候选
    else:
        raise AssertionError("应当抛 ToolNotFoundError")
    assert r.lookup_tool("b__add").server == "b"  # 限定名仍可路由


def test_resource_template_matching() -> None:
    r = AggregatedRegistry()
    r.register_resources("demo", [
        _resource("demo", "notes://list"),
        _resource("demo", "notes://{name}", is_template=True),
    ])
    assert r.lookup_resource("notes://list").server == "demo"
    assert r.lookup_resource("notes://layers").server == "demo"  # 模板通配匹配
    try:
        r.lookup_resource("notes://a/b/c")
    except ToolNotFoundError:
        pass
    else:
        raise AssertionError("多段不匹配单占位模板")


def test_discard() -> None:
    r = AggregatedRegistry()
    r.register_tools("a", [_tool("a", "x")])
    r.register_tools("b", [_tool("b", "x")])
    r.register_prompts("a", [PromptDescriptor(server="a", name="greet")])
    r.discard("a")
    assert [t.qualified_name for t in r.all_tools()] == ["b__x"]
    assert r.all_prompts() == []


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
