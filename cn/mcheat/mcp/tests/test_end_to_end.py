"""端到端测试（需 mcp SDK）：拉起真实服务端子进程走完整协议链路。

覆盖：

* stdio 模式：发现 → 工具调用 → 资源（静态 + 模板）→ 提示 → 优雅关闭；
* streamable-http 模式：启动 demo_server --http → 远程连接 → 调用 → 关闭；
* 失败隔离与懒路由（不先 get_tools 直接 call_tool）；
* 关闭后操作防御（SessionClosedError）。

从项目根目录运行：python -m cn.mcheat.mcp.tests.test_end_to_end
"""
from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]  # tests→mcp→mcheat→cn→Hello-Agent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cn.mcheat.mcp import MultiMCPManager  # noqa: E402
from cn.mcheat.mcp.client import MCPConnection  # noqa: E402
from cn.mcheat.mcp.config import StdioServerConfig  # noqa: E402
from cn.mcheat.mcp.exceptions import SessionClosedError  # noqa: E402

DEMO_SERVER = str(_ROOT / "cn/mcheat/mcp/examples/servers/demo_server.py")


def _stdio_entry() -> dict:
    return {"transport": "stdio", "command": sys.executable, "args": [DEMO_SERVER]}


def _stdio_config(name: str = "demo") -> StdioServerConfig:
    return StdioServerConfig(
        name=name, transport="stdio", command=sys.executable, args=[DEMO_SERVER]
    )


# ---------------------------------------------------------------------------
# stdio 全链路
# ---------------------------------------------------------------------------


def test_stdio_full_cycle() -> None:
    async def run() -> None:
        mgr = MultiMCPManager({"demo": _stdio_entry()})
        try:
            tools = {t.qualified_name for t in await mgr.get_tools()}
            assert tools == {"demo__add", "demo__multiply", "demo__get_weather"}
            assert (await mgr.call_tool("demo__add", {"a": 2, "b": 3})).text() == "5.0"
            assert (await mgr.call_tool("multiply", {"a": 4, "b": 5})).text() == "20.0"

            resources = {r.uri for r in await mgr.list_resources()}
            assert "notes://list" in resources and "notes://{name}" in resources
            note = (await mgr.read_resource("notes://proto"))[0].text
            assert "MCP" in note

            prompts = {p.name for p in await mgr.list_prompts()}
            assert prompts == {"greeting", "review"}
            greet = (await mgr.get_prompt("greeting", {"name": "测试"}))[0].text
            assert "测试" in greet
        finally:
            await mgr.close()

    asyncio.run(run())


def test_failure_isolation_and_lazy_routing() -> None:
    async def run() -> None:
        mgr = MultiMCPManager({
            "good": _stdio_entry(),
            "bad": {"transport": "stdio", "command": sys.executable,
                    "args": ["/nonexistent.py"]},
        })
        try:
            tools = await mgr.get_tools()
            assert all(t.server == "good" for t in tools)
            assert "bad" in mgr.failures

            # 懒路由：新增 manager 直接 call_tool（未先 get_tools）
            result = await mgr.call_tool("good__add", {"a": 1, "b": 1})
            assert result.text() == "2.0"
        finally:
            await mgr.close()

    asyncio.run(run())


def test_connection_closed_semantics() -> None:
    async def run() -> None:
        conn = MCPConnection(_stdio_config())
        await conn.ensure_connected()
        info = conn.server_info
        assert info.protocol_version  # 协商出真实版本号
        await conn.close()
        await conn.close()  # 幂等
        try:
            await conn.call_tool("add", {"a": 1, "b": 2})
        except SessionClosedError:
            pass
        else:
            raise AssertionError("关闭后调用应抛 SessionClosedError")

    asyncio.run(run())


# ---------------------------------------------------------------------------
# streamable-http 全链路
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_http_full_cycle() -> None:
    async def run() -> None:
        port = _free_port()
        proc = await asyncio.create_subprocess_exec(
            sys.executable, DEMO_SERVER, "--http", "--port", str(port),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            # 等端口就绪（服务器启动需要一点时间）
            for _ in range(100):
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                        break
                except OSError:
                    await asyncio.sleep(0.1)
            else:
                raise AssertionError("HTTP 服务端未在 10s 内就绪")

            mgr = MultiMCPManager({
                "remote": {"transport": "streamable-http",
                           "url": f"http://127.0.0.1:{port}/mcp", "timeout": 10.0},
            })
            try:
                tools = await mgr.get_tools()
                assert any(t.name == "add" for t in tools)
                result = await mgr.call_tool("remote__add", {"a": 7, "b": 8})
                assert result.text() == "15.0"
                note = (await mgr.read_resource("notes://layers"))[0].text
                assert "数据层" in note
            finally:
                await mgr.close()
        finally:
            proc.terminate()
            await proc.wait()

    asyncio.run(run())


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
