"""客户端示例 ②：多服务器聚合 + 失败隔离 + 按名路由。

两台 stdio 子进程（同一 demo 服务端，模拟不同部门提供的服务）+ 一台
故意坏掉的配置，演示：并发发现、单点失败隔离（failures 透出）、
跨服务器同名工具的限定名消歧、运行期 add_server。

运行（从 Hello-Agent 项目根目录）::

    python cn/mcheat/mcp/examples/client_multi.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]  # examples→mcp→mcheat→cn→Hello-Agent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cn.mcheat.mcp import MultiMCPManager, ToolNotFoundError  # noqa: E402

DEMO = ["cn/mcheat/mcp/examples/servers/demo_server.py"]


async def main() -> None:
    manager = MultiMCPManager({
        # 同一个服务端脚本跑两份，模拟「数学组」与「运维组」各一台服务器
        "math": {"transport": "stdio", "command": sys.executable, "args": DEMO},
        "ops": {"transport": "stdio", "command": sys.executable, "args": DEMO},
        # 故意配错：演示失败隔离（不影响 math / ops 的聚合）
        "broken": {"transport": "stdio", "command": sys.executable,
                   "args": ["/nonexistent/server.py"]},
    })

    try:
        # ① 并发发现 + 失败隔离
        tools = await manager.get_tools()
        print(f"聚合 {len(tools)} 个工具（来自 {manager.servers}）：")
        for t in tools:
            print(f"  - {t.qualified_name}")
        print(f"失败的服务器：{list(manager.failures)}（已隔离，不影响其余）")

        # ② 跨服务器同名：限定名消歧
        r = await manager.call_tool("math__add", {"a": 1, "b": 2})
        print(f"\nmath__add(1, 2) = {r.text()}")
        try:
            await manager.call_tool("add", {"a": 1, "b": 2})  # 短名在两台都有 → 多义
        except ToolNotFoundError as e:
            print(f"短名 add 多义（正确行为）：{e}")

        # ③ 运行期追加一台服务器（name 唯一即可）
        manager.add_server("extra", {"transport": "stdio",
                                     "command": sys.executable, "args": DEMO})
        tools = await manager.get_tools(refresh=True)
        print(f"\n追加 extra 后共 {len(tools)} 个工具，可用服务器：{manager.servers}")

        # ④ 直接 call_tool（无需先 get_tools 的懒路由演示）
        r = await manager.call_tool("extra__multiply", {"a": 8, "b": 9})
        print(f"extra__multiply(8, 9) = {r.text()}")
    finally:
        await manager.close()
        print("\n已关闭全部连接。")


if __name__ == "__main__":
    asyncio.run(main())
