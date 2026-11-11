"""客户端示例 ①：单服务器 stdio 模式。

从配置文件（或 dict）加载 → 懒连接发现 → 调用工具 / 读资源 / 渲染提示
→ 优雅关闭。演示 `MultiMCPManager` 的完整生命周期。

运行（从 Hello-Agent 项目根目录）::

    python cn/mcheat/mcp/examples/client_stdio.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 保证项目根可导入（与 demo_server.py 同样的子进程 path 处理）
_ROOT = Path(__file__).resolve().parents[4]  # examples→mcp→mcheat→cn→Hello-Agent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cn.mcheat.mcp import MultiMCPManager, ToolDescriptor  # noqa: E402


def print_tools(tools: list[ToolDescriptor]) -> None:
    print("发现工具：")
    for t in tools:
        print(f"  - {t.qualified_name}: {t.description or '(无描述)'}")


async def main() -> None:
    # ① 配置驱动：也可以 load_config("cn/mcheat/mcp/examples/config.example.json")
    manager = MultiMCPManager({
        "demo": {
            "transport": "stdio",
            "command": sys.executable,  # 保证用当前解释器拉起子进程
            "args": ["cn/mcheat/mcp/examples/servers/demo_server.py"],
        }
    })

    try:
        # ② 懒发现：首次 get_tools() 才建立连接
        tools = await manager.get_tools()
        print_tools(tools)

        # ③ 调用：限定名（demo__add）或唯一短名（add）均可
        r = await manager.call_tool("demo__add", {"a": 20, "b": 22})
        print(f"\ndemo__add(20, 22) = {r.text()}")
        r = await manager.call_tool("multiply", {"a": 6, "b": 7})
        print(f"multiply(6, 7) = {r.text()}")

        # ④ 资源与提示（三大原语的另外两块）
        contents = await manager.read_resource("notes://layers")
        print(f"\nnote(layers) = {contents[0].text}")
        messages = await manager.get_prompt("greeting", {"name": "MCP"})
        print(f"greeting = {messages[0].text}")

        # ⑤ OpenAI function calling 格式：可直接交给支持 tools 的模型
        openai_tools = [t.to_openai_tool() for t in tools]
        print(f"\nOpenAI 工具定义（{len(openai_tools)} 个），第 1 个：")
        print(f"  {openai_tools[0]}")
    finally:
        # ⑥ 优雅关闭：幂等，异常路径也不泄漏子进程
        await manager.close()
        print("\n已关闭全部连接。")


if __name__ == "__main__":
    asyncio.run(main())
