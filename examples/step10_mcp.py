"""Step 10：通过 MCP 让智能体调用外部工具。

MCP（Model Context Protocol，模型上下文协议）是 Anthropic 提出的开放标准，
用来把「工具 / 数据」以统一协议暴露给大模型。好处是：
- 工具提供方按协议实现一次服务端，任何兼容 MCP 的客户端（Claude、LangChain 等）都能直接用；
- 与具体厂商、具体框架解耦，不用为每个工具重写一遍适配代码。

本示例结构：
    mcp_demo_server.py  —— 一个 stdio 版 MCP 服务端，暴露 add / get_weather 两个工具
    step10_mcp.py        —— 本文件，作为 MCP 客户端，连接服务端、把远程工具挂到 agent 上

执行流程：
    START -> agent（模型决定直接回答 or 调用 MCP 工具）
         -> 有 tool_calls 走 tools 节点（真正远程调用 MCP 服务），否则 END
    tools 节点执行完 -> 回到 agent，形成 ReAct 循环

运行：
    pip install "mcp>=1.0" "langchain-mcp-adapters>=0.0.9"
    python examples/step10_mcp.py

（客户端会以子进程方式自动拉起 mcp_demo_server.py，无需手动启动服务端。）
"""

import asyncio
from pathlib import Path

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from llm import get_model


async def main() -> None:
    # 服务端脚本的绝对路径（与当前文件同目录）。
    server_path = str(Path(__file__).resolve().parent / "mcp_demo_server.py")

    # 用 MultiServerMCPClient 连接一个或多个 MCP 服务端。
    # 这里用 stdio 传输：客户端启动 `python mcp_demo_server.py` 子进程并与之通信。
    # 若连接远程/多服务，可加入更多键，名称（"demo"）任意，仅作标识。
    client = MultiServerMCPClient(
        {
            "demo": {
                "command": "python",
                "args": [server_path],
                "transport": "stdio",
            },
            # 若连接远程 MCP 服务，可用如下配置（需先启动服务端）：
    #         "remote": {
    #             "url": "https://your-mcp-server.example.com/mcp",  # 远端 MCP 端点
    #             "transport": "streamable_http",
    #             # 鉴权：把 token 放请求头（多数远程服务需要）
    #             "headers": {
    #                 "Authorization": f"Bearer {os.getenv('MCP_TOKEN')}",
    #             },
    #             "timeout": 30,  # 可选，超时秒数
    #             "sse_read_timeout": 60,  # 可选，仅 SSE 时有意义
    #         }
    #     },
        }
    )
    # get_tools() 会把远端暴露的工具拉取并转换成 LangChain 的 Tool 对象。
    # 此时 agent 完全不知道工具来自本地还是远程 MCP 服务——对上层透明。
    tools = await client.get_tools()
    print(f"已从 MCP 服务端加载工具：{[t.name for t in tools]}\n")

    # 把 MCP 工具挂到模型上，并用 LangGraph 预建的 ReAct agent 串成图。
    agent = create_agent(get_model(), tools)

    # 问一个需要调用远程工具的问题。
    question = "帮我用 MCP 工具算一下 12 加 30 等于多少？再查一下北京的天气。"
    result = await agent.ainvoke({"messages": question})

    for m in result["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
