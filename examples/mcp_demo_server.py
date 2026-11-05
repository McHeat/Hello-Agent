"""MCP 演示服务端（stdio 传输）。

这是一个最小可运行的 MCP（Model Context Protocol）服务器，通过标准输入/输出
与客户端进程通信。它对外暴露两个工具，供客户端（大模型 agent）远程调用。

运行方式（通常无需手动运行，客户端会作为子进程拉起它）：
    python examples/mcp_demo_server.py

依赖：pip install "mcp>=1.0"
"""

from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务实例，名字随意，用于标识服务。
mcp = FastMCP("hello-agent-mcp-demo")


@mcp.tool()
def add(a: float, b: float) -> float:
    """两个数相加。当需要进行加法运算时使用。"""
    return a + b


@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的当前天气（演示数据）。

    Args:
        city: 城市名称，例如 "北京"
    """
    # 真实场景这里会调用气象 API；这里返回固定演示数据。
    return f"{city} 当前晴，气温 22℃，适合外出。"


if __name__ == "__main__":
    # 以 stdio 方式启动服务，等待客户端通过标准输入/输出通信。
    mcp.run(transport="stdio")
