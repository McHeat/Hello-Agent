"""MCP 集成框架示范服务端：math 工具 + notes 资源 + greeting 提示。

同时演示 ``MCPServerBuilder`` 的三种装饰器用法与双传输启动。

用法（从 Hello-Agent 项目根目录运行）::

    # stdio 模式：由宿主（MCP 客户端）作为子进程拉起，无需手动运行
    python cn/mcheat/mcp/examples/servers/demo_server.py --stdio

    # HTTP 模式：手动启动，客户端经 http://127.0.0.1:8000/mcp 连接
    python cn/mcheat/mcp/examples/servers/demo_server.py --http [--port 8000]
"""
from __future__ import annotations

import sys
from pathlib import Path

# 子进程模式下 sys.path[0] 是本脚本目录（…/examples/servers），
# 需把项目根（Hello-Agent）加入 path 才能导入 cn.mcheat.mcp。
_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cn.mcheat.mcp import MCPServerBuilder

server = MCPServerBuilder(
    "mcheat-demo",
    instructions="演示服务器：math 工具、notes 资源、greeting 提示模板。",
)

# --- Tools（模型可调用） -------------------------------------------------------


@server.tool
def add(a: float, b: float) -> float:
    """两个数相加。"""
    return a + b


@server.tool
def multiply(a: float, b: float) -> float:
    """两个数相乘。"""
    return a * b


@server.tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气（演示数据）。

    Args:
        city: 城市名称，例如 "北京"
    """
    return f"{city} 当前晴，气温 22℃，适合外出。"


# --- Resources（上下文数据源） ---------------------------------------------------

_NOTES = {
    "proto": "MCP：Model Context Protocol，Anthropic 开放标准，JSON-RPC 2.0。",
    "layers": "MCP 分数据层（JSON-RPC）与传输层（stdio / Streamable HTTP）。",
}


@server.resource("notes://{name}", name="get_note", mime_type="text/plain")
def get_note(name: str) -> str:
    """读取一条框架笔记（name: proto / layers）。"""
    return _NOTES.get(name, f"(没有叫 {name!r} 的笔记)")


@server.resource("notes://list", name="list_notes", mime_type="text/plain")
def list_notes() -> str:
    """列出全部可用笔记名。"""
    return "、".join(_NOTES)


# --- Prompts（交互模板） ---------------------------------------------------------


@server.prompt(name="greeting")
def greeting(name: str) -> str:
    """生成一段问候语（参数：name）。"""
    return f"你好，{name}！这是来自 mcheat MCP 框架的问候。"


@server.prompt
def review(code: str) -> str:
    """生成一段代码评审提示（参数：code）。"""
    return (
        "请评审以下代码，从正确性、可读性、异常处理三方面给出意见：\n"
        f"```\n{code}\n```"
    )


def main() -> None:
    args = sys.argv[1:]
    if "--http" in args:
        port = 8000
        if "--port" in args:
            port = int(args[args.index("--port") + 1])
        server.run("streamable-http", host="127.0.0.1", port=port)
    else:
        server.run("stdio")  # 默认 stdio（--stdio 可省略）


if __name__ == "__main__":
    main()
