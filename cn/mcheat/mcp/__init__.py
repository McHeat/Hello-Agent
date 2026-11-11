"""可插拔 MCP 集成框架：配置驱动的多服务器接入与聚合。

在官方 ``mcp`` SDK 之上构建集成层，解决「一个应用同时接入多个 MCP 服务器」
的工程问题：配置化、注册表、连接生命周期、跨服务器聚合与路由。

分层与零依赖约定（详见 ``doc/mcp/01-architecture.md``）：

====== ====================== =====================================
编号   模块                     依赖
====== ====================== =====================================
X1     ``exceptions.py``        无
X2     ``_utils.py``            无
M1     ``config.py``            无（YAML 可选）
M2     ``core/interfaces.py``   无（全部抽象接口）
M3     ``models.py``            无
M4     ``transport/``           mcp（stdio / streamable-http）
M5     ``client/connection.py`` mcp
M6     ``api.py``               mcp（仅类型）
M7     ``registry/tools.py``    无
M8     ``server/builder.py``    mcp
====== ====================== =====================================

快速开始（客户端）::

    from cn.mcheat.mcp import MultiMCPManager, load_config

    manager = MultiMCPManager(load_config("config.json"))
    tools = await manager.get_tools()          # 并发发现，聚合统一清单
    result = await manager.call_tool("add", {"a": 1, "b": 2})
    await manager.close()                      # 优雅关闭全部连接

快速开始（服务端）::

    from cn.mcheat.mcp import MCPServerBuilder

    server = MCPServerBuilder("demo")

    @server.tool
    def add(a: float, b: float) -> float:
        \"\"\"两数相加。\"\"\"
        return a + b

    server.run("stdio")   # 或 "streamable-http"

从项目根目录（Hello-Agent）运行，保证 ``cn.mcheat.mcp`` 可被导入。
完整说明见包内 ``README.md``，架构总纲见项目 ``doc/mcp/01-architecture.md``。
"""
from __future__ import annotations

from ._utils import get_logger
from .api import MultiMCPManager
from .client import MCPConnection
from .config import (
    HttpServerConfig,
    MCPClientSettings,
    StdioServerConfig,
    expand_env,
    load_config,
)
from .core.interfaces import (
    CapabilityRegistry,
    ConnectionFactory,
    ServerConnection,
    Transport,
    TransportFactory,
)
from .exceptions import (
    ConfigError,
    DiscoveryError,
    MCPError,
    ServerBuildError,
    ServerUnreachableError,
    SessionClosedError,
    ToolCallError,
    ToolNotFoundError,
    TransportError,
    UnsupportedTransportError,
)
from .models import (
    CallToolResult,
    ContentBlock,
    PromptDescriptor,
    PromptMessage,
    ResourceDescriptor,
    ServerInfo,
    ToolDescriptor,
)
from .registry import AggregatedRegistry
from .server.builder import MCPServerBuilder
from .transport import create_transport, register_transport

__version__ = "1.0.0"

__all__ = [
    # 门面
    "MultiMCPManager", "MCPConnection", "MCPServerBuilder",
    # 配置
    "load_config", "MCPClientSettings", "StdioServerConfig", "HttpServerConfig",
    "expand_env",
    # 领域模型
    "ToolDescriptor", "ResourceDescriptor", "PromptDescriptor", "PromptMessage",
    "CallToolResult", "ContentBlock", "ServerInfo",
    # 抽象接口（扩展点）
    "Transport", "ServerConnection", "CapabilityRegistry",
    "TransportFactory", "ConnectionFactory",
    # 传输注册
    "create_transport", "register_transport", "AggregatedRegistry",
    # 异常
    "MCPError", "ConfigError", "TransportError", "UnsupportedTransportError",
    "ServerUnreachableError", "SessionClosedError", "DiscoveryError",
    "ToolNotFoundError", "ToolCallError", "ServerBuildError",
    # 工具
    "get_logger", "__version__",
]
