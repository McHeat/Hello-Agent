"""全部**抽象接口 / 扩展点**（M2，``abc.ABC``，零第三方依赖）。

这是框架「高度抽象、可插拔」的核心：上层编排（``api``、``client/connection``、
``server/builder``）只依赖本文件的接口，不依赖任何
具体实现。想替换组件时，二选一（与 ``cn.mcheat.rag`` 的约定一致）：

1. 继承对应 ABC 实现新方法，然后注入：``MCPConnection(config,
   transport_factory=my_factory)`` / ``MultiMCPManager(connection_factory=...)``；
2. 自定义传输额外支持注册：``register_transport("grpc", MyTransport)``
   之后配置里就能写 ``"transport": "grpc"``。

接口清单（按数据流顺序）::

    ServerConfig（config.py）
        -> Transport.open() -> TransportStreams        [底层通道]
        -> ServerConnection.ensure_connected()          [会话生命周期]
        -> list_tools / call_tool / list_resources ...  [三大原语]
        -> CapabilityRegistry.register / lookup         [跨服务器聚合]
        -> MultiMCPManager（api.py，编排上面全部）

流类型刻意用 ``Any``：真实的 anyio 内存流类型属于官方 SDK 实现细节，
抽象层不 import SDK（保证本文件可被无 SDK 环境导入与单测）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from ..config import ServerConfig
from ..models import (
    CallToolResult,
    ContentBlock,
    PromptDescriptor,
    PromptMessage,
    ResourceContent,
    ResourceDescriptor,
    ServerInfo,
    ToolDescriptor,
)

__all__ = [
    "TransportStreams",
    "Transport",
    "TransportFactory",
    "ServerConnection",
    "ConnectionFactory",
    "CapabilityRegistry",
]


# ---------------------------------------------------------------------------
# 传输层抽象
# ---------------------------------------------------------------------------


class TransportStreams:
    """传输建立后的读写通道（reader / writer 为 SDK 内存流对象）。"""

    def __init__(self, reader: Any, writer: Any) -> None:
        self.reader = reader
        self.writer = writer


class Transport(ABC):
    """一条到 MCP 服务器的底层通道（stdio 子进程 / HTTP 连接）。

    生命周期：``open()`` 建立通道 → 反复收发 → ``aclose()`` 释放。
    同时实现 async context manager 协议，可直接挂到 ``AsyncExitStack``。
    实现方注意：``open()`` 幂等（已打开时直接返回），``aclose()`` 对未打开
    /已关闭状态安全。
    """

    def __init__(self, config: ServerConfig) -> None:
        self.config = config

    @property
    def server_name(self) -> str:
        """通道所属服务器名（取自配置）。"""
        return self.config.name

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """通道当前是否处于打开状态。"""

    @abstractmethod
    async def open(self) -> TransportStreams:
        """建立底层通道，返回读写流。

        Raises:
            TransportError: 无法建立通道（命令不存在 / 端点连不上等），
                消息需含服务器名与原因。
        """

    @abstractmethod
    async def aclose(self) -> None:
        """释放通道及全部底层资源（子进程 / HTTP 连接）。"""

    async def __aenter__(self) -> TransportStreams:
        return await self.open()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()


#: 传输工厂签名：``ServerConfig`` → ``Transport``。
TransportFactory = Callable[[ServerConfig], Transport]


# ---------------------------------------------------------------------------
# 连接抽象（会话生命周期 + 三大原语）
# ---------------------------------------------------------------------------


class ServerConnection(ABC):
    """与单个 MCP 服务器的会话封装：懒连接、原语访问、优雅关闭。

    实现方注意：

    * ``ensure_connected()`` 必须并发安全（内部加锁、双重检查）；
    * 全部 ``list_*`` / ``call_*`` / ``read_*`` / ``get_*`` 方法在调用时
      **自动确保已连接**（触发懒连接）；
    * ``close()`` 幂等；关闭后继续调用应抛 :class:`SessionClosedError`。
    """

    @property
    @abstractmethod
    def server_name(self) -> str:
        """服务器名（与配置 key 一致，用于聚合命名与错误定位）。"""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """会话当前是否处于已建立状态。"""

    @abstractmethod
    async def ensure_connected(self) -> ServerInfo:
        """建立会话（懒连接入口；已连接时直接返回服务器信息）。

        Returns:
            ``initialize`` 能力协商得到的 :class:`ServerInfo`。

        Raises:
            ServerUnreachableError: 建立失败（含原因）。
        """

    # -- Tools ---------------------------------------------------------------

    @abstractmethod
    async def list_tools(self) -> list[ToolDescriptor]:
        """拉取工具清单（``tools/list``）。"""

    @abstractmethod
    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        """调用远端工具（``tools/call``）。

        Args:
            name: 工具原始名（不带服务器前缀）。
            arguments: 符合 inputSchema 的参数。

        Raises:
            ToolCallError: 服务器报告执行错误且 ``raise_on_error`` 开启。
        """

    # -- Resources -----------------------------------------------------------

    @abstractmethod
    async def list_resources(self) -> list[ResourceDescriptor]:
        """拉取资源清单（``resources/list``）。"""

    @abstractmethod
    async def read_resource(self, uri: str) -> list[ResourceContent]:
        """读取指定 URI 的资源内容（``resources/read``）。"""

    # -- Prompts -------------------------------------------------------------

    @abstractmethod
    async def list_prompts(self) -> list[PromptDescriptor]:
        """拉取提示模板清单（``prompts/list``）。"""

    @abstractmethod
    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> list[PromptMessage]:
        """渲染提示模板为消息列表（``prompts/get``）。"""

    @abstractmethod
    async def close(self) -> None:
        """优雅关闭会话与底层通道（幂等）。"""


#: 连接工厂签名：``ServerConfig`` → ``ServerConnection``（管理器注入用）。
ConnectionFactory = Callable[[ServerConfig], ServerConnection]


# ---------------------------------------------------------------------------
# 聚合注册表抽象
# ---------------------------------------------------------------------------


class CapabilityRegistry(ABC):
    """跨服务器的工具 / 资源 / 提示聚合索引（内存态，无 IO）。

    职责：登记发现结果、按限定名路由到来源服务器、处理跨服务器同名冲突。
    实现方注意：登记同名工具（同 server 同 name）为**覆盖更新**
    （服务器侧 toolsListChanged 后重新发现就是覆盖语义）；
    跨服务器撞名由 ``qualified_name`` 天然消歧，不需报错。
    """

    @abstractmethod
    def register_tools(self, server: str, tools: list[ToolDescriptor]) -> None:
        """登记（覆盖）某服务器的工具集合。"""

    @abstractmethod
    def register_resources(self, server: str, resources: list[ResourceDescriptor]) -> None:
        """登记（覆盖）某服务器的资源集合。"""

    @abstractmethod
    def register_prompts(self, server: str, prompts: list[PromptDescriptor]) -> None:
        """登记（覆盖）某服务器的提示集合。"""

    @abstractmethod
    def lookup_tool(self, qualified_name: str) -> ToolDescriptor:
        """按限定名查工具。

        Raises:
            ToolNotFoundError: 不存在（错误信息含可用工具预览）。
        """

    @abstractmethod
    def lookup_resource(self, uri: str) -> ResourceDescriptor:
        """按 URI 查资源描述符。

        Raises:
            ToolNotFoundError: 不存在。
        """

    @abstractmethod
    def lookup_prompt(self, name: str) -> PromptDescriptor:
        """按限定名查提示。

        Raises:
            ToolNotFoundError: 不存在。
        """

    @abstractmethod
    def all_tools(self) -> list[ToolDescriptor]:
        """全部已登记工具（快照副本）。"""

    @abstractmethod
    def all_resources(self) -> list[ResourceDescriptor]:
        """全部已登记资源（快照副本）。"""

    @abstractmethod
    def all_prompts(self) -> list[PromptDescriptor]:
        """全部已登记提示（快照副本）。"""

    @abstractmethod
    def discard(self, server: str) -> None:
        """移除某服务器的全部登记（连接失败 / 关闭时清场）。"""
