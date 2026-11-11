"""单服务器连接封装（M5）：懒连接 + 会话生命周期 + 三大原语访问。

本文件是**唯一**做「官方 SDK 类型 ↔ 领域模型（models.py）」转换的地方
（防腐层，anti-corruption layer）：SDK 版本演进时只改这里，上层不动。

生命周期（对应架构时序图 ② ③ ④ 段）::

    ensure_connected()                    # 懒连接：Lock + 双重检查
        -> TransportFactory -> Transport.open()
        -> ClientSession + initialize()   # 能力协商（显式调用）
    list_tools / call_tool / ...          # 使用前自动确保连接
    close()                               # AsyncExitStack 逆序释放（幂等）

注意：本 SDK 版本的 ``ClientSession.__aenter__`` 只启动内部任务组，
**不**自动 initialize——故此处显式调用 ``session.initialize()``。
"""
from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack
from typing import Any

from .._utils import get_logger, require_module
from ..config import ServerConfig
from ..core.interfaces import ServerConnection, TransportFactory
from ..exceptions import (
    ServerUnreachableError,
    SessionClosedError,
    ToolCallError,
)
from ..models import (
    CallToolResult,
    ContentBlock,
    PromptArgument,
    PromptDescriptor,
    PromptMessage,
    ResourceContent,
    ResourceDescriptor,
    ServerInfo,
    ToolDescriptor,
)
from ..transport.factory import create_transport

logger = get_logger(__name__)

__all__ = ["MCPConnection"]


class MCPConnection(ServerConnection):
    """与单个 MCP 服务器的默认连接实现（基于官方 SDK ``ClientSession``）。

    Args:
        config: 服务器配置（stdio / streamable-http）。
        transport_factory: 传输工厂，默认 :func:`create_transport`
            （内置分发 + 支持自定义注册）；注入自定义工厂即可替换传输策略。
        list_cache_ttl: ``list_*`` 结果的缓存秒数，``0``（默认）不缓存。
            服务器侧会变更的部署建议保持 0，或在管理器层主动刷新。
        raise_on_error: ``call_tool`` 收到服务器 ``isError`` 结果时是否抛
            :class:`ToolCallError`（默认 True；False 时通过结果对象自判）。
    """

    def __init__(
        self,
        config: ServerConfig,
        transport_factory: TransportFactory | None = None,
        list_cache_ttl: float = 0.0,
        raise_on_error: bool = True,
    ) -> None:
        self._config = config
        self._transport_factory = transport_factory or create_transport
        self._cache_ttl = float(list_cache_ttl)
        self._raise_on_error = raise_on_error
        self._lock = asyncio.Lock()
        self._stack: AsyncExitStack | None = None
        self._session: Any = None  # mcp.ClientSession（Any：SDK 类型不外泄）
        self._server_info: ServerInfo | None = None
        self._closed = False
        self._cache: dict[str, tuple[float, Any]] = {}

    # -- 属性 -----------------------------------------------------------------

    @property
    def server_name(self) -> str:
        return self._config.name

    @property
    def config(self) -> ServerConfig:
        """本连接的服务器配置（只读视图）。"""
        return self._config

    @property
    def connected(self) -> bool:
        return self._session is not None and not self._closed

    @property
    def server_info(self) -> ServerInfo | None:
        """最近一次能力协商的结果（未连接过则为 ``None``）。"""
        return self._server_info

    # -- 生命周期 --------------------------------------------------------------

    async def ensure_connected(self) -> ServerInfo:
        """建立会话（懒连接入口，并发安全幂等）。

        Returns:
            能力协商得到的 :class:`ServerInfo`。

        Raises:
            ServerUnreachableError: 建立 / 协商失败。
            SessionClosedError: 连接已被 ``close()``，不允许复用。
        """
        if self.connected:
            assert self._server_info is not None
            return self._server_info
        if self._closed:
            raise SessionClosedError(self.server_name, "ensure_connected")
        async with self._lock:  # 双重检查：并发首次调用只建一次
            if self.connected:
                assert self._server_info is not None
                return self._server_info
            if self._closed:
                raise SessionClosedError(self.server_name, "ensure_connected")
            await self._connect()
            assert self._server_info is not None
            return self._server_info

    async def _connect(self) -> None:
        require_module("mcp", "MCP 客户端连接")
        from mcp import ClientSession

        name = self.server_name
        transport = self._transport_factory(self._config)
        stack = AsyncExitStack()
        try:
            streams = await stack.enter_async_context(transport)
            session = await stack.enter_async_context(
                ClientSession(streams.reader, streams.writer)
            )
            # 显式能力协商（见模块 docstring）。SDK 结果为 camelCase 的
            # pydantic 模型；提取也放进 try——任何失败都必须走清理路径。
            init_result = await session.initialize()
            impl = getattr(init_result, "serverInfo", None)
            caps = getattr(init_result, "capabilities", None)
            server_info = ServerInfo(
                name=str(getattr(impl, "name", None) or name),
                version=str(getattr(impl, "version", None) or "?"),
                protocol_version=str(getattr(init_result, "protocolVersion", None) or "?"),
                capabilities=(
                    caps.model_dump(exclude_none=True)
                    if hasattr(caps, "model_dump")
                    else dict(caps or {})
                ),
            )
        except ServerUnreachableError:
            await stack.aclose()
            raise
        except SessionClosedError:
            await stack.aclose()
            raise
        except Exception as exc:
            await stack.aclose()
            raise ServerUnreachableError(name, f"建立连接 / 能力协商失败：{exc}") from exc
        self._stack = stack
        self._session = session
        self._server_info = server_info
        self._cache.clear()
        logger.info(
            "已连接服务器 %s（协议 %s，能力：%s）",
            name, self._server_info.protocol_version,
            sorted(self._server_info.capabilities) or "(无)",
        )

    async def close(self) -> None:
        """优雅关闭（幂等）：AsyncExitStack 逆序释放 session → transport。"""
        stack, session = self._stack, self._session
        self._stack, self._session, self._server_info = None, None, None
        self._closed = True
        self._cache.clear()
        if stack is not None:
            await stack.aclose()
        if session is not None:
            logger.info("已断开服务器 %s", self.server_name)

    # -- 内部：操作前置检查与缓存 -----------------------------------------------

    def _require_session(self, operation: str) -> Any:
        if self._closed or self._session is None:
            # 未连接不算错误：懒连接语义下这里到不了（公开方法先 ensure），
            # 仅防御内部误用；真正关闭后的场景由公开方法抛 SessionClosedError。
            raise SessionClosedError(self.server_name, operation)
        return self._session

    async def _call(self, operation: str, coro_factory):
        """统一入口：确保连接（懒连接）→ 执行 → 单点失败隔离。"""
        if self._closed:
            raise SessionClosedError(self.server_name, operation)
        await self.ensure_connected()
        session = self._require_session(operation)
        try:
            return await coro_factory(session)
        except ServerUnreachableError:
            raise
        except (ToolCallError, SessionClosedError):
            raise
        except Exception as exc:  # 网络中断 / 超时 / 协议错误统一包装
            raise ServerUnreachableError(
                self.server_name, f"{operation} 执行失败：{exc}"
            ) from exc

    def _cached(self, key: str):
        if self._cache_ttl <= 0:
            return None
        hit = self._cache.get(key)
        if hit and (time.monotonic() - hit[0]) < self._cache_ttl:
            return hit[1]
        return None

    def _store_cache(self, key: str, value: Any) -> None:
        if self._cache_ttl > 0:
            self._cache[key] = (time.monotonic(), value)

    # -- Tools -----------------------------------------------------------------

    async def list_tools(self) -> list[ToolDescriptor]:
        cached = self._cached("tools")
        if cached is not None:
            return list(cached)

        async def run(session):
            result = await session.list_tools()
            return [
                _to_tool_descriptor(self.server_name, t) for t in (result.tools or [])
            ]

        tools = await self._call("list_tools", run)
        self._store_cache("tools", list(tools))
        return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        async def run(session):
            result = await session.call_tool(name, arguments or {})
            return _to_call_result(self.server_name, name, result)

        result = await self._call(f"tools/call({name})", run)
        if result.is_error and self._raise_on_error:
            raise ToolCallError(self.server_name, name, result.text() or "(无错误详情)")
        return result

    # -- Resources ---------------------------------------------------------------

    async def list_resources(self) -> list[ResourceDescriptor]:
        cached = self._cached("resources")
        if cached is not None:
            return list(cached)

        async def run(session):
            # 静态资源（list_resources）+ 资源模板（list_resource_templates，
            # FastMCP 的 @resource("notes://{name}") 注册在这里），合并返回。
            static = [
                _to_resource_descriptor(self.server_name, r)
                for r in ((await session.list_resources()).resources or [])
            ]
            try:
                templates = (await session.list_resource_templates()).resourceTemplates or []
            except Exception:
                templates = []  # 服务器未声明 resources 能力时容忍缺失
            static += [
                _to_resource_descriptor(self.server_name, t, is_template=True)
                for t in templates
            ]
            return static

        resources = await self._call("list_resources", run)
        self._store_cache("resources", list(resources))
        return resources

    async def read_resource(self, uri: str) -> list[ResourceContent]:
        async def run(session):
            result = await session.read_resource(uri)
            return [_to_resource_content(c) for c in (result.contents or [])]

        return await self._call(f"resources/read({uri})", run)

    # -- Prompts -----------------------------------------------------------------

    async def list_prompts(self) -> list[PromptDescriptor]:
        cached = self._cached("prompts")
        if cached is not None:
            return list(cached)

        async def run(session):
            result = await session.list_prompts()
            return [
                _to_prompt_descriptor(self.server_name, p)
                for p in (result.prompts or [])
            ]

        prompts = await self._call("list_prompts", run)
        self._store_cache("prompts", list(prompts))
        return prompts

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> list[PromptMessage]:
        async def run(session):
            result = await session.get_prompt(name, arguments or {})
            return [
                PromptMessage(
                    role=str(m.role),
                    text=str(getattr(m.content, "text", "") or ""),
                )
                for m in (result.messages or [])
            ]

        return await self._call(f"prompts/get({name})", run)

    # -- 便捷：async context manager ----------------------------------------------

    async def __aenter__(self) -> "MCPConnection":
        await self.ensure_connected()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# SDK 类型 → 领域模型转换（防腐层，唯一触碰 mcp.types 的地方）
# ---------------------------------------------------------------------------


def _to_tool_descriptor(server: str, tool: Any) -> ToolDescriptor:
    return ToolDescriptor(
        server=server,
        name=str(tool.name),
        description=str(getattr(tool, "description", "") or ""),
        title=getattr(tool, "title", None),
        input_schema=dict(getattr(tool, "inputSchema", None) or {}),
    )


def _to_content_block(content: Any) -> ContentBlock:
    ctype = str(getattr(content, "type", "unknown"))
    return ContentBlock(
        type=ctype,
        text=getattr(content, "text", None),
        data=getattr(content, "data", None) or getattr(content, "blob", None),
        mime_type=getattr(content, "mimeType", None),
    )


def _to_call_result(server: str, tool: str, result: Any) -> CallToolResult:
    structured = getattr(result, "structuredContent", None)
    return CallToolResult(
        server=server,
        tool=tool,
        content=[_to_content_block(c) for c in (result.content or [])],
        is_error=bool(getattr(result, "isError", False)),
        structured=dict(structured) if isinstance(structured, dict) else None,
    )


def _to_resource_descriptor(
    server: str, resource: Any, is_template: bool = False
) -> ResourceDescriptor:
    # 静态资源字段是 uri；资源模板字段是 uriTemplate（SDK camelCase 约定）
    uri = getattr(resource, "uri", None) or getattr(resource, "uriTemplate", None)
    return ResourceDescriptor(
        server=server,
        uri=str(uri),
        name=str(getattr(resource, "name", "") or ""),
        description=str(getattr(resource, "description", "") or ""),
        mime_type=getattr(resource, "mimeType", None),
        is_template=is_template,
    )


def _to_resource_content(content: Any) -> ResourceContent:
    return ResourceContent(
        uri=str(content.uri),
        text=getattr(content, "text", None),
        data=getattr(content, "blob", None),
        mime_type=getattr(content, "mimeType", None),
    )


def _to_prompt_descriptor(server: str, prompt: Any) -> PromptDescriptor:
    args = [
        PromptArgument(
            name=str(a.name),
            description=str(getattr(a, "description", "") or ""),
            required=bool(getattr(a, "required", False)),
        )
        for a in (getattr(prompt, "arguments", None) or [])
    ]
    return PromptDescriptor(
        server=server,
        name=str(prompt.name),
        description=str(getattr(prompt, "description", "") or ""),
        arguments=args,
    )
