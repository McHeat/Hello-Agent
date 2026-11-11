"""Streamable HTTP 传输实现（M4）：连接远程 MCP 服务。

协议要点（规范 2025-11-25 Transports）：

* 单一 MCP 端点（如 ``https://example.com/mcp``），POST 发消息、
  可选 GET 开 SSE 流接收服务器推送；
* 会话由服务器经 ``MCP-Session-Id`` 响应头分配，客户端后续请求必须带上——
  这些细节由官方 SDK ``streamablehttp_client`` 处理，本类只暴露会话 id 查询；
* 鉴权走标准 HTTP 头（``Authorization`` 等），token 建议在配置里用
  ``${VAR}`` 插值，避免明文落盘（见 config.py）。
"""
from __future__ import annotations

from typing import Callable

from .._utils import get_logger, require_module
from ..config import HttpServerConfig
from ..core.interfaces import Transport, TransportStreams
from ..exceptions import TransportError

logger = get_logger(__name__)

__all__ = ["StreamableHttpTransport"]


class StreamableHttpTransport(Transport):
    """基于官方 SDK ``streamablehttp_client`` 的远程 HTTP 通道。

    Args:
        config: streamable-http 服务器配置（url / headers / timeout）。
    """

    def __init__(self, config: HttpServerConfig) -> None:
        if not isinstance(config, HttpServerConfig):
            raise TransportError(
                f"StreamableHttpTransport 需要 HttpServerConfig，"
                f"收到 {type(config).__name__}。"
                f"请检查配置里 transport 是否写成了 'streamable-http'。"
            )
        super().__init__(config)
        self._stack = None
        self._streams: TransportStreams | None = None
        self._get_session_id: Callable[[], str | None] | None = None

    @property
    def is_open(self) -> bool:
        return self._streams is not None

    @property
    def session_id(self) -> str | None:
        """服务器分配的 ``MCP-Session-Id``（未连接时为 ``None``）。"""
        if self._get_session_id is None:
            return None
        try:
            return self._get_session_id()
        except Exception:  # SDK 在无会话时可能抛底层异常
            return None

    async def open(self) -> TransportStreams:
        if self._streams is not None:
            return self._streams  # 幂等
        require_module("mcp", "streamable-http 传输（远程 MCP 服务器）")
        from contextlib import AsyncExitStack

        from mcp.client.streamable_http import streamablehttp_client

        cfg = self.config
        logger.debug("连接远程 MCP 端点：%s（服务器 %s）", cfg.url, cfg.name)
        stack = AsyncExitStack()
        try:
            read, write, get_session_id = await stack.enter_async_context(
                streamablehttp_client(
                    url=cfg.url,
                    headers=dict(cfg.headers) if cfg.headers else None,
                    timeout=cfg.timeout,
                )
            )
        except Exception as exc:
            await stack.aclose()
            raise TransportError(
                f"无法连接服务器 {cfg.name!r} 的端点 {cfg.url}：{exc}。"
                f"请检查 URL 与网络，确认远端已启动且为 streamable-http 模式。"
            ) from exc
        self._stack = stack
        self._streams = TransportStreams(reader=read, writer=write)
        self._get_session_id = get_session_id
        return self._streams

    async def aclose(self) -> None:
        streams, stack = self._streams, self._stack
        self._streams, self._stack, self._get_session_id = None, None, None
        if stack is not None:
            # terminate_on_close=True（SDK 默认）：发 HTTP DELETE 结束会话
            await stack.aclose()
        if streams is not None:
            logger.debug("http 通道已关闭（服务器 %s）", self.config.name)
