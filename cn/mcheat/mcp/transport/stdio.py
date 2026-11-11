"""stdio 传输实现（M4）：把 MCP 服务器作为本地子进程拉起。

协议要点（规范 2025-11-25 Transports）：

* 子进程从 **stdin** 读 JSON-RPC、向 **stdout** 写 JSON-RPC，
  消息换行分隔、UTF-8，stdout 禁止出现任何非协议输出；
* 服务器日志应写 **stderr**——``stdio_client`` 的 ``errlog`` 参数默认
  ``sys.stderr``，即子进程 stderr 转发到当前进程 stderr，符合协议约定；
  框架仅在该行为之上记日志，不改道 stdout。
"""
from __future__ import annotations

import sys
from typing import IO

from .._utils import get_logger, require_module
from ..config import StdioServerConfig
from ..core.interfaces import Transport, TransportStreams
from ..exceptions import TransportError

logger = get_logger(__name__)

__all__ = ["StdioTransport"]


class StdioTransport(Transport):
    """基于官方 SDK ``stdio_client`` 的本地子进程通道。

    Args:
        config: stdio 服务器配置（command / args / env / cwd）。
        errlog: 子进程 stderr 的落点，默认 ``sys.stderr``（协议约定通道，
            不要改成 stdout）。
    """

    def __init__(self, config: StdioServerConfig, errlog: IO[str] | None = None) -> None:
        if not isinstance(config, StdioServerConfig):
            raise TransportError(
                f"StdioTransport 需要 StdioServerConfig，收到 {type(config).__name__}。"
                f"请检查配置里 transport 是否写成了 'stdio'。"
            )
        super().__init__(config)
        self._errlog = errlog or sys.stderr
        self._stack = None  # AsyncExitStack，open() 时创建
        self._streams: TransportStreams | None = None

    @property
    def is_open(self) -> bool:
        return self._streams is not None

    async def open(self) -> TransportStreams:
        if self._streams is not None:
            return self._streams  # 幂等
        require_module("mcp", "stdio 传输（本地子进程 MCP 服务器）")
        from contextlib import AsyncExitStack

        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        cfg = self.config
        params = StdioServerParameters(
            command=cfg.command,
            args=list(cfg.args),
            env=dict(cfg.env) if cfg.env else None,  # None = 继承全部父进程环境
            cwd=cfg.cwd,
        )
        logger.debug(
            "拉起 stdio 子进程：%s %s（服务器 %s）",
            cfg.command, " ".join(cfg.args), cfg.name,
        )
        stack = AsyncExitStack()
        try:
            # 本 SDK 版本 stdio_client yield (read_stream, write_stream) 二元组
            read, write = await stack.enter_async_context(
                stdio_client(params, errlog=self._errlog)
            )
        except Exception as exc:
            await stack.aclose()
            raise TransportError(
                f"无法为服务器 {cfg.name!r} 启动子进程 "
                f"（{' '.join([cfg.command, *cfg.args])}）：{exc}。"
                f"请检查 command 是否存在、args 路径是否正确。"
            ) from exc
        self._stack = stack
        self._streams = TransportStreams(reader=read, writer=write)
        return self._streams

    async def aclose(self) -> None:
        streams, stack = self._streams, self._stack
        self._streams, self._stack = None, None
        if stack is not None:
            # 逆序释放：先关会话层流、再终止子进程
            await stack.aclose()
        if streams is not None:
            logger.debug("stdio 通道已关闭（服务器 %s）", self.config.name)
