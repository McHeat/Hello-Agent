"""传输工厂（M4）：按配置分发到具体传输实现，并支持自定义传输注册。

扩展点（对应 ``core/interfaces.Transport``）::

    from cn.mcheat.mcp.transport import register_transport
    from cn.mcheat.mcp.core.interfaces import Transport

    class GrpcTransport(Transport):
        ...

    register_transport("grpc", GrpcTransport)
    # 此后配置里即可写 {"transport": "grpc", ...}

内置传输（懒导入，import 本模块零 SDK 依赖）：

====== ====================== =============================================
名字     实现类                   配置类
====== ====================== =============================================
stdio   StdioTransport          StdioServerConfig
====== ====================== =============================================
streamable-http / streamable_http / http
        StreamableHttpTransport  HttpServerConfig
====== ====================== =============================================
"""
from __future__ import annotations

from .._utils import get_logger
from ..config import HttpServerConfig, ServerConfig, StdioServerConfig
from ..core.interfaces import Transport, TransportFactory
from ..exceptions import TransportError, UnsupportedTransportError

logger = get_logger(__name__)

__all__ = ["create_transport", "register_transport", "transport_registry"]

# 名字 → 工厂。值刻意用「懒函数」而不是类本身，避免 import 本模块时连带
# import 官方 SDK（stdio.py / http.py 顶部也不 import SDK，实现在 open() 时才加载）。
_TRANSPORT_TYPES: dict[str, TransportFactory] = {}

#: 配置里 streamable-http 的常见别名。
_HTTP_ALIASES = ("streamable-http", "streamable_http", "http")


def _build_stdio(config: ServerConfig) -> Transport:
    from .stdio import StdioTransport

    return StdioTransport(config)  # type: ignore[arg-type]  # 构造内做类型校验


def _build_http(config: ServerConfig) -> Transport:
    from .http import StreamableHttpTransport

    return StreamableHttpTransport(config)  # type: ignore[arg-type]


def _register_defaults() -> None:
    _TRANSPORT_TYPES.update(
        {
            "stdio": _build_stdio,
            "streamable-http": _build_http,
            "streamable_http": _build_http,
            "http": _build_http,
        }
    )


_register_defaults()


def register_transport(transport: str, factory: TransportFactory) -> None:
    """注册自定义传输（扩展点）。

    Args:
        transport: 传输类型名（配置 ``transport`` 字段的取值）。
        factory: 工厂，``ServerConfig`` → ``Transport``。注册内置名时覆盖默认。
    """
    _TRANSPORT_TYPES[transport] = factory
    logger.info("已注册自定义传输类型 %r -> %s", transport, factory)


def transport_registry() -> dict[str, TransportFactory]:
    """当前全部已注册传输的快照（名称 → 工厂）。"""
    return dict(_TRANSPORT_TYPES)


def create_transport(config: ServerConfig) -> Transport:
    """按配置分发传输实现（工厂入口）。

    Args:
        config: 服务器配置（含 ``transport`` 字段）。

    Returns:
        尚未 ``open()`` 的 :class:`Transport` 实例。

    Raises:
        UnsupportedTransportError: 类型名未注册。
        TransportError: 配置类型与传输实现不匹配。
    """
    key = config.transport
    factory = _TRANSPORT_TYPES.get(key)
    if factory is None:
        raise UnsupportedTransportError(key, config.name)
    if key == "stdio" and not isinstance(config, StdioServerConfig):
        raise TransportError(
            f"服务器 {config.name!r} 声明 stdio 传输但配置字段不全"
            f"（需要 command/args），收到 {type(config).__name__}。"
        )
    if key in _HTTP_ALIASES and not isinstance(config, HttpServerConfig):
        raise TransportError(
            f"服务器 {config.name!r} 声明 {key} 传输但配置字段不全"
            f"（需要 url/headers），收到 {type(config).__name__}。"
        )
    return factory(config)
