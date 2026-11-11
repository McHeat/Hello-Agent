"""传输工厂（M4）与接口抽象（M2）单元测试：不建立真实连接。

覆盖：内置传输分发、配置类型不匹配防御、自定义传输注册（扩展点）、
HttpServerConfig 字段、UnsupportedTransportError 提示信息。
"""
from __future__ import annotations

import asyncio
import sys

from cn.mcheat.mcp.config import HttpServerConfig, StdioServerConfig
from cn.mcheat.mcp.core.interfaces import Transport, TransportStreams
from cn.mcheat.mcp.exceptions import TransportError, UnsupportedTransportError
from cn.mcheat.mcp.transport import create_transport, register_transport, transport_registry


def test_factory_dispatch_stdio() -> None:
    t = create_transport(StdioServerConfig(name="a", transport="stdio", command="python"))
    assert isinstance(t, Transport)


def test_factory_dispatch_http_aliases() -> None:
    for alias in ("streamable-http", "streamable_http", "http"):
        t = create_transport(HttpServerConfig(name="a", transport=alias,
                                              url="http://127.0.0.1:8000/mcp"))
        assert isinstance(t, Transport)


def test_factory_unknown_transport() -> None:
    try:
        create_transport(StdioServerConfig(name="a", transport="grpc", command="x"))
    except UnsupportedTransportError as e:
        assert "grpc" in str(e) and "core.interfaces.Transport" in str(e)
    else:
        raise AssertionError("应当抛 UnsupportedTransportError")


def test_factory_type_mismatch() -> None:
    # 声明 stdio 但给了 HTTP 配置对象 → 构造期防御
    try:
        create_transport(HttpServerConfig(name="a", transport="stdio",
                                          url="http://x/mcp"))
    except TransportError as e:
        assert "stdio" in str(e)
    else:
        raise AssertionError("应当抛 TransportError")


def test_custom_transport_registration() -> None:
    """扩展点：注册自定义传输后，配置里即可使用。"""

    class EchoTransport(Transport):
        def __init__(self, config):
            super().__init__(config)
            self.opened = False

        @property
        def is_open(self) -> bool:
            return self.opened

        async def open(self) -> TransportStreams:
            self.opened = True
            return TransportStreams(reader="r", writer="w")

        async def aclose(self) -> None:
            self.opened = False

    register_transport("echo", EchoTransport)
    try:
        t = create_transport(StdioServerConfig(name="a", transport="echo", command="x"))
        assert isinstance(t, EchoTransport)

        async def cycle():
            async with t as streams:  # __aenter__ 返回 TransportStreams
                assert streams.reader == "r"
                assert t.is_open
            assert not t.is_open  # __aexit__ 已释放

        asyncio.run(cycle())
    finally:
        register_transport.__globals__["_TRANSPORT_TYPES"].pop("echo", None)


def test_registry_snapshot() -> None:
    names = transport_registry()
    assert "stdio" in names and "streamable-http" in names


def _run_all() -> int:
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
