"""M4 传输层（工厂 + stdio / streamable-http 实现）。

顶层 import 零 SDK 依赖（实现懒加载），安全用于无 mcp 环境。
"""
from __future__ import annotations

from .factory import create_transport, register_transport, transport_registry

__all__ = ["create_transport", "register_transport", "transport_registry"]
