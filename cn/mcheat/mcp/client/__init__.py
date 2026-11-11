"""客户端侧连接封装（M5）：``MCPConnection``。

门面 :class:`~cn.mcheat.mcp.api.MultiMCPManager` 位于包顶层 ``api.py``
（与 ``cn.mcheat.rag.api`` 的布局一致）。
"""
from __future__ import annotations

from .connection import MCPConnection

__all__ = ["MCPConnection"]
