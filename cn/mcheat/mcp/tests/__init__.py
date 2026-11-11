"""MCP 集成框架测试套件。

兼容两种运行方式（项目根目录）::

    python -m cn.mcheat.mcp.tests.test_config      # 自带 runner，无需 pytest
    python -m cn.mcheat.mcp.tests.test_transport
    python -m cn.mcheat.mcp.tests.test_registry
    python -m cn.mcheat.mcp.tests.test_end_to_end  # 端到端（需 mcp SDK）

    pytest cn/mcheat/mcp/tests/                     # 装了 pytest 的话
"""
from __future__ import annotations
