"""领域模型（M3）：工具 / 资源 / 提示的统一描述符，零第三方依赖。

设计意图：上层应用**只认这里的数据结构**，不认官方 SDK 的 ``mcp.types``。
好处：

1. SDK 类型随版本演进（字段增删）时，上层代码不动——转换逻辑集中在
   ``client/connection.py`` 一处；
2. 描述符自带 ``server`` 来源字段，跨服务器聚合与路由无需二次查找；
3. ``qualified_name`` 用 ``<server>__<name>`` 消歧，``to_openai_tool()``
   可直接喂给支持 function calling 的任意大模型。

转换原则（connection 层实现）：SDK 对象缺字段时取安全默认值，而不是
把 SDK 的 Optional 原样传染给全框架。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ContentBlock",
    "ToolDescriptor",
    "ResourceDescriptor",
    "ResourceContent",
    "PromptArgument",
    "PromptDescriptor",
    "PromptMessage",
    "CallToolResult",
    "ServerInfo",
]


# ---------------------------------------------------------------------------
# 内容块（tools/call 与资源读取的统一载荷）
# ---------------------------------------------------------------------------


@dataclass
class ContentBlock:
    """MCP content 数组中的一个块（文本 / 图片 / 内嵌资源）。

    Attributes:
        type: 块类型，``"text"`` / ``"image"`` / ``"resource"``。
        text: 文本内容（type=text 时非空）。
        data: base64 数据（type=image 时非空）。
        mime_type: 资源 / 图片的 MIME 类型（如 ``"text/plain"``）。
    """

    type: str
    text: str | None = None
    data: str | None = None
    mime_type: str | None = None

    def as_text(self) -> str:
        """返回该块的文本表示（非文本块返回占位描述）。"""
        if self.text is not None:
            return self.text
        if self.data is not None:
            return f"<{self.type} mime={self.mime_type or '?'} data={len(self.data)}B>"
        return f"<{self.type}>"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


@dataclass
class ToolDescriptor:
    """一个跨服务器聚合后的工具描述符。

    Attributes:
        server: 来源服务器名（聚合命名空间，也用于按名路由回源）。
        name: 工具在来源服务器上的原始名。
        title: 可选展示标题（无则回落 name）。
        description: 给大模型看的功能描述。
        input_schema: JSON Schema（调用参数的形状）。
    """

    server: str
    name: str
    description: str = ""
    title: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """聚合命名空间里的全局唯一名：``<server>__<name>``。"""
        return f"{self.server}__{self.name}"

    def to_openai_tool(self) -> dict[str, Any]:
        """转成 OpenAI function calling 的 tools 数组条目。

        兼容任何「name + description + parameters」约定的 LLM API，
        Anthropic 格式可取 ``function`` 子对象自行映射。
        """
        return {
            "type": "function",
            "function": {
                "name": self.qualified_name,
                "description": self.description or f"(no description) {self.name}",
                "parameters": self.input_schema
                or {"type": "object", "properties": {}},
            },
        }


@dataclass
class CallToolResult:
    """一次 ``tools/call`` 的领域结果。

    Attributes:
        server: 执行该调用的来源服务器。
        tool: 工具原始名。
        content: 内容块列表。
        is_error: 服务器是否报告执行错误。
        structured: 结构化输出（服务器提供时）。
    """

    server: str
    tool: str
    content: list[ContentBlock] = field(default_factory=list)
    is_error: bool = False
    structured: dict[str, Any] | None = None

    def text(self) -> str:
        """拼接全部文本块的文本（多数工具只有一段 text，直接用它）。"""
        return "\n".join(block.as_text() for block in self.content)


# ---------------------------------------------------------------------------
# 资源
# ---------------------------------------------------------------------------


@dataclass
class ResourceDescriptor:
    """一个可读取的上下文数据源（URI 定位）。

    Attributes:
        is_template: 是否为资源模板（URI 含 ``{param}`` 占位，
            读取时用实参替换占位并按模板路由）。
    """

    server: str
    uri: str
    name: str = ""
    description: str = ""
    mime_type: str | None = None
    is_template: bool = False

    def __str__(self) -> str:  # 方便日志与打印
        return f"{self.uri} ({self.name or 'unnamed'}) @ {self.server}"


@dataclass
class ResourceContent:
    """``resources/read`` 返回的单一内容。"""

    uri: str
    text: str | None = None
    data: str | None = None
    mime_type: str | None = None


# ---------------------------------------------------------------------------
# 提示模板
# ---------------------------------------------------------------------------


@dataclass
class PromptArgument:
    """提示模板的一个参数占位。"""

    name: str
    description: str = ""
    required: bool = False


@dataclass
class PromptDescriptor:
    """一个可复用的交互模板。"""

    server: str
    name: str
    description: str = ""
    arguments: list[PromptArgument] = field(default_factory=list)


@dataclass
class PromptMessage:
    """``prompts/get`` 渲染出的一条消息。"""

    role: str  # "user" / "assistant"
    text: str


# ---------------------------------------------------------------------------
# 连接元信息
# ---------------------------------------------------------------------------


@dataclass
class ServerInfo:
    """``initialize`` 能力协商得到的服务器身份与能力摘要。"""

    name: str
    version: str
    protocol_version: str
    capabilities: dict[str, Any] = field(default_factory=dict)
