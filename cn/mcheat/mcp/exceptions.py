"""MCP 集成框架的异常体系（X1）。

设计原则（与 ``cn.mcheat.rag.exceptions`` 保持一致）：

* 所有异常继承 :class:`MCPError`，调用方可以 ``except MCPError`` 一网打尽；
* 按**阶段**分层（配置 / 传输 / 连接 / 发现 / 调用 / 服务端），
  便于上层做差异化重试与告警；
* 报错信息必须包含「发生了什么 + 上下文（哪个服务器）+ 怎么修」，
  不允许裸 ``raise``。
"""
from __future__ import annotations


class MCPError(Exception):
    """本框架所有异常的基类。"""


# --- 配置 -------------------------------------------------------------------


class ConfigError(MCPError):
    """配置文件 / 环境变量非法：解析失败、结构不对、必填项缺失等。"""


class DuplicateServerError(ConfigError):
    """同一配置来源里出现了重名的服务器。"""

    def __init__(self, name: str, source: str) -> None:
        super().__init__(
            f"配置来源 {source!r} 中服务器名 {name!r} 重复。"
            f"请改名或删除重复条目（跨多个来源时后者覆盖前者，属正常行为）。"
        )


# --- 传输 -------------------------------------------------------------------


class TransportError(MCPError):
    """传输层错误：无法建立底层通道（子进程拉起失败 / HTTP 连不上等）。"""


class UnsupportedTransportError(TransportError):
    """配置的 transport 类型既不是 stdio 也不是 streamable-http，且未注册自定义实现。"""

    def __init__(self, transport: str, server: str) -> None:
        super().__init__(
            f"服务器 {server!r} 的传输类型 {transport!r} 不受支持。"
            f"可用值：'stdio' / 'streamable-http'；"
            f"自定义传输请实现 cn.mcheat.mcp.core.interfaces.Transport 并在工厂注册。"
        )


# --- 连接 -------------------------------------------------------------------


class ConnectionError(MCPError):
    """连接生命周期错误：能力协商失败、会话中断、重连耗尽等。

    注意：有意遮蔽内建 :class:`ConnectionError`——本框架所有连接问题
    统一走 MCPError 体系，调用方无需同时捕获两个基类。
    """


class ServerUnreachableError(ConnectionError):
    """某台服务器无法建立连接（发现阶段隔离使用，不阻断其余服务器）。"""

    def __init__(self, server: str, reason: str) -> None:
        self.server = server
        self.reason = reason
        super().__init__(
            f"服务器 {server!r} 无法连接：{reason}。"
            f"多服务器聚合时该服务器已被跳过，其余服务器不受影响；"
            f"请检查 command/args（stdio）或 url/网络（streamable-http）。"
        )


class SessionClosedError(ConnectionError):
    """在已关闭 / 尚未建立的连接上执行操作。"""

    def __init__(self, server: str, operation: str) -> None:
        super().__init__(
            f"服务器 {server!r} 的会话已关闭或尚未建立，无法执行 {operation!r}。"
            f"请先 ensure_connected()（或经 MultiMCPManager 触发懒连接），"
            f"close() 之后不要继续使用连接对象。"
        )


# --- 发现与调用 --------------------------------------------------------------


class DiscoveryError(MCPError):
    """能力发现失败：tools/list 等请求出错或响应结构非法。"""


class ToolNotFoundError(DiscoveryError):
    """按名调用工具时，聚合注册表中不存在该工具。"""

    def __init__(self, name: str, available: list[str]) -> None:
        preview = "、".join(available[:10]) or "（空）"
        more = f" 等共 {len(available)} 个" if len(available) > 10 else ""
        super().__init__(
            f"工具 {name!r} 不存在。当前可用工具：{preview}{more}。"
            f"跨服务器同名工具的聚合名为 '<server>__<tool>'，"
            f"可调用 manager.get_tools() 查看完整清单。"
        )


class ToolCallError(MCPError):
    """远端 tools/call 执行失败（服务器返回 isError，或参数不合法）。"""

    def __init__(self, server: str, tool: str, reason: str) -> None:
        self.server = server
        self.tool = tool
        super().__init__(f"服务器 {server!r} 上工具 {tool!r} 调用失败：{reason}")


# --- 服务端 -----------------------------------------------------------------


class ServerBuildError(MCPError):
    """服务端构建 / 启动错误：装饰器用法不对、传输类型非法等。"""
