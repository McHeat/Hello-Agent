"""服务端构建器（M8）：装饰器注册三大原语，双传输启动。

用法（对应架构图 L1 → L2 → L3 的服务端方向）::

    from cn.mcheat.mcp import MCPServerBuilder

    server = MCPServerBuilder("demo", instructions="演示服务器")

    @server.tool                      # 裸用法：函数名即工具名
    def add(a: float, b: float) -> float:
        \"\"\"两数相加。\"\"\"
        return a + b

    @server.tool(name="multiply")     # 带参用法
    def mul(a: float, b: float) -> float: ...

    @server.resource("notes://{name}")   # 模板资源（URI 参数 → 函数参数）
    def get_note(name: str) -> str: ...

    @server.prompt
    def greet(name: str) -> str: ...

    server.run("stdio")               # 本地子进程模式（Host 拉起）
    server.run("streamable-http")     # 远程服务模式（默认 127.0.0.1:8000）

设计（与其他模块一致的接口抽象原则）：

* **零依赖 import**：装饰器只把函数记进内存清单，官方 SDK 延迟到
  ``run()`` 才 import——无 mcp 环境也能定义/静态检查服务器组件；
* **构建与运行分离**：``build()`` 产出底层 ``FastMCP`` 实例（可注入
  自定义实现 / 单测直接检查注册结果），``run()`` 才进入阻塞事件循环；
* **生命周期语义外露**：``run()`` 阻塞（协议决定的常驻循环），编排方
  需要「跑在后台」时自行放入线程 / 子进程，框架不偷偷起线程。

协议要点落实：HTTP 模式默认仅绑定 ``127.0.0.1``（规范安全建议），
日志由 FastMCP 走 stderr，stdout 保持协议通道纯净（stdio 模式硬约束）。
"""
from __future__ import annotations

from typing import Any, Callable, Literal

from .._utils import get_logger, require_module
from ..exceptions import ServerBuildError

logger = get_logger(__name__)

__all__ = ["MCPServerBuilder"]

TransportName = Literal["stdio", "streamable-http", "sse"]


class MCPServerBuilder:
    """装饰器风格的服务端构建器（内部组装官方 ``FastMCP``）。

    Args:
        name: 服务器名（能力协商时上报给客户端的 ``serverInfo.name``）。
        instructions: 给客户端 / 模型的使用说明（可选）。
        host: HTTP 模式绑定地址，默认 ``127.0.0.1``（仅本地，协议安全建议；
            对外服务显式传 ``0.0.0.0`` 并自行做好认证与 Origin 校验）。
        port: HTTP 模式端口，默认 8000。
    """

    def __init__(
        self,
        name: str,
        instructions: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        if not name or not isinstance(name, str):
            raise ServerBuildError(f"服务器名必须为非空字符串，收到：{name!r}")
        self._name = name
        self._instructions = instructions
        self._host = host
        self._port = port
        # (函数, 注册参数) 清单——build() 时才真正落到 FastMCP 上
        self._tools: list[tuple[Callable, dict[str, Any]]] = []
        self._resources: list[tuple[Callable, dict[str, Any]]] = []
        self._prompts: list[tuple[Callable, dict[str, Any]]] = []
        self._built: Any = None  # build() 产出的 FastMCP 实例

    # -- 装饰器 -----------------------------------------------------------------

    def tool(
        self, fn: Callable | None = None, *, name: str | None = None, description: str | None = None
    ) -> Callable:
        """注册工具（``tools/list`` 可见，模型可 ``tools/call``）。

        裸用法 ``@server.tool`` 与带参用法 ``@server.tool(name=...)`` 均支持；
        参数 Schema 由 SDK 从类型注解自动生成。
        """

        def decorator(f: Callable) -> Callable:
            self._tools.append((f, {"name": name, "description": description}))
            logger.debug("登记工具 %s（name=%s）", getattr(f, "__name__", f), name or f.__name__)
            return f

        return decorator(fn) if callable(fn) else decorator

    def resource(self, uri: str, *, name: str | None = None, mime_type: str | None = None) -> Callable:
        """注册资源（模板 URI 的 ``{param}`` 映射到同名函数参数）。

        Args:
            uri: 资源 URI 或模板，如 ``"notes://{name}"`` / ``"config://app"``。
        """

        def decorator(f: Callable) -> Callable:
            self._resources.append(
                (f, {"uri": uri, "name": name, "mime_type": mime_type})
            )
            return f

        return decorator

    def prompt(self, fn: Callable | None = None, *, name: str | None = None) -> Callable:
        """注册提示模板（参数占位 → ``prompts/get`` 时渲染成消息）。"""

        def decorator(f: Callable) -> Callable:
            self._prompts.append((f, {"name": name}))
            return f

        return decorator(fn) if callable(fn) else decorator

    # -- 自检 -------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """已注册组件的清单（不依赖 SDK，构建前自检用）。"""
        return {
            "name": self._name,
            "tools": [kw.get("name") or fn.__name__ for fn, kw in self._tools],
            "resources": [kw["uri"] for _, kw in self._resources],
            "prompts": [kw.get("name") or fn.__name__ for fn, kw in self._prompts],
        }

    # -- 构建与运行 ---------------------------------------------------------------

    def build(self) -> Any:
        """组装底层 ``FastMCP`` 实例（幂等；单测 / 高级用法可直接拿实例）。

        Raises:
            ServerBuildError: 无可注册组件或依赖缺失。
        """
        if self._built is not None:
            return self._built
        if not (self._tools or self._resources or self._prompts):
            raise ServerBuildError(
                f"服务器 {self._name!r} 未注册任何 tool / resource / prompt，"
                f"run() 起来也不会暴露能力。请先用装饰器注册组件。"
            )
        require_module("mcp", "MCP 服务端构建")
        from mcp.server.fastmcp import FastMCP

        # host/port 必须在构造时传入：FastMCP.run() 不接受地址参数，
        # 绑定地址由实例构造决定（本 SDK 版本实测行为）。
        mcp = FastMCP(
            self._name,
            instructions=self._instructions,
            host=self._host,
            port=self._port,
        )
        for fn, kw in self._tools:
            mcp.tool(
                name=kw["name"],
                description=kw["description"],
            )(fn)
        for fn, kw in self._resources:
            mcp.resource(
                uri=kw["uri"],
                name=kw["name"],
                mime_type=kw["mime_type"],
            )(fn)
        for fn, kw in self._prompts:
            mcp.prompt(name=kw["name"])(fn)
        self._built = mcp
        return mcp

    def run(
        self,
        transport: TransportName = "stdio",
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """启动服务器（**阻塞**，直到传输关闭）。

        Args:
            transport: ``"stdio"``（本地子进程模式）或
                ``"streamable-http"``（远程服务模式）；``"sse"`` 为旧版兼容。
            host / port: HTTP 模式覆盖项（默认取构造时的 127.0.0.1:8000）。
        """
        # 地址覆盖须在 build() 之前生效（FastMCP 构造期决定绑定地址）
        if transport in ("streamable-http", "sse"):
            if host is not None:
                self._host = host
            if port is not None:
                self._port = port
            logger.info(
                "启动 %s：%s（http://%s:%s/mcp）",
                self._name, transport, self._host, self._port,
            )
        else:
            logger.info("启动 %s（stdio，等待宿主拉起）", self._name)
        mcp = self.build()
        mcp.run(transport=transport)
