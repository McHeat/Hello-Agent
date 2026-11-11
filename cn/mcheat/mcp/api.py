"""多服务器聚合管理器（M6）：框架对上层应用的门面（与 ``cn.mcheat.rag.api``
的 RAGSystem 同位，位于包顶层）。

编排 ``M5 连接``与 ``M7 注册表``，职责（对应架构时序图全部四段）：

* **配置驱动**：从 ``MCPClientSettings`` 登记服务器（或运行期 ``add_server``）；
* **并发发现**：``refresh()`` / ``get_tools()`` 用 ``asyncio.gather``
  并发拉取各服务器能力，O(n) 连接数；
* **失败隔离**：单台不可达只记入 ``failures``，不阻断其余服务器
  （可诊断、可重试，见 X1 异常体系）；对应注册表侧 ``discard`` 清场；
* **按名路由**：``call_tool("math__add", ...)`` 经注册表定位来源连接转发；
  注册表为空时自动触发一次发现再重试（可直接 call_tool 不先 get_tools）；
* **优雅关闭**：``close()`` 并发关闭全部连接（幂等）。

依赖注入（接口抽象）：``connection_factory`` / ``registry`` 均可替换，
上层测试可注入 mock 连接，不拉真实子进程。
"""
from __future__ import annotations

import asyncio
from typing import Any

from ._utils import get_logger
from .client.connection import MCPConnection
from .config import MCPClientSettings, ServerConfig
from .core.interfaces import CapabilityRegistry, ConnectionFactory
from .exceptions import ConfigError, MCPError, ToolCallError
from .models import CallToolResult, PromptMessage, ResourceContent, ToolDescriptor
from .registry.tools import AggregatedRegistry

logger = get_logger(__name__)

__all__ = ["MultiMCPManager"]


class MultiMCPManager:
    """面向应用的门面：配置进来，聚合能力出去。

    Args:
        settings: 已校验的多服务器设置（``load_config`` / ``from_dict`` 产物）。
        connection_factory: 连接工厂（依赖注入点），默认 :class:`MCPConnection`
            构造；单测可注入假实现。
        registry: 聚合注册表（依赖注入点），默认
            :class:`~cn.mcheat.mcp.registry.AggregatedRegistry`。
        list_cache_ttl: 透传给连接的 ``list_*`` 缓存秒数（0 不缓存）。
        raise_on_error: 透传给连接；工具执行错误是否抛 :class:`ToolCallError`。

    用法::

        manager = MultiMCPManager(load_config("config.json"))
        tools = await manager.get_tools()
        result = await manager.call_tool("math__add", {"a": 1, "b": 2})
        await manager.close()
    """

    def __init__(
        self,
        settings: MCPClientSettings | dict[str, Any],
        connection_factory: ConnectionFactory | None = None,
        registry: CapabilityRegistry | None = None,
        list_cache_ttl: float = 0.0,
        raise_on_error: bool = True,
    ) -> None:
        if isinstance(settings, dict):  # 便捷：直接传 dict 配置
            settings = MCPClientSettings.from_dict(settings)
        if not isinstance(settings, MCPClientSettings):
            raise ConfigError(
                f"settings 需为 MCPClientSettings 或 dict，收到 {type(settings).__name__}"
            )
        self._settings = settings
        self._connection_factory = connection_factory or self._default_connection_factory(
            list_cache_ttl, raise_on_error
        )
        self._registry = registry or AggregatedRegistry()
        self._connections: dict[str, Any] = {}  # name → ServerConnection
        #: 发现阶段不可达的服务器：name → 异常（透出诊断，不阻断聚合）。
        self.failures: dict[str, MCPError] = {}
        for name, config in settings.servers.items():
            self._register(name, config)

    @staticmethod
    def _default_connection_factory(
        list_cache_ttl: float, raise_on_error: bool
    ) -> ConnectionFactory:
        def factory(config: ServerConfig):
            return MCPConnection(
                config, list_cache_ttl=list_cache_ttl, raise_on_error=raise_on_error
            )

        return factory

    # -- 注册 -------------------------------------------------------------------

    def _register(self, name: str, config: ServerConfig) -> None:
        if name in self._connections:
            raise ConfigError(
                f"服务器名 {name!r} 已注册。同一 manager 内重名会破坏按名路由，"
                f"请改名（跨配置来源合并请用 MCPClientSettings.merge）。"
            )
        self._connections[name] = self._connection_factory(config)
        logger.debug("已登记服务器 %s（%s）", name, config.transport)

    def add_server(self, name: str, entry: dict[str, Any]) -> None:
        """运行期追加一台服务器（配置 dict 结构与文件一致）。"""
        config = self._settings.add_server(name, entry)
        self._register(name, config)

    @property
    def servers(self) -> list[str]:
        """已登记的服务器名列表。"""
        return list(self._connections)

    def connection(self, name: str) -> Any:
        """取某台服务器的连接对象（进阶用法：直接访问 ServerConnection 接口）。"""
        return self._connections[name]

    # -- 并发发现 -----------------------------------------------------------------

    async def _gather(self, op_name: str, fn) -> dict[str, Any]:
        """并发对全部服务器执行 ``fn(connection)``，失败隔离进 ``failures``。"""
        self.failures = {}
        tasks = {
            name: asyncio.ensure_future(fn(conn))
            for name, conn in self._connections.items()
        }
        results: dict[str, Any] = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except MCPError as exc:  # 框架异常：记入 failures，继续聚合
                self.failures[name] = exc
                self._registry.discard(name)
                logger.warning("服务器 %s 在 %s 阶段失败，已隔离：%s", name, op_name, exc)
            except Exception as exc:  # 未知异常同样隔离，但包装可诊断类型
                wrapped = MCPError(f"服务器 {name!r} 在 {op_name} 阶段发生未预期错误：{exc}")
                self.failures[name] = wrapped
                self._registry.discard(name)
                logger.warning("服务器 %s 在 %s 阶段异常（未预期）：%s", name, op_name, exc)
        return results

    async def refresh(self) -> dict[str, list[ToolDescriptor]]:
        """并发重新发现全部服务器的能力（覆盖式更新注册表）。

        Returns:
            ``{服务器名: 工具列表}``（不含失败的服务器）。
            失败清单见 ``self.failures``。
        """
        results = await self._gather("发现", lambda c: c.list_tools())
        for name, tools in results.items():
            self._registry.register_tools(name, tools)
            try:
                self._registry.register_resources(name, await self._connections[name].list_resources())
                self._registry.register_prompts(name, await self._connections[name].list_prompts())
            except MCPError as exc:
                # 工具可用而资源/提示拉取失败：保留工具，记部分失败
                self.failures[f"{name}(resources/prompts)"] = exc
                logger.warning("服务器 %s 资源/提示拉取失败（工具保留）：%s", name, exc)
        logger.info(
            "发现完成：%d 台可用 / %d 台失败 / %d 个工具",
            len(results), len(self.failures), len(self._registry.all_tools()),
        )
        return results

    async def get_tools(self, refresh: bool = False) -> list[ToolDescriptor]:
        """聚合全部服务器的工具清单（懒发现：注册表为空时自动 refresh）。

        Args:
            refresh: ``True`` 时强制重新发现（服务器侧可能新增了工具）。
        """
        if refresh or len(self._registry) == 0:
            await self.refresh()
        return self._registry.all_tools()

    async def list_resources(self, refresh: bool = False) -> list[Any]:
        """聚合全部服务器的资源清单。"""
        if refresh or len(self._registry) == 0:
            await self.refresh()
        return self._registry.all_resources()

    async def list_prompts(self, refresh: bool = False) -> list[Any]:
        """聚合全部服务器的提示模板清单。"""
        if refresh or len(self._registry) == 0:
            await self.refresh()
        return self._registry.all_prompts()

    # -- 按名路由调用 -------------------------------------------------------------

    async def _routing(self, lookup):
        """注册表查不到时自动发现一次再查（可直接 call_tool 免 get_tools）。"""
        try:
            return lookup()
        except Exception:
            if len(self._registry) == 0:
                await self.refresh()
                return lookup()
            raise

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        """按限定名（或唯一短名）路由到来源服务器执行 ``tools/call``。

        Args:
            name: ``"server__tool"`` 限定名；跨服务器唯一时也可直接用原始名。
            arguments: 工具参数。

        Raises:
            ToolNotFoundError: 名字不存在 / 多义（错误信息含候选）。
            ToolCallError: 远端执行失败。
            ServerUnreachableError: 来源服务器不可达。
        """
        tool = await self._routing(lambda: self._registry.lookup_tool(name))
        logger.debug("工具 %s 路由到服务器 %s", name, tool.server)
        return await self._connections[tool.server].call_tool(tool.name, arguments)

    async def read_resource(self, uri: str) -> list[ResourceContent]:
        """按 URI 路由到来源服务器读取资源。"""
        resource = await self._routing(lambda: self._registry.lookup_resource(uri))
        return await self._connections[resource.server].read_resource(uri)

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> list[PromptMessage]:
        """按限定名（或唯一短名）路由到来源服务器渲染提示。"""
        prompt = await self._routing(lambda: self._registry.lookup_prompt(name))
        return await self._connections[prompt.server].get_prompt(prompt.name, arguments)

    # -- 生命周期 -------------------------------------------------------------------

    async def connect_all(self) -> dict[str, Any]:
        """并发建立全部连接（可选预热；不调用则首次 get_tools 懒连接）。

        Returns:
            ``{服务器名: ServerInfo}``（失败的服务器不在其中，见 failures）。
        """
        results = await self._gather("连接", lambda c: c.ensure_connected())
        return results

    async def close(self) -> None:
        """并发优雅关闭全部连接（幂等）。"""
        await asyncio.gather(
            *(conn.close() for conn in self._connections.values()),
            return_exceptions=True,  # 单台关闭失败不影响其余
        )
        logger.info("已关闭全部 %d 台服务器连接", len(self._connections))

    async def __aenter__(self) -> "MultiMCPManager":
        await self.connect_all()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()
