"""跨服务器聚合注册表（M7）：工具 / 资源 / 提示的内存索引，零依赖。

实现 ``core/interfaces.CapabilityRegistry``。三个职责：

1. **登记**：``register_tools(server, tools)`` 以 ``<server>__<name>`` 为主键
   **覆盖式**登记——服务器侧工具列表变更后重新发现，旧条目自动失效
   （MCP toolsListChanged 语义）；
2. **路由**：``lookup_tool`` 先按限定名精确匹配，再允许唯一短名
   （跨服务器唯一时直接用原始名调用，多数单服务器场景更顺手）；
3. **清场**：``discard(server)`` 移除某服务器的全部登记（连接失败 / 关闭时）。

线程模型：纯内存字典操作，全部同步方法；异步调用方（manager）保证
不会跨线程并发写（asyncio 单线程模型天然满足）。
"""
from __future__ import annotations

import re

from ..core.interfaces import CapabilityRegistry
from ..exceptions import ToolNotFoundError
from ..models import PromptDescriptor, ResourceDescriptor, ToolDescriptor

__all__ = ["AggregatedRegistry"]


def _template_matches(template_uri: str, uri: str) -> bool:
    """判断具体 URI 是否匹配资源模板（``notes://{name}`` 匹配 ``notes://proto``）。

    模板里的每个 ``{param}`` 占位视为一个不含 ``/`` 的通配段。
    """
    parts = re.split(r"(\{[^/}]+\})", template_uri)
    pattern = "".join(
        "[^/]+" if part.startswith("{") and part.endswith("}") else re.escape(part)
        for part in parts
        if part
    )
    return re.fullmatch(pattern, uri) is not None


class AggregatedRegistry(CapabilityRegistry):
    """默认聚合注册表（可整体替换：实现 ``CapabilityRegistry`` 注入管理器即可）。"""

    def __init__(self) -> None:
        # 限定名 / URI / 限定名 → 描述符
        self._tools: dict[str, ToolDescriptor] = {}
        self._resources: dict[str, ResourceDescriptor] = {}
        self._prompts: dict[str, PromptDescriptor] = {}

    # -- 登记 -----------------------------------------------------------------

    def register_tools(self, server: str, tools: list[ToolDescriptor]) -> None:
        """覆盖式登记某服务器的工具集合（旧条目全部失效后重新写入）。"""
        self.discard_tools(server)
        for tool in tools:
            self._tools[tool.qualified_name] = tool

    def register_resources(self, server: str, resources: list[ResourceDescriptor]) -> None:
        self.discard_resources(server)
        for resource in resources:
            self._resources[resource.uri] = resource

    def register_prompts(self, server: str, prompts: list[PromptDescriptor]) -> None:
        self.discard_prompts(server)
        for prompt in prompts:
            key = prompt.name if "__" in prompt.name else f"{server}__{prompt.name}"
            self._prompts[key] = prompt

    # -- 路由 -----------------------------------------------------------------

    def lookup_tool(self, name: str) -> ToolDescriptor:
        """按限定名（``server__tool``）或唯一原始名查工具。

        Raises:
            ToolNotFoundError: 不存在；或原始名匹配到多个服务器
                （错误信息列出全部候选，提示用限定名消歧）。
        """
        tool = self._tools.get(name)
        if tool is not None:
            return tool
        # 短名匹配：所有 name 字段等于入参的条目
        candidates = [t for t in self._tools.values() if t.name == name]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ToolNotFoundError(
                name, [t.qualified_name for t in sorted(candidates, key=lambda t: t.server)]
            )
        raise ToolNotFoundError(name, sorted(self._tools))

    def lookup_resource(self, uri: str) -> ResourceDescriptor:
        """按 URI 查资源：精确匹配 → 模板匹配（``{param}`` 视为通配段）。

        Raises:
            ToolNotFoundError: 不存在（错误信息含已登记 URI 列表）。
        """
        resource = self._resources.get(uri)
        if resource is not None:
            return resource
        for descriptor in self._resources.values():
            if descriptor.is_template and _template_matches(descriptor.uri, uri):
                return descriptor
        raise ToolNotFoundError(
            f"resource:{uri}", sorted(self._resources) or []
        )

    def lookup_prompt(self, name: str) -> PromptDescriptor:
        prompt = self._prompts.get(name)
        if prompt is None:
            candidates = [p for p in self._prompts.values() if p.name == name]
            if len(candidates) == 1:
                return candidates[0]
            raise ToolNotFoundError(f"prompt:{name}", sorted(self._prompts))
        return prompt

    # -- 快照 -----------------------------------------------------------------

    def all_tools(self) -> list[ToolDescriptor]:
        return sorted(self._tools.values(), key=lambda t: t.qualified_name)

    def all_resources(self) -> list[ResourceDescriptor]:
        return sorted(self._resources.values(), key=lambda r: r.uri)

    def all_prompts(self) -> list[PromptDescriptor]:
        return sorted(self._prompts.values(), key=lambda p: p.name)

    # -- 清场 -----------------------------------------------------------------

    def discard(self, server: str) -> None:
        """移除某服务器登记的全部工具 / 资源 / 提示。"""
        self.discard_tools(server)
        self.discard_resources(server)
        self.discard_prompts(server)

    def discard_tools(self, server: str) -> None:
        self._tools = {k: t for k, t in self._tools.items() if t.server != server}

    def discard_resources(self, server: str) -> None:
        self._resources = {k: r for k, r in self._resources.items() if r.server != server}

    def discard_prompts(self, server: str) -> None:
        self._prompts = {k: p for k, p in self._prompts.items() if p.server != server}

    def __len__(self) -> int:
        """已登记的工具总数（资源 / 提示另计，debug 用）。"""
        return len(self._tools)
