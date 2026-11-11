"""配置层（M1）：dataclass 模型 + 多来源加载，零第三方依赖。

职责（对应 ``doc/mcp/01-architecture.md`` 数据流图第 ① 段）：

* 把 JSON / YAML / 代码内 dict 三类来源解析成统一的
  :class:`MCPClientSettings`；
* 结构校验**前置**：配置错误在建连之前就暴露（早失败），并给出
  「发生了什么 + 哪个服务器 + 怎么修」的完整信息；
* 字符串值支持 ``${ENV_VAR}`` 环境变量插值——鉴权 token 等敏感信息
  可留配置文件之外。

配置格式（dict 的 key 即服务器名，与 langchain-mcp-adapters 习惯一致）::

    {
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["examples/servers/demo_server.py", "--stdio"]
        },
        "remote": {
            "transport": "streamable-http",
            "url": "http://127.0.0.1:8000/mcp",
            "headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
            "timeout": 30.0
        }
    }

环境变量展开规则：``${VAR}`` 已定义则替换，未定义则替换为空串并记
warning（可用 ``${VAR:-default}`` 给默认值）。

YAML 支持是**可选依赖**（pyyaml）：只有传入 .yaml/.yml 文件时才需要，
缺依赖时报带修复提示的 ImportError，不影响 JSON 路径与框架 import。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ._utils import get_logger, require_module
from .exceptions import ConfigError

logger = get_logger(__name__)

__all__ = [
    "ServerConfig",
    "StdioServerConfig",
    "HttpServerConfig",
    "MCPClientSettings",
    "load_config",
    "expand_env",
]


# ---------------------------------------------------------------------------
# 环境变量插值
# ---------------------------------------------------------------------------

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """递归展开配置值里的 ``${VAR}`` / ``${VAR:-default}``。

    Args:
        value: 任意配置片段（dict / list / str / 其他标量原样返回）。

    Returns:
        同结构的新对象，所有字符串完成插值；未定义且无默认值的变量
        展开为空串并记 warning。
    """
    if isinstance(value, str):
        def _sub(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            if var in os.environ:
                return os.environ[var]
            if default is not None:
                return default
            logger.warning("配置引用的环境变量 %s 未定义，已展开为空串。", var)
            return ""

        return _ENV_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# 服务器配置模型
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    """单个 MCP 服务器的连接元数据（注册表登记的就是它）。

    Attributes:
        name: 服务器名（来自配置 dict 的 key），聚合命名空间与错误信息都用它。
        transport: 传输类型，``"stdio"`` 或 ``"streamable-http"``。
    """

    name: str
    transport: str


@dataclass
class StdioServerConfig(ServerConfig):
    """本地子进程型服务器（stdio 传输）。

    Attributes:
        command: 可执行命令，如 ``"python"`` / ``"uvx"`` / ``"npx"``。
        args: 命令行参数列表。
        env: 附加环境变量（合并不覆盖父进程全部环境）。
        cwd: 子进程工作目录；``None`` 表示继承当前目录。
    """

    transport: str = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


@dataclass
class HttpServerConfig(ServerConfig):
    """远程服务型服务器（Streamable HTTP 传输）。

    Attributes:
        url: MCP 端点，如 ``"http://127.0.0.1:8000/mcp"``。
        headers: 随请求发送的 HTTP 头（鉴权 token 建议用 ``${VAR}`` 插值）。
        timeout: 请求超时秒数。
    """

    transport: str = "streamable-http"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0


_STDIO_KEYS = {"name", "transport", "command", "args", "env", "cwd"}
_HTTP_KEYS = {"name", "transport", "url", "headers", "timeout"}


def _check_keys(server: str, data: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = {k for k in data if not k.startswith("_")} - allowed
    if unknown:
        raise ConfigError(
            f"服务器 {server!r}（{data.get('transport', '?')!r} 传输）"
            f"含未知配置键：{sorted(unknown)}。允许的键：{sorted(allowed - {'name'})}。"
            f"多写的键多半是笔误，会静默失效，故直接报错。"
        )


# ---------------------------------------------------------------------------
# 顶层设置与解析入口
# ---------------------------------------------------------------------------


@dataclass
class MCPClientSettings:
    """多服务器客户端设置：name → ServerConfig 的有序集合。

    调用 :meth:`from_dict` / :meth:`merge` 完成构造后即已通过全部校验，
    后续模块可以放心使用。
    """

    servers: dict[str, ServerConfig] = field(default_factory=dict)

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MCPClientSettings":
        """从 ``{"servers": {...}}``（或直接服务器 dict）解析并校验。

        Args:
            data: 配置片段。允许顶层带 ``"servers"`` 键，也允许直接就是
                ``{"<server-name>": {...}, ...}``。

        Raises:
            ConfigError: 结构不对、必填缺失、键值非法、名字重复。
        """
        servers_data = data.get("servers", data) if isinstance(data, Mapping) else None
        if not isinstance(servers_data, Mapping):
            raise ConfigError(
                f"配置必须是对象（含 servers 或直接为服务器映射），收到：{type(data).__name__}"
            )
        settings = cls()
        for name, entry in servers_data.items():
            settings.add_server(name, entry)
        return settings

    def add_server(self, name: str, entry: Mapping[str, Any]) -> ServerConfig:
        """登记一台服务器（已存在的名字报错，防止静默覆盖）。

        Args:
            name: 服务器名。
            entry: 该服务器的配置 dict，必须含 ``transport`` 键。

        Returns:
            构造好的 :class:`StdioServerConfig` 或 :class:`HttpServerConfig`。

        Raises:
            DuplicateServerError: 名字已存在。
            ConfigError: 条目结构 / 必填项 / 取值非法。
        """
        if name in self.servers:
            from .exceptions import DuplicateServerError  # 局部导入避免顶部拥挤

            raise DuplicateServerError(name, "同一份配置内")
        if not isinstance(entry, Mapping):
            raise ConfigError(
                f"服务器 {name!r} 的配置必须是对象，收到 {type(entry).__name__}"
            )
        merged = dict(entry)
        merged.setdefault("name", name)
        config = self._build_server(name, expand_env(merged))
        self.servers[name] = config
        return config

    @staticmethod
    def _build_server(name: str, data: Mapping[str, Any]) -> ServerConfig:
        transport = data.get("transport")
        if transport == "stdio":
            _check_keys(name, data, _STDIO_KEYS)
            command = data.get("command")
            if not command or not isinstance(command, str):
                raise ConfigError(
                    f"stdio 服务器 {name!r} 必须给出非空字符串 'command'"
                    f"（例如把 MCP 服务端跑成子进程的命令）。"
                )
            args = data.get("args", [])
            env = data.get("env", {})
            cwd = data.get("cwd")
            if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                raise ConfigError(f"stdio 服务器 {name!r} 的 args 必须是字符串列表。")
            if not isinstance(env, Mapping) or not all(
                isinstance(v, str) for v in env.values()
            ):
                raise ConfigError(f"stdio 服务器 {name!r} 的 env 必须是 str→str 对象。")
            if cwd is not None and not isinstance(cwd, str):
                raise ConfigError(f"stdio 服务器 {name!r} 的 cwd 必须是字符串或省略。")
            return StdioServerConfig(
                name=name, transport="stdio", command=command,
                args=list(args), env=dict(env), cwd=cwd,
            )
        if transport == "streamable-http":
            _check_keys(name, data, _HTTP_KEYS)
            url = data.get("url")
            if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
                raise ConfigError(
                    f"streamable-http 服务器 {name!r} 必须给出以 http:// 或 "
                    f"https:// 开头的 'url'，收到：{url!r}"
                )
            headers = data.get("headers", {})
            timeout = data.get("timeout", 30.0)
            if not isinstance(headers, Mapping) or not all(
                isinstance(v, str) for v in headers.values()
            ):
                raise ConfigError(f"服务器 {name!r} 的 headers 必须是 str→str 对象。")
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ConfigError(f"服务器 {name!r} 的 timeout 必须为正数（秒）。")
            return HttpServerConfig(
                name=name, transport="streamable-http",
                url=url, headers=dict(headers), timeout=float(timeout),
            )
        raise ConfigError(
            f"服务器 {name!r} 的 transport 必须是 'stdio' 或 'streamable-http'，"
            f"收到：{transport!r}。自定义传输见 transport/factory.py 的注册说明。"
        )

    # -- 合并与查询 ----------------------------------------------------------

    def merge(self, other: "MCPClientSettings", override: bool = True) -> "MCPClientSettings":
        """合并另一份设置（环境变量 / 代码覆盖场景，后者优先）。

        Args:
            other: 另一份设置。
            override: ``True`` 时 ``other`` 的同名服务器覆盖本地
                （对应优先级「后者覆盖前者」）；``False`` 时保留本地。

        Returns:
            ``self``，便于链式调用。
        """
        for name, config in other.servers.items():
            if override or name not in self.servers:
                self.servers[name] = config
        return self

    def __len__(self) -> int:
        return len(self.servers)


# ---------------------------------------------------------------------------
# 文件加载入口
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> MCPClientSettings:
    """从 JSON / YAML 文件加载配置（字符串值会做 ``${VAR}`` 插值）。

    Args:
        path: 配置文件路径，按后缀 ``.json`` / ``.yaml`` / ``.yml`` 选择解析器。

    Returns:
        校验通过的 :class:`MCPClientSettings`。

    Raises:
        ConfigError: 文件不存在 / 后缀不支持 / 解析失败 / 结构校验失败。
        ImportError: YAML 文件但未安装 pyyaml（提示安装命令）。
    """
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"配置文件不存在：{file}（相对路径请从项目根目录起算）。")
    suffix = file.suffix.lower()
    text = file.read_text(encoding="utf-8")
    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"JSON 配置解析失败（{file}）：{exc}") from exc
    elif suffix in (".yaml", ".yml"):
        require_module("yaml", "YAML 配置文件支持")
        import yaml  # 局部导入：pyyaml 为可选依赖

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML 配置解析失败（{file}）：{exc}") from exc
    else:
        raise ConfigError(
            f"不支持的配置文件后缀 {suffix!r}（{file}）。支持：.json / .yaml / .yml。"
        )
    logger.info("已加载 MCP 配置：%s", file)
    return MCPClientSettings.from_dict(data)
