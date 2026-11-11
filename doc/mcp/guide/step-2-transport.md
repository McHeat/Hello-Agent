# 分步引导 · Step 2：抽象接口、领域模型、传输层与连接封装（M2 / M3 / M4 / M5）

> 系列总纲见 [`../01-architecture.md`](../01-architecture.md)。本步是框架的
> **协议对接核心**：从「能解析配置」推进到「能连上一台真实的 MCP 服务器并调用工具」。

## 当前目标

Step 1 回答了「服务器从哪来」，本步回答「怎么连、怎么说话」：

1. **先定契约再写实现**（接口抽象）：把传输、连接、注册表三个扩展点全部
   定义为 ABC——上层此后只依赖接口，实现可替换（自定义 gRPC 传输、
   mock 连接做测试等）；
2. **建立防腐层**：上层应用只认 `models.py` 的领域描述符，不认
   `mcp.types`——SDK 版本演进（字段改名、结构变化）时只改
   `client/connection.py` 一处。本步验证时就真实踩到一次：SDK 的
   `InitializeResult` 字段是 camelCase（`serverInfo`），且旧版
   `stdio_client` 返回三元组 / 新版二元组——都被防腐层吸收，未扩散；
3. **管好异步资源**：MCP 连接 = 子进程 / HTTP 会话 + anyio 任务组，
   任何泄漏都会变成悬垂进程。全部资源挂 `AsyncExitStack` 统一逆序释放，
   异常路径同样清理（验证脚本覆盖）。

## 产出内容

| 文件 | 编号 | 职责 |
|------|------|------|
| `models.py` | M3 | `ToolDescriptor` / `ResourceDescriptor` / `PromptDescriptor` / `CallToolResult` / `ServerInfo` / `ContentBlock`，含 `qualified_name` 与 `to_openai_tool()` |
| `core/interfaces.py` | M2 | `Transport` / `ServerConnection` / `CapabilityRegistry` 三个 ABC + `TransportFactory` / `ConnectionFactory` 类型签名 |
| `transport/stdio.py` | M4 | `StdioTransport`：子进程拉起、stderr 转发、进程树优雅终止 |
| `transport/http.py` | M4 | `StreamableHttpTransport`：远程端点、`MCP-Session-Id` 查询、HTTP DELETE 终止会话 |
| `transport/factory.py` | M4 | `create_transport()` 分发 + `register_transport()` 自定义传输注册 |
| `client/connection.py` | M5 | `MCPConnection`：懒连接（Lock + 双重检查）、能力协商、三大原语、TTL 缓存、幂等关闭 |

接口抽象的关键设计：

- `Transport`：`open() → TransportStreams` / `aclose()`，同时是 async
  context manager。流类型用 `Any`——抽象层不 import SDK；
- `ServerConnection`：`ensure_connected()` / `list_*` / `call_tool` /
  `close()`，实现方契约写进 docstring（幂等、并发安全、关闭后抛
  `SessionClosedError`）；
- 传输工厂支持**自定义传输注册**：`register_transport("grpc", GrpcTransport)`
  之后配置里就能写 `"transport": "grpc"`；
- `MCPConnection.__init__` 接受 `transport_factory` 注入——单测时可以
  注入假传输，不依赖真实子进程。

零依赖分层不变：M2/M3 顶层不 import SDK；M4/M5 的 SDK import 全部
延迟到 `open()` / `_connect()` 时（`require_module` 先行检查，缺依赖报
「怎么修」的 ImportError）。

## 与架构图的对应关系

- M4 + M5 对应**时序图 ② 段「发现阶段」**的前半：`ensure_connected →
  create_transport → initialize → 能力协商返回`；
- `AsyncExitStack` 逆序释放对应**时序图 ④ 段「关闭阶段」**：
  session → transport 逐层退出，子进程按协议顺序（关 stdin → 等待 →
  SIGTERM → SIGKILL）终止；
- `ServerInfo` 即时序图里 `capabilities + serverInfo` 返回值的落点；
- `ToolDescriptor.qualified_name`（`demo__add`）为 Step 3 的
  **M7 聚合注册表**准备好路由键——按名调用时从中解析出目标服务器；
- `transport/factory.py` 是架构图中「M4 传输工厂」节点的完整实现，
  虚线「实现接口」即 `StdioTransport` / `StreamableHttpTransport`
  继承 `core.interfaces.Transport`。

## 验证方式

已执行的端到端验证（从项目根目录，用 hello-agent conda 环境的 Python）：

```text
工具清单: [('add', 'demo__add'), ('get_weather', 'demo__get_weather')]
ServerInfo: hello-agent-mcp-demo 2025-11-25 ['experimental', 'prompts', 'resources', 'tools']
openai tool: demo__add
调用 add -> 42.0 | structured: {'result': 42.0}
OK: 关闭后 -> SessionClosedError          # close() 幂等 + 关闭后操作报错
OK: 不可达 -> 服务器 'bad' 无法连接…      # 失败被包装为 ServerUnreachableError
```

覆盖点：懒连接自动触发、能力协商、工具发现与限定名、真实工具调用
（含 structuredContent）、优雅关闭幂等性、关闭后操作防御、不可达
服务器的错误隔离、资源零泄漏（进程正常退出）。

复现脚本（等价于上面已执行的验证）：

```bash
cd Hello-Agent && python - <<'EOF'
import asyncio
from cn.mcheat.mcp.config import StdioServerConfig
from cn.mcheat.mcp.client import MCPConnection

async def main():
    conn = MCPConnection(StdioServerConfig(
        name="demo", transport="stdio",
        command="python", args=["examples/mcp_demo_server.py"]))
    tools = await conn.list_tools()
    print([(t.name, t.qualified_name) for t in tools])
    print(await (await conn.call_tool("add", {"a": 12, "b": 30})).text())
    await conn.close()

asyncio.run(main())
EOF
```
