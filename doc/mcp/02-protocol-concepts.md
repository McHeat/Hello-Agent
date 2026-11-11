# MCP 协议概念速览

> 配合 [`01-architecture.md`](01-architecture.md) 阅读：架构图讲「框架怎么搭」，
> 本文讲「协议本身说了什么」。实现细节无需全部记住，但每条都对应框架里的
> 一处设计决策（见最后一节的映射表）。

## 1. 三角色与一对一连接

| 角色 | 职责 | 框架对应 |
|------|------|----------|
| **Host** | AI 应用，协调多个 Client、决定上下文 | 你的应用 + `MultiMCPManager` |
| **Client** | 与某台 Server 的专用连接 | `MCPConnection`（每个服务器一个） |
| **Server** | 提供 tools / resources / prompts | `MCPServerBuilder` 构建的进程 |

每个 Client ↔ Server 是**专用一对一连接**：本地 stdio Server 通常只服务一个
Client；远程 HTTP Server 可服务多个。Host 为每台 Server 创建独立 Client
——正是 M6 管理器为每条配置建一个 `MCPConnection` 的协议依据。

## 2. 双层协议

**数据层**（JSON-RPC 2.0）：

- 请求带 `id` 期待响应；通知（notification）无 `id` 不需要响应；
- 方法名形如 `tools/list`、`tools/call`、`resources/read`、`prompts/get`；
- 全部消息 UTF-8 编码。

**传输层**（对上层透明）：

| 传输 | 机制 | 适用 |
|------|------|------|
| **stdio** | 服务器作为子进程；stdin/stdout 收发换行分隔的 JSON-RPC；**stdout 禁止非协议输出**，日志走 stderr | 本地，零网络开销 |
| **Streamable HTTP** | 单一端点（如 `/mcp`）；POST 发消息（响应可为 JSON 或 SSE 流）；GET 可开服务器推送流；`MCP-Session-Id` 会话管理 | 远程，多客户端 |

同一套 JSON-RPC 消息可跑在任意传输上——框架因此能把传输选择封装成
工厂（M4），上层完全不感知差异。

## 3. 服务端三大原语

| 原语 | 消费者 | 方法 | 框架模型 |
|------|--------|------|----------|
| **Tools** | 模型（决定何时调用） | `tools/list` → `tools/call` | `ToolDescriptor`（含 JSON Schema） |
| **Resources** | 应用（决定放进上下文） | `resources/list` → `resources/read` | `ResourceDescriptor`（URI 定位，支持模板） |
| **Prompts** | 用户（手动触发） | `prompts/list` → `prompts/get` | `PromptDescriptor`（参数占位） |

客户端原语 **Elicitation**（服务器反向向用户补问信息）现行保留；
Sampling / Logging 原语在 2026-07-28 新版规范中已弃用——本框架未实现，
接口层也未为其预留，待新版协议稳定后再评估。

## 4. 生命周期

稳定版（2025-11-25，本框架实测的协商版本）：

1. **initialize**：客户端发能力声明 → 服务器回 `capabilities + serverInfo +
   protocolVersion`——即 `MCPConnection.ensure_connected()` 里的显式协商，
   结果落在 `ServerInfo`；
2. **消息交换**：请求-响应 + 通知；
3. **工具变更通知**：`toolsListChanged` → 客户端重新 `tools/list`
   （框架的对应语义：`get_tools(refresh=True)` 覆盖式重发现）；
4. **shutdown**：stdio 关 stdin → 服务器退出（SDK 处理进程终止梯度）；
   HTTP 发 DELETE 结束会话。

新版（2026-07-28，无状态化趋势）：`initialize` 握手将被 `server/discover`
发现机制取代，通知改订阅制。框架的对策：M2 接口只暴露
「ensure_connected / list / call / close」，不暴露握手细节——协议演进时
只改 M5/M4 的实现，上层不动。

## 5. 安全要点

- stdio：stdout 是协议通道，服务器调试信息必须走 stderr
  （`StdioTransport` 的 `errlog` 默认转发 `sys.stderr`）；
- HTTP 本地部署：仅绑定 `127.0.0.1`，并校验 `Origin` 头防 DNS 重绑定
  （`MCPServerBuilder` 默认 host=127.0.0.1；对外暴露需自行加认证）；
- HTTP 鉴权：标准头（`Authorization: Bearer …`），token 建议配置里写
  `${VAR}` 插值，不落盘（M1）；
- 会话 ID：服务器分配的 `MCP-Session-Id` 需防劫持（SDK 处理，框架在
  `StreamableHttpTransport.session_id` 透出查询）。

## 6. 协议 → 框架落点速查

| 协议概念 | 框架落点 |
|----------|----------|
| Host / Client / Server 三角色 | 应用 / `MCPConnection` / `MCPServerBuilder` |
| JSON-RPC 请求-响应 | 官方 SDK `ClientSession`（框架不自造） |
| initialize 能力协商 | `MCPConnection.ensure_connected()` → `ServerInfo` |
| `tools/list` / `tools/call` | `get_tools()` / `call_tool()`（限定名路由） |
| `resources/read`（含模板） | `read_resource()`（M7 模板通配匹配） |
| `prompts/get` | `get_prompt()` |
| `toolsListChanged` | `get_tools(refresh=True)` 覆盖式重发现 |
| stdio 通道纯净性 | `StdioTransport` errlog 转发 stderr |
| Streamable HTTP 会话 | `StreamableHttpTransport`（`MCP-Session-Id` 可查） |
| 本地部署仅绑 localhost | `MCPServerBuilder(host="127.0.0.1")` 默认值 |

调研来源：modelcontextprotocol.io 官方规范（2025-11-25 稳定版 +
2026-07-28 新版）、官方 python-sdk（24.8k stars）、PrefectHQ/fastmcp
（28k+ stars）、langchain-mcp-adapters。
