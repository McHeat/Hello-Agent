# MCP 整体架构图

> 本文是 `cn.mcheat.mcp` 集成框架的总纲。文中给出三类 Mermaid 图：
> **分层架构图**（组件构成）、**交互时序图**（调用关系）、**数据流图**（数据流向）。
> 后续所有分步引导文档（`guide/step-1..5.md`）都通过 **模块编号（M1~M8）**
> 回链到本文，保证「每一步」都能在架构图中找到自己的位置。

---

## 1. MCP 协议全景（外部视角）

MCP（Model Context Protocol，模型上下文协议）是 Anthropic 提出的开放标准，
用来把「工具 / 数据 / 提示模板」以统一协议暴露给大模型应用。

协议三个角色与两种标准传输：

- **Host（宿主）**：AI 应用程序（如 Claude Desktop、IDE、Agent 框架），协调一个或多个 Client；
- **Client（客户端）**：宿主内维护与某个 Server **一对一连接**的组件；
- **Server（服务器）**：向 Client 提供上下文与能力的程序，可本地（子进程）或远程（HTTP 服务）运行。

```mermaid
flowchart LR
    subgraph HOST["MCP Host（AI 应用，如 Agent / IDE）"]
        A1[业务逻辑]
        C1[MCP Client #1]
        C2[MCP Client #2]
        A1 --> C1
        A1 --> C2
    end

    S1["MCP Server A（本地子进程）<br/>tools / resources / prompts"]
    S2["MCP Server B（远程 HTTP 服务）<br/>tools / resources / prompts"]

    C1 -- "stdio（JSON-RPC over stdin/stdout）" --> S1
    C2 -- "Streamable HTTP（JSON-RPC over POST + SSE）" --> S2
```

要点：

1. 每个 Client ↔ Server 是**专用一对一连接**（Host 为每个 Server 创建独立 Client）；
2. 协议分**两层**：内层为数据层（JSON-RPC 2.0 消息 + 原语定义），外层为传输层
   （stdio / Streamable HTTP），同一套消息格式可跑在任意传输之上；
3. 服务端三大原语：**Tools**（可执行函数）、**Resources**（上下文数据源）、
   **Prompts**（交互模板）；客户端原语：**Elicitation**（向用户补问信息）。

---

## 2. 框架分层架构图（内部视角，核心）

`cn.mcheat.mcp` 定位为**基于官方 `mcp` SDK 的集成层**：不自造 JSON-RPC 协议，
而是解决「一个应用同时接入多个 MCP 服务器」的工程问题——配置化、注册表、
连接生命周期、跨服务器聚合与路由。

图中每个模块带编号（**M1~M8**），横切关注点编号 **X1~X2**：

```mermaid
flowchart TB
    subgraph L0["L0 应用层（Host / Agent）"]
        APP["业务应用<br/>只认统一工具清单与调用入口"]
    end

    subgraph L1["L1 集成层 —— cn.mcheat.mcp（本框架）"]
        M6["M6 MultiMCPManager<br/>多服务器聚合管理器<br/>（注册 / 并发发现 / 聚合 / 路由）"]
        M7["M7 AggregatedRegistry<br/>跨服务器工具注册表<br/>（名称路由 / 冲突处理）"]
        M5["M5 MCPConnection<br/>单服务器连接封装<br/>（懒连接 / 重试 / 优雅关闭）"]
        M2["M2 core/interfaces.py<br/>全部抽象接口<br/>（Transport / Connection / Registry）"]
        M4["M4 TransportFactory + 双传输实现<br/>stdio（子进程）/ Streamable HTTP"]
        M1["M1 配置层 config.py<br/>dataclass 模型 + JSON/YAML/env 加载"]
        M3["M3 models.py<br/>领域模型：工具/资源/提示描述符"]
        M8["M8 MCPServerBuilder<br/>服务端构建器<br/>（装饰器注册 / 双传输启动）"]
    end

    subgraph L2["L2 SDK 层 —— 官方 mcp 库"]
        CS["ClientSession（JSON-RPC 2.0 会话）"]
        FM["FastMCP（服务端高层 API）"]
        SDKSTD["stdio_client 传输"]
        SDKHTTP["streamablehttp_client 传输"]
    end

    subgraph L3["L3 协议层（MCP 规范）"]
        PROT["JSON-RPC 2.0 消息<br/>tools/list · tools/call · resources/read …"]
    end

    X1["X1 exceptions.py<br/>MCPError 分阶段异常体系"]
    X2["X2 _utils.py<br/>get_logger 日志命名空间"]

    APP -->|get_tools / call_tool| M6
    M6 --> M7
    M6 --> M5
    M5 -.->|依赖抽象| M2
    M4 -.->|实现接口| M2
    M6 --> M1
    M7 --> M3
    M6 --> M3
    M1 --> M5
    M5 --> M4
    M4 --> SDKSTD
    M4 --> SDKHTTP
    SDKSTD --> CS
    SDKHTTP --> CS
    M8 --> FM
    CS --> PROT
    FM --> PROT

    X1 -.被所有层引用.- L1
    X2 -.被所有层引用.- L1
```

**设计原则（与 `cn.mcheat.rag` 保持一致）：**

1. **接口抽象 / 依赖倒置**：`M2 core/interfaces.py` 集中定义全部抽象接口
   （`Transport`、`ServerConnection`、`CapabilityRegistry`）。上层编排（M5/M6/M8）
   **只依赖接口**，不依赖具体实现；`M2/M3/X1/X2` 本身零第三方依赖；
   `M4` 的两个实现文件才 import 官方 `mcp` SDK——缺依赖时只影响对应实现，
   不影响框架其余部分的 import 与测试。
2. **可插拔**：新增自定义传输（如 gRPC、WebSocket）只需实现 `Transport` 接口
   并在工厂注册；替换注册表策略只需实现 `CapabilityRegistry`。
3. **核心链路面向协议编程**：框架围绕 MCP 规范的「发现 → 协商 → 调用 → 关闭」
   生命周期抽象，不绑定具体握手细节（兼容规范演进，如新版 `server/discover`
   无状态化趋势）。

---

## 3. 交互时序图（客户端完整链路）

框架客户端链路 = 配置加载 → 懒连接 → 能力协商 → 工具发现 → 调用路由 → 优雅关闭：

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用(Host)
    participant Mgr as M6 MultiMCPManager
    participant Conn as M5 MCPConnection
    participant TF as M4 TransportFactory
    participant Sess as SDK ClientSession
    participant Srv as MCP Server

    Note over App,Mgr: ① 启动阶段（配置驱动，零连接开销）
    App->>Mgr: load_config("config.json") 注册多服务器
    Note right of Mgr: 仅解析配置并登记，不建立任何连接

    Note over App,Sess: ② 发现阶段（首次调用才连接 = 懒连接）
    App->>Mgr: get_tools()
    Mgr->>Conn: ensure_connected()
    Conn->>TF: create_transport(config)
    TF-->>Conn: stdio / http 传输上下文
    Conn->>Sess: initialize() 能力协商
    Sess-->>Conn: capabilities + serverInfo
    Mgr->>Sess: tools/list（asyncio.gather 并发 ×N 服务器）
    Sess->>Srv: JSON-RPC 请求
    Srv-->>Sess: 工具清单（name / description / inputSchema）
    Mgr-->>App: 聚合后的统一工具清单（带 server 来源）

    Note over App,Srv: ③ 调用阶段（名称路由到来源服务器）
    App->>Mgr: call_tool("add", {"a": 1, "b": 2})
    Mgr->>Mgr: M7 注册表查 name → 定位所属服务器
    Mgr->>Conn: call_tool(...)
    Conn->>Sess: tools/call
    Sess->>Srv: 执行工具函数
    Srv-->>App: content 结果（text / image / structured）

    Note over App,Mgr: ④ 关闭阶段（资源不泄漏）
    App->>Mgr: close()
    Mgr->>Conn: 逐个优雅关闭（AsyncExitStack 逆序释放）
    Conn-->>Srv: 终止子进程 / 断开 HTTP 会话
```

时序关键点：

- **懒连接**：`get_tools()` 首次触发才建连接，应用启动零开销；
- **并发发现**：多服务器 `tools/list` 用 `asyncio.gather` 并发，O(n) 连接数；
- **单点失败隔离**：某服务器连不上时收集错误、继续聚合其余服务器，
  结果中透出可诊断的 `ServerUnreachableError`（X1）；
- **AsyncExitStack**：所有传输上下文挂到栈上统一逆序释放，异常路径也不泄漏。

---

## 4. 数据流图（配置 → 注册表 → 应用）

```mermaid
flowchart LR
    subgraph SRC["配置来源"]
        F1["config.json / .yaml"]
        F2["环境变量<br/>MCHEAT_MCP_*"]
        F3["代码内 dict"]
    end

    M1["M1 配置层<br/>解析 + 校验 + 合并"]

    subgraph REG["注册表（元数据，无连接）"]
        R1["服务器表：<br/>name → ServerConfig"]
    end

    subgraph DISC["发现（懒触发，并发）"]
        D1["各服务器 tools/resources/prompts 清单"]
    end

    M7["M7 聚合注册表"]
    APP["应用统一视图"]

    F1 --> M1
    F2 --> M1
    F3 --> M1
    M1 -->|"servers: dict"| R1
    R1 -->|"ensure_connected + list"| D1
    D1 -->|"聚合并标注 server 来源"| M7
    M7 -->|"get_tools() / call_tool(name)"| APP
```

数据流说明：

1. **三类来源 → 统一 `MCPClientSettings`**：优先级为 文件 < 环境变量 < 代码覆盖
   （与 `cn.mcheat.rag` 的配置合并约定一致）；
2. **元数据与连接分离**：注册表先登记「服务器怎么连」，连接推迟到首次使用；
3. **聚合即标注**：跨服务器同名工具按 `server__tool` 规则消歧，调用时按名路由回源。

---

## 5. 模块编号 ↔ 目录 ↔ 引导步骤对照表

| 编号 | 模块 | 源码位置 | 依赖第三方 | 分步引导 |
|------|------|----------|-----------|----------|
| X1 | 异常体系 | `exceptions.py` | 无 | step-1 |
| X2 | 通用工具 | `_utils.py` | 无 | step-1 |
| M1 | 配置层 | `config.py` | 无（YAML 可选） | step-1 |
| M2 | 抽象接口 | `core/interfaces.py` | 无 | step-2 |
| M3 | 领域模型 | `models.py` | 无 | step-2 |
| M4 | 传输工厂与双传输 | `transport/{factory,stdio,http}.py` | mcp | step-2 |
| M5 | 连接封装 | `client/connection.py` | mcp | step-2 |
| M7 | 聚合注册表 | `registry/tools.py` | 无 | step-3 |
| M6 | 多服务器管理器 | `api.py` | mcp（仅类型） | step-3 |
| M8 | 服务端构建器 | `server/builder.py` | mcp | step-4 |
| — | 示例与端到端 | `examples/` + `tests/` | mcp | step-5 |

> 表格的「依赖第三方」列体现零依赖分层：X1/X2/M1/M2/M3/M7 完全不依赖
> 官方 SDK，可单独单测；只有 M4/M5/M6/M8 触及 `mcp` 库。

---

## 6. 协议要点备忘（实现时对照）

| 协议要求 | 框架落点 |
|----------|----------|
| stdio：stdout 只许 JSON-RPC，日志走 stderr | M4 `StdioTransport` 捕获/转发子进程 stderr |
| stdio：消息以换行分隔、UTF-8 | 由官方 SDK 的 `stdio_client` 保证 |
| HTTP：`MCP-Session-Id` 会话管理、`MCP-Protocol-Version` 头 | 由官方 SDK 的 `streamablehttp_client` 保证 |
| HTTP：本地部署仅绑定 localhost、校验 Origin | M8 构建器默认 host=127.0.0.1 |
| initialize 能力协商 → tools/list → tools/call | M5 连接封装完整实现该生命周期 |
| 生命周期抽象需兼容规范演进（新版无状态 discover） | M2 接口不暴露握手细节，只暴露「ensure_connected / list / call / close」 |

（规范细节另见 `02-protocol-concepts.md`；调研来源：modelcontextprotocol.io
官方规范 2025-11-25 稳定版与 2026-07-28 新版、官方 python-sdk、
PrefectHQ/fastmcp、langchain-mcp-adapters。）
