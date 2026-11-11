# MCP 集成框架（cn.mcheat.mcp）——基于官方 mcp SDK 的可插拔集成层

一个「多服务器 MCP 接入」框架：用一份配置声明多个 MCP 服务器
（本地 stdio 子进程 / 远程 Streamable HTTP），框架负责连接生命周期、
并发能力发现、跨服务器工具聚合与按名路由，对上层应用完全透明。
同时提供装饰器风格的服务端构建器，几行代码暴露 tools/resources/prompts。

## 架构总纲

**先读项目文档 [`doc/mcp/01-architecture.md`](../../../doc/mcp/01-architecture.md)**：
MCP 协议全景、框架分层架构图（模块编号 M1~M8）、客户端交互时序图、
数据流图，以及模块编号 ↔ 目录 ↔ 引导步骤对照表。

```
L0 应用层（Host / Agent）
 └─ L1 cn.mcheat.mcp 集成层   ← 本框架（M1 配置 / M2 接口 / M3 模型 /
     │                          M4 传输 / M5 连接 / M6 管理器 / M7 注册表 / M8 构建器）
     └─ L2 官方 mcp SDK（ClientSession / FastMCP / 双传输）
         └─ L3 MCP 协议（JSON-RPC 2.0）
```

## 分步引导索引（推荐按序阅读）

文档统一存放于项目 `doc/mcp/`（与 `cn.mcheat.rag` 的文档布局一致）：

| 步骤 | 文档 | 内容 | 覆盖模块 |
|------|------|------|----------|
| 1 | [`doc/mcp/guide/step-1-config.md`](../../../doc/mcp/guide/step-1-config.md) | 异常体系、通用工具、配置层 | X1 / X2 / M1 |
| 2 | [`doc/mcp/guide/step-2-transport.md`](../../../doc/mcp/guide/step-2-transport.md) | 抽象接口、领域模型、传输工厂、连接封装 | M2 / M3 / M4 / M5 |
| 3 | [`doc/mcp/guide/step-3-registry.md`](../../../doc/mcp/guide/step-3-registry.md) | 聚合注册表、多服务器管理器 | M7 / M6 |
| 4 | [`doc/mcp/guide/step-4-server.md`](../../../doc/mcp/guide/step-4-server.md) | 服务端构建器 | M8 |
| 5 | [`doc/mcp/guide/step-5-examples.md`](../../../doc/mcp/guide/step-5-examples.md) | 端到端示例与测试（收官） | 全部 |

协议概念速览另见 [`doc/mcp/02-protocol-concepts.md`](../../../doc/mcp/02-protocol-concepts.md)
（含协议→框架落点速查表）。每篇引导文档按「当前目标 → 产出内容 →
与架构图的对应关系 → 验证方式」四段结构撰写，与架构图模块编号一一对应。

## 快速开始

依赖（项目根 `requirements.txt` 已含 `mcp>=1.0`；YAML 支持可选装 `pyyaml`）。

客户端（见 `examples/client_stdio.py` / `examples/client_multi.py`）：

```python
from cn.mcheat.mcp import MultiMCPManager, load_config

manager = MultiMCPManager(load_config("examples/config.example.json"))
tools = await manager.get_tools()
result = await manager.call_tool("math__add", {"a": 1, "b": 2})
await manager.close()
```

服务端（见 `examples/servers/demo_server.py`）：

```python
from cn.mcheat.mcp import MCPServerBuilder

server = MCPServerBuilder("demo")

@server.tool
def add(a: float, b: float) -> float:
    """两数相加。"""
    return a + b

server.run("stdio")   # 或 "streamable-http"
```

## 测试

```bash
# 项目根目录（hello-agent 环境）；自带 runner，无需 pytest
python -m cn.mcheat.mcp.tests.test_config       # 10/10（零 SDK 依赖）
python -m cn.mcheat.mcp.tests.test_transport    #  6/6（零 SDK 依赖）
python -m cn.mcheat.mcp.tests.test_registry     #  5/5（零 SDK 依赖）
python -m cn.mcheat.mcp.tests.test_end_to_end   #  4/4（真实 stdio 子进程 + HTTP 服务器）
```

## 设计原则（与 cn.mcheat.rag 一致）

1. **接口抽象、依赖倒置**：`core/interfaces.py` 集中定义全部抽象接口，
   上层编排只依赖接口；X1/X2/M1/M2/M3/M7 零第三方依赖，可独立单测。
2. **可插拔**：自定义传输 / 注册表实现对应接口即可替换，无需改上层。
3. **配置驱动**：JSON / YAML / 环境变量三类来源，逐级合并 + 早校验。
4. **资源不泄漏**：所有连接挂 AsyncExitStack 统一逆序释放，异常路径覆盖测试。

## 目录

与 `cn.mcheat.rag` 布局对齐：顶层核心模块 + 领域子包 + `examples/` + `tests/`；
文档不在包内，统一放项目根 `doc/mcp/`。

```
cn/mcheat/mcp/
├── README.md / requirements.txt
├── api.py                  M6 门面：MultiMCPManager（对应 rag 的 RAGSystem）
├── config.py / exceptions.py / models.py / _utils.py
├── core/                   M2 抽象接口（interfaces.py）
├── transport/              M4 传输工厂与 stdio / streamable-http 实现
├── client/                 M5 连接封装（connection.py）
├── registry/               M7 跨服务器聚合注册表
├── server/                 M8 服务端构建器
├── examples/               示例（demo 服务端 + 客户端 + 配置样例）
└── tests/                  单元测试 + 端到端测试

Hello-Agent/doc/mcp/        全部框架文档（架构图 / 协议概念 / 分步引导）
```
