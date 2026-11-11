# 分步引导 · Step 5：端到端示例与测试（收官）

> 系列总纲见 [`../01-architecture.md`](../01-architecture.md)。前四步的模块
> 在本步组装成可交付的闭环：示例照抄即用、测试一键回归。

## 当前目标

框架代码写完 ≠ 可用。最后一步要回答使用者的三个实际问题：

1. **「我该怎么跑？」** —— 提供从配置到调用的最小可抄示例（客户端两种
   形态 + 服务端 + 配置样例）；
2. **「怎么知道没被我改坏？」** —— 单元测试（零 SDK 依赖的模块各测各的）
   + 端到端测试（真实子进程 / 真实 HTTP 服务器走完整协议），双模式
   运行：`python -m` 自带 runner（无需装 pytest）或 pytest 收集；
3. **「下一步往哪扩展？」** —— 本步结尾给出扩展点清单（自定义传输 /
   注册表 / LangChain 桥接等）。

## 产出内容

| 文件 | 内容 |
|------|------|
| `examples/servers/demo_server.py` | 示范服务端（Step 4 已产出）：3 工具 + 2 资源（静态 + 模板）+ 2 提示，`--stdio` / `--http` 双模式 |
| `examples/client_stdio.py` | 单服务器全生命周期：配置 → 懒发现 → 调用 / 读资源 / 渲染提示 → OpenAI 工具格式 → 优雅关闭 |
| `examples/client_multi.py` | 多服务器聚合：失败隔离、同名消歧、运行期 `add_server`、懒路由 |
| `examples/config.example.json` | 配置样例：一台 stdio + 一台 streamable-http（token 用 `${VAR:-default}` 插值） |
| `tests/test_config.py` | M1：解析 / 插值 / 五类校验失败 / 重复名 / 文件加载 / 合并优先级（10 项） |
| `tests/test_transport.py` | M4：分发 / 别名 / 类型不匹配防御 / **自定义传输注册扩展点** / async CM 语义（6 项） |
| `tests/test_registry.py` | M7：覆盖式登记 / 短名与多义 / 模板通配 / 清场（5 项） |
| `tests/test_end_to_end.py` | **端到端**：stdio 全链路（含模板资源）、HTTP 全链路（随机端口拉起真实服务器）、失败隔离 + 懒路由、关闭语义（4 项） |
| `02-protocol-concepts.md`（同目录） | 协议概念速览 + 协议→框架落点速查表 |

测试运行方式（项目根目录，hello-agent 环境）：

```bash
python -m cn.mcheat.mcp.tests.test_config       # 10/10
python -m cn.mcheat.mcp.tests.test_transport    #  6/6
python -m cn.mcheat.mcp.tests.test_registry     #  5/5
python -m cn.mcheat.mcp.tests.test_end_to_end   #  4/4（真实子进程 + HTTP 服务器）
```

## 与架构图的对应关系

- 端到端测试 = **分层架构图 L0→L3 全栈打通**：应用（测试充当 Host）
  → M6 管理器 → M7 路由 → M5 连接 → M4 传输 → SDK → JSON-RPC →
  框架自己写的 M8 服务端——**框架客户端连框架服务端的闭环**；
- 两种传输在端到端里各走一遍：`test_stdio_full_cycle`（架构图时序图
  stdio 分支）与 `test_http_full_cycle`（Streamable HTTP 分支，覆盖
  `MCP-Session-Id` 会话建立）；
- 单元测试对应「零依赖分层」表格：config / transport / registry 三组
  测试**不 import mcp SDK**——防腐层（M3）与接口抽象（M2）的价值在
  这里兑现：核心逻辑可以在无 SDK 环境快速回归；
- `test_custom_transport_registration` 直接验证架构文档承诺的扩展点：
  「新增自定义传输只需实现 Transport 接口并注册」。

## 验证结果（全部通过）

```text
test_config        10/10
test_transport      6/6
test_registry       5/5
test_end_to_end     4/4   # stdio 全链路 / HTTP 全链路 / 失败隔离+懒路由 / 关闭语义

examples/client_stdio.py   # 单服务器：42.0 / 资源 / 提示 / OpenAI 格式 / 关闭
examples/client_multi.py   # 多服务器：6 工具聚合 / broken 隔离 / 消歧 / add_server
```

开发过程中由验证驱动的修复（防腐层的直接收益，均只改动一处）：

- SDK `stdio_client` yield 二元组（新版）而非三元组 → 改 `transport/stdio.py`；
- SDK 结果字段为 camelCase（`serverInfo` / `uriTemplate`）→ 改 `connection.py`；
- SDK `ClientSession.__aenter__` 不自动 initialize → 显式协商；
- SDK `FastMCP.run()` 不接受地址参数，绑定地址须构造期传入 → 改 `builder.py`。

## 扩展点（框架下一步可以做什么）

| 想做的事 | 入口 | 说明 |
|----------|------|------|
| 自定义传输（gRPC / WebSocket） | `register_transport("grpc", MyTransport)` | 实现 `core/interfaces.Transport` 即可，配置里直接用 |
| 替换聚合 / 路由策略 | `MultiMCPManager(registry=MyRegistry())` | 实现 `CapabilityRegistry`（如按前缀路由、带权限过滤） |
| mock 连接做上层单测 | `MultiMCPManager(connection_factory=lambda cfg: FakeConn())` | 实现 `ServerConnection` |
| LangChain / OpenAI 桥接 | `ToolDescriptor.to_openai_tool()` 已备好 | 参照 `langchain-mcp-adapters` 的模式包一层 |
| 连接多来源合并 | `MCPClientSettings.merge()` | 文件 < 环境变量 < 代码覆盖的优先级语义 |
| 资源订阅（notifications/subscribe） | 待官方 SDK 稳定 | 2026-07-28 新版为订阅制，暂未纳入 |

## 收官自查（对照架构图）

- [x] M1~M8、X1~X2 全部模块有代码、有文档、有测试
- [x] 三张 Mermaid 图（分层 / 时序 / 数据流）与实际目录一一对应
- [x] 五篇引导文档均含「目标 → 产出 → 架构对应 → 验证」四段
- [x] 双传输端到端通过；失败隔离与资源零泄漏有测试覆盖
- [x] 零依赖模块可独立测试；SDK 依赖集中在 M4/M5/M8
