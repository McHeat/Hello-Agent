# 分步引导 · Step 4：服务端构建器（M8）

> 系列总纲见 [`../01-architecture.md`](../01-architecture.md)。前三步解决了
> 「作为客户端怎么用 MCP」，本步解决「作为服务提供方怎么暴露 MCP」。

## 当前目标

MCP 生态的价值在于「一次实现、处处可用」——把团队的能力（内部 API、
数据库查询、工具函数）包装成 MCP 服务器，任何兼容 Host（Claude、
IDE、Agent 框架，以及本框架 Step 3 的 `MultiMCPManager`）都能直接用。

官方 SDK 的 `FastMCP` 已经把 JSON-RPC 管道做了封装；本框架在其上补齐
工程化体验：

1. **装饰器统一入口**：`@server.tool` / `@server.resource("uri")` /
    `@server.prompt`，裸用法与带参用法都支持，Schema 从类型注解自动生成；
2. **双传输一键切换**：`run("stdio")`（宿主拉起子进程，本地场景）与
    `run("streamable-http")`（远程服务，默认只绑 `127.0.0.1`，协议安全建议）；
3. **零依赖 import**：装饰器只把函数记进内存清单，SDK 延迟到 `build()`
    / `run()` 才加载——定义阶段不依赖 `mcp`，静态检查友好；
4. **构建与运行分离**：`build()` 产出底层 `FastMCP` 实例（单测可直接
    检查注册结果），`run()` 才进入阻塞事件循环。

## 产出内容

| 文件 | 职责 |
|------|------|
| `server/builder.py` | `MCPServerBuilder`：三类装饰器 + `describe()` 自检 + `build()` 组装 + `run()` 双传输启动 |
| `examples/servers/demo_server.py` | 示范服务端：3 工具（add / multiply / get_weather）+ 2 资源（静态 list + 模板 `{name}`）+ 2 提示（greeting / review），`--stdio` / `--http` 启动 |

用法（见 demo_server）：

```python
from cn.mcheat.mcp import MCPServerBuilder

server = MCPServerBuilder("mcheat-demo", instructions="…")

@server.tool                            # 裸用法
def add(a: float, b: float) -> float:
    """两个数相加。"""                  # docstring 即工具描述
    return a + b

@server.resource("notes://{name}", name="get_note")   # 模板资源
def get_note(name: str) -> str: ...

@server.prompt(name="greeting")
def greeting(name: str) -> str: ...

server.describe()   # 构建前自检：{'tools': [...], 'resources': [...], 'prompts': [...]}
server.run("stdio")                  # 本地子进程模式
# server.run("streamable-http")      # 远程模式 http://127.0.0.1:8000/mcp
```

本步验证中发现并吸收的两个 SDK 差异（防腐层价值）：

- `@resource("notes://{name}")` 注册为**资源模板**（`list_resource_templates`，
  字段名 `uriTemplate`），与静态资源（`list_resources`，字段名 `uri`）
  分属两个列表——M5 的 `list_resources()` 已做合并，M7 的
  `lookup_resource()` 支持 `{param}` 通配段匹配，模板资源直接可读；
- 子进程模式下 `sys.path[0]` 是脚本目录而非项目根，示例服务端需把
  项目根插回 `sys.path` 才能 import `cn.mcheat.mcp`（demo_server 已处理）。

## 与架构图的对应关系

- M8 是**分层架构图** L1 集成层的右翼：`M8 → FastMCP → 协议层`，
  与客户端链路（M6 → M5 → M4 → ClientSession → 协议层）在 **L3 协议层
  汇合**——框架自己写的客户端连接自己写的服务端，正是端到端验证
  采用的闭环；
- 服务端三大原语与架构文档「协议要点备忘」表的对应：HTTP 模式默认
  绑定 `127.0.0.1`（本地部署安全建议）、日志走 stderr（stdio 模式
  stdout 纯净性硬约束）；
- `describe()` 体现「构建与运行分离」：不加载 SDK 即可拿到注册清单，
  对应分层架构图里 M8 的零依赖轮廓（虚线框：接口不依赖 SDK）。

## 验证方式

已执行的端到端验证（框架客户端 ↔ 框架服务端闭环，从项目根目录）：

```text
工具: ['demo__add', 'demo__get_weather', 'demo__multiply']
资源: ['notes://list', 'notes://{name}(模板)']
提示: [('greeting', ['name']), ('review', ['code'])]
add -> 42.0
multiply -> 42.0
weather -> 上海 当前晴，气温 22℃，适合外出。
note(模板) -> MCP 分数据层（JSON-RPC）与传输层（stdio / Streamable HTTP…
list_notes -> proto、layers
greeting -> 你好，MCP！这是来自 mcheat MCP 框架的问候。
review -> 请评审以下代码，从正确性、可读性、异常处理三方面给出意见：…
=== 全部通过 ===
```

覆盖点：三类装饰器注册、类型注解 → Schema、静态资源与模板资源、
提示参数占位、双模式启动入口（stdio 实测；HTTP 模式在 Step 5 用
`--http` 启动 + `streamable-http` 客户端配置实测）。

复现（项目根目录）：

```bash
python - <<'EOF'
import asyncio, sys
from cn.mcheat.mcp import MultiMCPManager

async def main():
    mgr = MultiMCPManager({"demo": {"transport": "stdio", "command": sys.executable,
        "args": ["cn/mcheat/mcp/examples/servers/demo_server.py"]}})
    print([t.qualified_name for t in await mgr.get_tools()])
    print((await mgr.call_tool("demo__add", {"a": 20, "b": 22})).text())
    await mgr.close()

asyncio.run(main())
EOF
```
