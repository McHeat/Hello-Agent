# 分步引导 · Step 3：聚合注册表与多服务器管理器（M7 / M6）

> 系列总纲见 [`../01-architecture.md`](../01-architecture.md)。本步把
> Step 2 的「单服务器连接」升级为「多服务器聚合」——框架对上层应用的门面。

## 当前目标

一个真实 Agent 往往同时接多台 MCP 服务器（本地 filesystem 子进程 + 远端
数据库服务 + 团队共享的搜索服务……）。此时上层会立刻遇到三个工程问题：

1. **并发与开销**：逐台串行发现又慢又脆——需要 `asyncio.gather` 并发、
   O(n) 连接数，且应用启动时不该先付全部连接成本（懒发现）；
2. **失败隔离**：一台连不上不应拖垮整体聚合——单点失败进 `failures`
   诊断清单，其余服务器照常服务；
3. **命名空间**：多台服务器都会叫 `add` / `search`——需要聚合命名
   （`server__tool`）+ 按名路由回源 + 唯一短名的便捷匹配。

这三个问题正是 M7（索引与路由）+ M6（并发编排与门面）的分工。

## 产出内容

| 文件 | 编号 | 职责 |
|------|------|------|
| `registry/tools.py` | M7 | `AggregatedRegistry`：覆盖式登记、限定名/唯一短名路由、多义名报错（列出候选）、按服务器清场 |
| `api.py` | M6 | `MultiMCPManager`：配置登记、`refresh()` 并发发现、`get_tools()` 懒发现、`call_tool()` 按名路由、`close()` 并发优雅关闭、`failures` 失败清单 |

上层 API（与 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 习惯对齐，
但返回带来源的领域模型而非 LangChain Tool）：

```python
manager = MultiMCPManager(load_config("config.json"))   # 或直接传 dict

tools = await manager.get_tools()                        # 并发发现 + 聚合
result = await manager.call_tool("math__add", {"a": 1, "b": 2})
result = await manager.call_tool("add", {"a": 1, "b": 2})  # 唯一短名也行
resources = await manager.list_resources()
messages = await manager.get_prompt("greet", {"name": "MCP"})
await manager.close()

manager.failures   # 发现阶段不可达的服务器：{name: MCPError}，可诊断可重试
```

关键设计：

- **懒发现**：不先 `get_tools()` 也能直接 `call_tool()`——注册表为空时
  自动发现一次再查（`_routing`），冷启动调用同样可用；
- **路由规则**：限定名精确匹配 → 唯一短名匹配 → 多义报错（错误信息
  直接列出 `a__add、b__add` 候选，用户照抄即可消歧）；
- **依赖注入**：`connection_factory` / `registry` 都从构造函数注入，
  manager 只依赖 `core/interfaces.py` 的抽象（单测注入 mock 连接，
  不拉真实子进程）；
- **部分失败语义**：工具可用而资源/提示拉取失败时保留工具、
  记部分失败（`failures` 键带 `(resources/prompts)` 后缀），不整台丢弃；
- **覆盖式登记**：`register_tools` 先清空该服务器旧条目再写入——
  对应 MCP `toolsListChanged` 后重新发现的覆盖语义。

## 与架构图的对应关系

- M6 是**分层架构图**里 L1 集成层的枢纽节点：`应用 → M6 → M7（路由）/ M5（连接）`
  三条边在本步全部接通，`M6 → M1` 的配置边即构造函数里的
  `MCPClientSettings` / dict 双入口；
- **时序图完整落地**：① 启动阶段（load_config 注册）→ ② 发现阶段
  （ensure_connected + tools/list，asyncio.gather 并发）→ ③ 调用阶段
  （M7 注册表查 name → 路由到对应连接 tools/call）→ ④ 关闭阶段
  （并发 close）；
- **数据流图**第 ② ③ 段（注册表元数据 → 并发发现 → 聚合并标注 server
  来源 → 应用统一视图）正是 `refresh()` → `AggregatedRegistry` →
  `get_tools()` 这条链；
- 失败隔离对应架构文档「性能与可靠性」一节：单服务器不可用不阻断聚合，
  错误透出为可诊断异常。

## 验证方式

已执行的验证（两台真实子进程 + 一台坏路径服务器）：

```text
聚合工具: ['good__add', 'good__get_weather']
failures: ['bad']                          # 坏服务器被隔离，good 正常服务
good__add -> 7.0                           # 限定名路由
短名 get_weather -> 北京 当前晴…           # 唯一短名匹配
双服务器工具数: 4
OK: 多义名消歧提示 -> …可用工具：a__add、b__add…
a__add -> 15.0
OK: 未知名 -> …可用工具：good__add、good__get_weather…
OK: 全部关闭
```

覆盖点：多服务器并发聚合、失败隔离（`failures` 透出）、限定名路由、
唯一短名路由、多义名消歧（错误列候选）、未知名报错（错误列可用清单）、
并发优雅关闭。

复现（项目根目录）：

```bash
python - <<'EOF'
import asyncio
from cn.mcheat.mcp.client import MultiMCPManager

async def main():
    m = MultiMCPManager({
        "good": {"transport": "stdio", "command": "python",
                 "args": ["examples/mcp_demo_server.py"]},
        "bad": {"transport": "stdio", "command": "python",
                "args": ["/nonexistent.py"]},
    })
    print([t.qualified_name for t in await m.get_tools()], m.failures.keys())
    print((await m.call_tool("good__add", {"a": 3, "b": 4})).text())
    await m.close()

asyncio.run(main())
EOF
```
