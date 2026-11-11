# 分步引导 · Step 1：异常体系、通用工具与配置层（X1 / X2 / M1）

> 系列总纲见 [`../01-architecture.md`](../01-architecture.md)。本步是框架地基：
> 先把「报错怎么说清楚」和「配置怎么进来」两件事定下来。

## 当前目标

任何框架动手写业务逻辑之前，先回答两个问题：

1. **出错时用户看到什么？** —— 没有统一异常体系，调用方就得 `except` 五花八门
   的内建异常（包括与 Python 内建 `ConnectionError` 同名冲突的坑），无法一网打尽；
2. **服务器从哪来？** —— 多服务器框架的第一输入是「有哪些服务器、怎么连」。
   配置解析与校验必须**前置**：让错误在建连之前、应用启动之时就暴露，
   而不是半夜调用工具时才报一个难懂的底层错误。

本步产出全部**零第三方依赖**——后续任何模块（乃至用户自己写的组件）都可以
安全 import，不受官方 `mcp` SDK 是否安装影响。

## 产出内容

| 文件 | 编号 | 职责 |
|------|------|------|
| `exceptions.py` | X1 | `MCPError` 基类 + 配置/传输/连接/发现/调用/服务端分阶段子类，错误信息统一含「发生了什么 + 哪个服务器 + 怎么修」 |
| `_utils.py` | X2 | `get_logger()`（`cn.mcheat.mcp` 日志命名空间）+ `require_module()`（可选依赖友好检查） |
| `config.py` | M1 | `ServerConfig` / `StdioServerConfig` / `HttpServerConfig` / `MCPClientSettings` + `load_config()` + `expand_env()` |

核心 API：

```python
from cn.mcheat.mcp.config import load_config, MCPClientSettings

settings = load_config("examples/config.example.json")   # 文件来源
settings = MCPClientSettings.from_dict({"math": {...}})  # 代码内 dict 来源
settings.servers["math"]                                 # → StdioServerConfig(name=...)
```

设计取舍：

- **dataclass 而非 pydantic**：与 `cn.mcheat.rag` 一致，核心零依赖；
  校验写在 `from_dict` 里，错误同样友好；
- **dict 的 key 即服务器名**：与 langchain-mcp-adapters 的 `MultiServerMCPClient`
  用法一致（你熟悉的风格），条目内不再写 `name`，避免两处不一致；
- **未知键直接报错**：多写的键（笔误）会静默失效，是最难排查的一类问题，
  故 fail-fast 而非宽容忽略；
- **`${VAR}` 插值**：`headers` 里的 token 写成 `"Bearer ${MCP_TOKEN}"`，
  敏感信息留在环境变量，配置文件可以安全提交版本库；
  未定义变量展开为空串并 warning（`${VAR:-default}` 可给默认值）；
- **YAML 可选**：只有真的传 `.yaml` 文件才需要 `pyyaml`，
  缺依赖时报带 `pip install` 提示的 `ImportError`，JSON 路径不受影响。

## 与架构图的对应关系

- X1 / X2 / M1 位于**分层架构图 L1 集成层**最底部，是所有其他模块的依赖项
  （横切关注点 X1/X2 被各层虚线引用）；
- 本步完成的是**数据流图第 ① 段**：`配置来源（JSON/YAML/env/dict）
  → M1 解析 + 校验 → MCPClientSettings`；
- 产出的 `MCPClientSettings.servers` 就是架构图中「注册表（元数据，无连接）」
  的数据来源——Step 3 的 `MultiMCPManager` 将逐条登记这些 `ServerConfig`；
- `UnsupportedTransportError` 的错误信息已经指向 `core/interfaces.Transport`
  （Step 2 产出）——异常文案即接口契约的文档。

## 验证方式

从项目根目录（Hello-Agent）运行：

```bash
python -m cn.mcheat.mcp.tests.test_config   # 单元测试（Step 5 产出，可先跳过）
python - <<'EOF'
from cn.mcheat.mcp.config import MCPClientSettings

# 1) 合法配置解析 + 环境变量插值
import os; os.environ["DEMO_TOKEN"] = "abc123"
s = MCPClientSettings.from_dict({
    "math": {"transport": "stdio", "command": "python", "args": ["x.py"]},
    "remote": {"transport": "streamable-http", "url": "http://127.0.0.1:8000/mcp",
               "headers": {"Authorization": "Bearer ${DEMO_TOKEN}"}},
})
assert s.servers["remote"].headers["Authorization"] == "Bearer abc123"
print("OK: 解析 + 插值")

# 2) 错误信息含「哪个服务器 + 怎么修」
try:
    MCPClientSettings.from_dict({"bad": {"transport": "stdio"}})
except Exception as e:
    print("OK:", e)   # 应提示 stdio 必须给出 command
EOF
```
