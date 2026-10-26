# LangGraph 学习框架

一个**渐进式、可读性强、面向生产**的 LangGraph 学习脚手架。基于 GitHub 高 star 项目的核心模式：

- [`langchain-ai/langgraph`](https://github.com/langchain-ai/langgraph)（~43k stars）—— 图编排框架本体
- [`langchain-ai/langgraph-supervisor-py`](https://github.com/langchain-ai/langgraph-supervisor-py)（~1.7k stars）—— 多智能体「监督者」模式

每个概念一个独立示例，从零到生产，逐步深入。

---

## 学习路线（7 步）

| 步骤 | 文件 | 学什么 | 需要 API key |
|------|------|--------|:---:|
| 1 | `step1_basic_graph.py` | 图基础：State / Node / Edge / 条件分支 | ❌ |
| 2 | `step2_chatbot.py` | 把 LLM 接进图：MessagesState、模型节点 | ✅ |
| 3 | `step3_tool_calling.py` | 工具调用 + ReAct 循环（agent 核心） | ✅ |
| 4 | `step4_memory.py` | 记忆：Checkpointer、多轮对话 | ✅ |
| 5 | `step5_human_in_the_loop.py` | 人工介入：interrupt / resume | ✅ |
| 6 | `step6_multi_agent.py` | 多智能体：Supervisor 监督者模式 | ✅ |
| 7 | `step7_production.py` | 生产化：create_agent + 流式 + 配置 | ✅ |

> 建议按编号顺序学习：每一步都建立在上一步的基础上，逐步引入新概念。

---

## 快速开始

### 1. 准备环境（已用 conda 完成）

```bash
conda activate hello-agent          # 项目专用 conda 环境（Python 3.13）
pip install -r requirements.txt     # 安装依赖（已装好，此命令用于复现）
```

### 2. 配置 API key

```bash
cp .env.example .env                # 复制模板
# 编辑 .env，填入你的 key，并选择提供商
```

```ini
LLM_PROVIDER=anthropic              # anthropic 或 openai
ANTHROPIC_API_KEY=sk-ant-...        # 用 Claude 时填
# 或
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...               # 用 GPT 时填
```

> 模型通过 `llm.py` 里的工厂层统一创建，**切换提供商只改 `.env`，代码零改动**。

### 3. 运行示例

从**项目根目录**运行（用 `python -m`，保证 `llm.py` 能被正确导入）：

```bash
python -m examples.step1_basic_graph      # 第一步，无需 key
python -m examples.step2_chatbot          # 从这步起需要 key
python -m examples.step3_tool_calling
# ...依次类推
```

---

## 核心概念速查

| 概念 | 一句话解释 | 在哪个 step 讲 |
|------|-----------|:---:|
| **State** | 图里流动的数据结构（`TypedDict`），节点靠它传数据 | 1 |
| **Node** | 接收 state、返回部分更新的函数 | 1 |
| **Edge** | 连接节点的「导线」，分无条件和条件两种 | 1 |
| **`add_conditional_edges`** | 按返回值把执行流分到不同节点 | 1 |
| **MessagesState** | 内置的、自动追加消息的 state | 2 |
| **`@tool` / `bind_tools`** | 把函数变成模型可调用的工具 | 3 |
| **ReAct 循环** | 模型「推理→调工具→看结果→再推理」的循环 | 3 |
| **Checkpointer** | 持久化状态，实现多轮记忆 | 4 |
| **`interrupt` / `Command(resume)`** | 暂停执行、等人批准后继续 | 5 |
| **`create_agent`** | 一行搭起带工具的 agent（LangGraph v1 推荐） | 6、7 |
| **`add_messages`** | 让自定义 state 里的消息「追加」而非覆盖 | 6 |
| **Supervisor 模式** | 主管分派、专家执行的编排范式 | 6 |
| **流式输出 / 配置** | `stream`、`recursion_limit` 等生产细节 | 7 |

---

## 目录结构

```
Hello-Agent/
├── README.md                 # 本文件：学习路线
├── requirements.txt          # 依赖清单
├── .env.example              # 环境变量模板（复制为 .env 填 key）
├── .gitignore
├── llm.py                    # 模型工厂：一处切换 Anthropic/OpenAI
└── examples/
    ├── step1_basic_graph.py      # 图基础（无 LLM）
    ├── step2_chatbot.py          # 第一个 LLM 节点
    ├── step3_tool_calling.py     # 工具 + ReAct 循环
    ├── step4_memory.py           # 检查点/记忆
    ├── step5_human_in_the_loop.py# 人工介入
    ├── step6_multi_agent.py      # 多智能体（Supervisor）
    └── step7_production.py       # 生产化（create_agent + 流式）
```

---

## 生产化进阶建议

学完 7 步后，往真正生产走的方向：

1. **换持久化检查点**：`MemorySaver` 换成 `langgraph-checkpoint-sqlite` 或 `langgraph-checkpoint-postgres`，服务重启不丢会话。
2. **用 `langgraph-supervisor` 库**：step6 是手写的教学版，生产上直接 `from langgraph_supervisor import create_supervisor` 一行搞定，含消息作用域等细节处理。
3. **可观测性**：接入 [LangSmith](https://smith.langchain.com/) 追踪每次运行的节点轨迹、token 消耗、延迟。
4. **部署**：LangGraph 提供 `langgraph-cli` + LangGraph Platform（或自托管）来部署成带流式 API 的服务。
5. **中间件**：`create_agent` 支持 `middleware`，可统一做重试、限流、审计。

---

## 常见问题

- **`ModuleNotFoundError: No module named 'llm'`**：请从项目根目录用 `python -m examples.stepN_xxx` 运行，而不是 `cd examples` 后直接跑。
- **Anthropic 返回 400 错误**：当前 Claude 系列模型不再接受 `temperature` 等采样参数，本项目的 `llm.py` 已刻意不传，请勿自行加回。
- **跑 step2 报 key 错误**：检查 `.env` 是否已创建并填好 key，且 `LLM_PROVIDER` 与实际填的 key 匹配。
