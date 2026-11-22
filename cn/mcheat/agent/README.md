# 生产级 Python 学习 Agent 框架（基于 LangGraph）

一个面向生产的 LangGraph Agent 框架，实现位置：`Hello-Agent/cn/mcheat/agent/`，
与 `mcp/`、`rag/` 平级，遵循同一套工程范式：**可插拔组件 + 具名注册表 +
多级配置 + 分阶段异常 + 标准库 unittest**。

三类能力 + human-in-the-loop 意图澄清：

| 能力 | 意图路由键 | 技能实现 | 数据来源 |
|---|---|---|---|
| Python 语法知识查询 | `syntax` | `skills/syntax.py` | 内置 16 主题本地知识库（未命中如实说明） |
| Python 环境状态检测 | `env` | `skills/environment.py` | subprocess 探针（版本/解释器/pip/venv/包清单） |
| 代码修改指导 | `review` | `skills/review.py` | LLM 三段式评审（问题定位/建议/改进代码） |
| 意图澄清（human-in-the-loop） | `unclear` | `graph/nodes.py` clarify 节点 | `interrupt()` 暂停 → 用户回复 → `Command(resume)` 继续 |

---

## 1. 快速开始

从**项目根目录**运行（保证 `cn.mcheat.agent` 与根目录 `llm` 可导入）：

```bash
python -m cn.mcheat.agent.examples.cli            # CLI 多轮交互（需 .env 里的 key）
```

最小代码：

```python
from llm import get_model                          # 根目录模型工厂（.env 切换厂商）
from cn.mcheat.agent import PythonTutorAgent, ClarifyInterrupt

agent = PythonTutorAgent(llm=get_model())
outcome = agent.ask("列表推导式怎么写？", thread_id="user-1")

while isinstance(outcome, ClarifyInterrupt):       # 意图不明 -> 问用户 -> 继续
    print(outcome.question)                        # 含 1/2/3 候选意图
    outcome = agent.resume(input("你："), thread_id="user-1")

print(outcome.content)                             # 最终答案
```

测试（零 key 零网络）：

```bash
python -m unittest discover -s cn/mcheat/agent/tests -t . -p "test_*.py"
```

---

## 2. 节点设计与状态流转

```
START -> intent ─┬─(syntax)────> syntax        ──┐
                 ├─(env)──────> environment    ──┼─> answer -> END
                 ├─(review)───> review         ──┤
                 └─(unclear)──> clarify ─┬─(熔断)─┘
                                       └─(resume 回复)─> intent（重新分类）
```

| 节点 | 职责 | 关键实现 |
|---|---|---|
| `intent` | LLM 意图分类（四分类） | 结构化解析失败 / LLM 异常一律降级 `unclear`，绝不中断对话 |
| `syntax` / `environment` / `review` | 三个可插拔 Skill 分支 | 业务逻辑在组件里，节点只做编排 |
| `clarify` | human-in-the-loop 澄清 | `interrupt(payload)` 暂停；回复并入 `question` 与 `messages` 后回 `intent`；`clarify_turns >= max_clarify_turns` 时熔断（如实说明并结束） |
| `answer` | 确定性汇总 | 直接采用技能产出或熔断话术，不再调 LLM |

**状态（`TutorState`）**：`messages`（`add_messages` 追加，配合 checkpointer 支撑
多轮记忆）+ 每轮由 `ask()` 重置的 `question` / `intent` / `clarify_turns` /
`skill_output` / `answer`（避免上一轮残留污染本轮路由）。

**交互逻辑**：`ask()` 返回 `AgentAnswer`（完成）或 `ClarifyInterrupt`（需要澄清，
调用 `resume(reply)` 继续）；`interrupt`/`resume` 依赖 checkpointer + `thread_id`，
默认 `MemorySaver`，可注入 sqlite / postgres。

---

## 3. 目录结构

```
cn/mcheat/agent/
├── README.md                  本文档
├── requirements.txt           可选依赖说明（langgraph / langchain-core 主项目已装）
├── __init__.py                公共 API + 启动时注册内置组件
├── api.py                     PythonTutorAgent 门面：ask / resume / history / 装配
├── config.py                  配置管理（defaults.json → 用户 JSON → AGENT_* → overrides）
├── models.py                  领域模型（Intent / SkillOutput / ClarifyInterrupt / ...）
├── exceptions.py              分阶段异常体系（带修复建议）
├── _utils.py                  统一日志入口（NullHandler）
├── configs/defaults.json      默认配置（唯一的参数入口）
├── core/
│   ├── interfaces.py          全部 ABC：LLMClient / IntentClassifier / Skill /
│   │                          SyntaxKnowledgeBase / EnvProbe
│   ├── registry.py            具名注册表 + BUILD_ORDER + "@组件名" 依赖注入
│   ├── adapters.py            LangChainLLMClient 适配器 / LlmIntentClassifier
│   └── bootstrap.py           内置实现自注册（幂等；sqlite checkpointer 懒加载）
├── graph/
│   ├── state.py               TutorState（TypedDict + add_messages）
│   ├── nodes.py               intent / clarify / 技能包装 / answer 节点与路由函数
│   └── builder.py             StateGraph 组装 + compile(checkpointer)
├── skills/                    syntax / environment / review 三个 Skill 实现
├── tools/                     本地语法知识库（16 主题）+ 环境探针 + @tool 封装
├── examples/cli.py            CLI 多轮交互演示（human-in-the-loop 完整闭环）
└── tests/                     46 个 unittest（FakeLLM 零 key 零网络）
```

---

## 4. 可插拔设计

组件键（`BUILD_ORDER`）：`llm` → `intent_classifier` → `knowledge_base` →
`env_probe` → `skill_syntax` / `skill_env` / `skill_review` → `checkpointer`。

**① 改配置名**（`configs/defaults.json` 或用户 JSON）：

```json
{
  "components": {
    "checkpointer": {"name": "sqlite", "params": {"db_path": "./.agent_ckpt.db"}},
    "knowledge_base": {"name": "my_vector_kb", "params": {}}
  }
}
```

**② 注册自定义实现**：

```python
from cn.mcheat.agent.core.registry import KNOWLEDGE_BASE_REGISTRY
from cn.mcheat.agent.core.interfaces import SyntaxKnowledgeBase

@KNOWLEDGE_BASE_REGISTRY.register("my_vector_kb")
class MyVectorKB(SyntaxKnowledgeBase):
    def search(self, query): ...        # 接入 Chroma / FAISS 等
```

**③ 直接注入实例**（测试 / 已有对象最方便）：

```python
agent = PythonTutorAgent(
    llm=get_model(),                          # BaseChatModel 自动包装
    overrides={"knowledge_base": MyVectorKB()},
)
```

配置里的 `"@llm"` / `"@knowledge_base"` 是依赖引用，构建时自动解析成已装配
的组件实例（与 rag 框架同一机制）。

---

## 5. 生产化建议

| 场景 | 做法 |
|---|---|
| 会话持久化 | `PythonTutorAgent(llm=..., checkpointer=SqliteSaver(...))`，或配置 `checkpointer.name = "sqlite"`（需 `pip install langgraph-checkpoint-sqlite`） |
| 替换 LLM | 换模型只改根目录 `.env`（`llm.py` 工厂）；换实现注入任意 `LLMClient` |
| 语法知识库扩容 | 注册向量检索实现替换 `knowledge_base`；内置库未命中会**如实说明**，不产生幻觉 |
| 并发会话 | 每个会话独立 `thread_id`；MemorySaver 进程内线程安全，跨进程用 sqlite/postgres |
| 观测 | `logging.getLogger("cn.mcheat.agent").setLevel(logging.DEBUG)` 或 `AGENT_LOG_LEVEL=DEBUG` |

---

## 6. 配置

优先级：`configs/defaults.json` < 用户 JSON < `AGENT_*` 环境变量 < overrides 字典。

高频环境变量：`AGENT_MAX_CLARIFY_TURNS`、`AGENT_RECURSION_LIMIT`、
`AGENT_ENV_TIMEOUT`、`AGENT_PIP_LIST_LIMIT`、`AGENT_HISTORY_WINDOW`、
`AGENT_CHECKPOINTER`、`AGENT_LOG_LEVEL`。

`validate()` 硬校验：`max_clarify_turns >= 1`、`recursion_limit` 足以支撑澄清
循环（`>= 4 × (轮数+1)`）、`clarify.question_template` 含 `{candidates}` 占位符、
探针超时为正等；偏离建议区间（澄清轮数 > 3、pip 截断 > 50 行）打软告警。

---

## 7. 错误处理

`AgentError` 为根，按阶段分层：`ConfigError`、`DependencyMissingError`、
`LlmError`、`SkillError`、`ClarifyError`、`GraphError`。

- 所有报错带**上下文 + 修复建议**（如缺依赖直接给 `pip install ...`）；
- 意图分类 LLM 异常降级为 `unclear`（走澄清链路），不中断对话；
- `LlmError` 在技能层**原类型透出**，其余技能异常包成 `SkillError`；
- 探针 subprocess 全部带超时（默认 10s），单项失败记入 `issues` 友好呈现；
- 库代码不配置日志 handler（只挂 `NullHandler`）。

---

## 8. 测试

```bash
python -m unittest discover -s cn/mcheat/agent/tests -t . -p "test_*.py"
# 46 个用例：配置 / 注册表契约 / 技能 / 图路由 / 端到端 clarify 恢复
```

关键端到端断言：意图四路分发、`interrupt → resume → 重分类 → 技能执行` 全链路、
澄清熔断话术、多轮记忆累积、线程隔离、`LlmError` 透传。

---

## 9. 已知限制

1. 内置语法知识库是 16 主题的静态 dict（词面匹配），不是语义检索；
   大规模知识场景请替换 `knowledge_base` 组件；
2. 意图分类是单次 LLM 调用 + 容错解析，没有用 `with_structured_output`
   （保持 `LLMClient` 抽象与厂商无关）；追求更高准确率可在 `intent_classifier`
   注入更强的分类器（如 few-shot、微调小模型）；
3. `EnvSkill` 探测的是**当前进程**所在环境（`sys.executable`）；跨机器 / 容器
   场景请实现远程 `EnvProbe`；
4. `answer` 节点不调用 LLM（确定性汇总）；如需统一文风可扩展该节点。
