# 可插拔 RAG 框架（离线索引 + 在线检索）

一个**零硬依赖**、面向生产权衡的 RAG 框架，实现位置：`Hello-Agent/cn/mcheat/rag/`。

设计目标来自 [`doc/rag编码注意事项.md`](../../../doc/rag编码注意事项.md) 的 5 条权衡点；
框架把每条都做成**可验证、可量化、可替换**的具体组件，而不是写在注释里的口号。

- 核心链路只使用 Python 3.10+ **标准库**，不需要 API key、不需要 numpy；
- 所有关键组件都是 `abc.ABC` 接口 + 具名注册表，改配置或注入对象即可替换；
- 离线索引与在线检索可分成两个进程，通过 `IndexArtifact` 传递状态；
- 自带 Golden Set 回归评测与 CI 门禁、完整异常体系、122 个单元测试。

---

## 1. 快速开始

从**项目根目录**运行（保证 `cn.mcheat.rag` 可导入）：

```bash
python -m cn.mcheat.rag.examples.demo            # 一站式演示，直接可跑
python -m cn.mcheat.rag.examples.offline_build   # 离线建索引 -> examples/.rag_index/
python -m cn.mcheat.rag.examples.online_query "父子切分解决什么问题？" --show-prompt
python -m cn.mcheat.rag.examples.run_evaluation --save-baseline  # 生成基线
python -m cn.mcheat.rag.examples.run_evaluation --gate            # 回归门禁
```

最小代码：

```python
from cn.mcheat.rag import RAGSystem

system = RAGSystem()
system.index(["cn/mcheat/rag/examples/data"])   # 离线：单个目录吃下 5 种格式
system.save("./.rag_index")                     # 落盘（含配置快照）

system = RAGSystem.load("./.rag_index")          # 在线：另一个进程直接加载
result = system.retrieve("Top-K 为什么不是越大越好？")

if result.answer_ready:                         # 只有这种情况才允许调用 LLM
    answer = your_llm.invoke(result.prompt)     # prompt 里只含精排后的 3~5 条
else:
    answer = result.fallback.message            # 转人工 / 如实说「没找到相关信息」
```

> **必须遵守** `answer_ready` 判定：兜底链路判定为转人工或证据不足时，
> `result.prompt` 被置为空，调用方直接返回 `result.fallback.message`，
> 否则会产生「看似流畅、毫无依据」的幻觉答案。

---

## 2. 《注意事项》逐条落实

| # | 原文要点 | 落实位置 | 具体机制 |
|:-:|---|---|---|
| 1 | Top-K 不是越大越好 | `retrieval/hybrid.py`、`rerank.py`、`pipeline.py` | 向量 + BM25 各召回 Top-K → RRF 融合成 **Top-30~50 粗候选** → Reranker **精排到 Top-3~5**；编排层强制截断，只有精排结果进 `prompt` |
| 2 | chunk 大小两难 | `indexing/splitters.py`、`retrieval/context.py` | **Small-to-Big**：只 embedding 小子块（准），命中后按 `parent_id` 从 `ParentStore` 取回大段落（全），并按父块去重 |
| 3 | Lost in the Middle | `retrieval/context.py` | `edges` 策略把最相关的放**开头**、次相关放**结尾**、最不相关放中间（U 形分布）；`max_context_chars` 超预算时优先砍中间；prompt 末尾重复问题强化近因 |
| 4 | 用指标量化 | `evaluation/` | **Context Recall 在粗召回算、Context Precision 在精排后算**；Golden Set 批量回归 + 基线比对 + CI 门禁；提供 `noop` 精排 / `null` 稀疏通路 / `fixed` 切分等消融实现 |
| 5 | 兜底链路 | `retrieval/fallback.py` | 有序策略链：**扩 Top-K 重试**（有次数 / Top-K 双上限）→ **转人工** → **如实说「没找到相关信息」**；触发原因与每步动作全部记进 `diagnostics` |

### 2.1 第 1 条：两段式 Top-K

```
query ──> dense(Embedder + VectorStore) ── Top-K ─┐
      └─> sparse(BM25)                 ── Top-K ─┴─> RRF 融合去重 ─> coarse (≤50)
                                                                      │
                                                      Reranker.rerank(top_n=5)
                                                                      │
                                                                      v
                                                            final (≤5) -> prompt
```

`coarse_top_k` / `rerank_top_n` 是两个独立配置，且有 `rerank_top_n <= coarse_top_k`
的硬校验。粗召回结果保留在 `RetrievalResult.coarse` 中，**只用于诊断和评测**，
永远不会被拼进 prompt。

> 默认 Reranker 是零依赖的 `LexicalOverlapReranker`（词面覆盖率 + 词项聚集度 +
> 融合名次先验）。**它是占位实现，不是语义模型**；生产环境请切换为
> `components.reranker.name = "cross_encoder"`（见第 6 节）。

### 2.2 第 2 条：父子切分不变量

`CharacterParentChildSplitter` 保证每个子块都满足：

```python
parent.text[child.start:child.end] == child.text
```

即子块一定是父块的连续子串，`test_04_splitting.py` 对此有断言。子块带
`child_overlap`，避免一句话被切在边界上；切分顺序是「段落 → 句末标点 → 空格 →
单字 → 定长硬切」，任何输入都不会静默丢内容。

### 2.3 第 3 条：U 形上下文编排

`placement="edges"`（`n=5` 时展示顺序为 `[0, 2, 4, 3, 1]`，括号里是相关性名次）：

```
[最相关] ... [最不相关] ... [次相关]        高相关在两端，低相关埋在中间
```

`ContextBuilder` 返回 `placements`（位置 → 相关性名次）可供断言/可视化；
`SequentialContextBuilder`（`placement="sequential"`）是消融基线，用同一套
Golden Set 就能量化对比两种编排的收益。

### 2.4 第 4 条：分阶段度量

| 指标 | 公式 | 度量阶段 |
|---|---|---|
| Context Recall | `\|检索到 ∩ 相关\| / \|相关\|` | `result.coarse`（粗召回） |
| Context Precision | `\|检索到 ∩ 相关\| / \|检索到\|` | `result.final`（真正进 prompt） |
| Average Precision / MRR / HitRate / F1 | 见 `evaluation/metrics.py` | 精排后的有序列表 |

Recall 低 → 改 embedding / BM25 / 切分；Precision 低 → 改精排 / Top-N / 兜底阈值。
`EvalReport.regression_check()` 比较 4 项门禁指标，跌幅超过
`evaluation.max_regression_drop` 即判定失败，可直接作为 CI 退出码。

### 2.5 第 5 条：可观测的降级梯度

```python
result.fallback.triggers        # 为什么触发：召回不足 / 精排分低 / 上下文为空 ...
result.fallback.steps           # 每一步动作：widen_top_k -> human_handoff -> honest_no_answer
result.diagnostics["retry_history"]  # [{from_top_k, to_top_k, reason}, ...]
```

单路召回**抛异常**时不会中断链路：记录 `degraded` 后继续用另一路；
两路都异常时抛内部信号 `EmptyRecallError`，由兜底链路接管。

---

## 3. 目录结构

```
cn/mcheat/rag/
├── README.md                  本文档
├── requirements.txt           可选依赖清单（核心零依赖）
├── __init__.py                公共 API + 启动时注册内置组件
├── api.py                     RAGSystem 门面：index / retrieve / evaluate / save / load
├── config.py                  配置管理（默认值→文件→环境变量→代码覆盖 + 校验）
├── models.py                  领域模型（Document / ParentChunk / ChildChunk / ...）
├── exceptions.py              分阶段异常体系
├── _utils.py                  稳定哈希、中英文分词、向量运算、日志
├── configs/defaults.json      默认配置（唯一的参数入口）
├── core/
│   ├── interfaces.py          全部 ABC 抽象接口 / 扩展点
│   ├── registry.py            具名组件注册表 + "@组件名" 依赖注入
│   └── bootstrap.py           内置实现自注册（幂等）
├── loaders/                   5 种 Document Loader + 目录组合器 + 后缀分发
│   ├── base.py / text_loader.py / json_loader.py / csv_loader.py
│   └── html_loader.py / pdf_loader.py
├── cleaning/                  7 步可组合清洗流水线 + 清洗报告
├── indexing/                  父子切分 / ParentStore(内存 & SQLite) / OfflineIndexer
├── embeddings/                HashingEmbedder（默认）+ LangChain 适配器
├── vectorstores/              InMemoryVectorStore + FileBackedVectorStore
├── retrieval/                 在线检索全链路
│   ├── recall.py / bm25.py / hybrid.py / rerank.py
│   └── context.py / fallback.py / pipeline.py
├── evaluation/                GoldenSet / metrics / EvalRunner / 回归门禁
├── examples/                  示例语料 + 4 个可执行脚本 + golden_set.json
└── tests/                     122 个 unittest 测试（pytest 亦可运行）
```

---

## 4. 5 种 Document Loader 与清洗流程

| Loader | 后缀 | 关键处理 |
|---|---|---|
| `TextLoader` | `.txt .log .md .markdown .rst` | 编码回退、Markdown front matter → metadata、首标题提取、可选按空行分段 |
| `JsonLoader` | `.json .jsonl .ndjson` | 顶层 list / `data,items,records,documents` 展开、`text_template` 控制正文、短标量并入 metadata |
| `CsvLoader` | `.csv .tsv` | 一行一文档、列名语义拼接、BOM / 引号内换行 / 空行处理 |
| `HtmlLoader` | `.html .htm .xhtml` | 标准库 `HTMLParser`，剔除 script/style/nav/footer 等非正文，抽取 title 与 description |
| `PdfLoader` | `.pdf` | 一页一文档；默认懒加载 `pypdf`，**可注入 `text_extractor`** 从而完全免依赖 |

`DirectoryLoader` 是**组合器**（不是第 6 种格式）：递归遍历目录并按后缀分发，
跳过隐藏文件与未知后缀，`strict=True` 时改为报错做数据质量卡点。

清洗流水线默认 7 步（`cleaning.steps` 可增删、可注册自定义 Cleaner）：

```
unicode_normalize → control_chars → html_strip → whitespace
    → boilerplate → drop_empty → deduplicate
```

顺序有讲究：**清洗必须在切分之前**，否则文本长度变化会让子块的
`start/end` 偏移失真。`DeduplicateCleaner` 的状态按批次重置，重复建索引不会
误删文档；`CleaningReport` 精确记录是哪个步骤丢弃了哪篇文档。

---

## 5. 可插拔设计

### 5.1 接口清单（`core/interfaces.py`）

`DocumentLoader`、`Cleaner`、`ParentChildSplitter`、`Embedder`、
**`VectorStore`（可插拔向量库）**、`RecallRetriever`（召回通路）、
`FusionStrategy`、`Reranker`、`ParentStore`、`ContextBuilder`、
`FallbackStrategy`、`CrossEncoderScorer`（Protocol）、`Stateful`（持久化 mixin）。

`Stateful` 让组件随 `RAGSystem.save()` 一起导出；自带持久化的外部组件覆写
`export_state()` 返回 `{"external": True, ...}` 即可。

### 5.2 三种替换方式

**① 改配置名**（最常用，`configs/defaults.json` 或用户 JSON）：

```json
{
  "components": {
    "vector_store": {"name": "file", "params": {"path": "./.rag_index/vectors.json"}},
    "parent_store": {"name": "sqlite", "params": {"path": "./.rag_index/parents.db"}},
    "fusion":       {"name": "weighted", "params": {}}
  }
}
```

> 切换 `name` 时框架会**自动丢弃旧实现继承来的 params**，只保留为新实现显式
> 写的参数，避免 `TypeError`；同名实现的局部调参仍然走深度合并。

**② 注册自定义实现**：

```python
from cn.mcheat.rag.core.registry import VECTOR_STORE_REGISTRY
from cn.mcheat.rag.vectorstores import InMemoryVectorStore

@VECTOR_STORE_REGISTRY.register("my_store")
class MyVectorStore(InMemoryVectorStore):
    def search(self, query_vector, k, metadata_filter=None):
        ...  # 换成 FAISS / Milvus / pgvector / 自研 ANN
```

然后 `"vector_store": {"name": "my_store", "params": {...}}`。注册表会做
`isinstance` 契约校验，鸭子类型写错方法名会立刻报出明确错误。

**③ 直接注入实例**（适合接已有对象 / 测试替身）：

```python
system = RAGSystem(overrides={"vector_store": MyVectorStore()})
```

配置还支持 `"@组件名"` 引用做依赖注入：

```json
"dense_retriever": {
  "name": "embedding",
  "params": {"embedder": "@embedder", "vector_store": "@vector_store"}
}
```

构建顺序由 `core.registry.BUILD_ORDER` 保证。

---

## 6. 生产化：替换占位实现

默认实现是为了「零依赖、可复现、开箱即跑」，**不是语义最优**：

| 组件 | 默认值 | 生产建议 |
|---|---|---|
| `embedder` | `hashing`（特征哈希，只捕捉词面重叠） | `langchain` + `HuggingFaceEmbeddings` / OpenAI / DashScope |
| `reranker` | `lexical`（词面覆盖率） | `cross_encoder` + `sentence-transformers` |
| `vector_store` | `memory`（暴力余弦） | `file`（小库）/ `langchain_chroma` / 自研 FAISS 适配器 |
| `parent_store` | `memory` | `sqlite`（父块量大 / 在线进程直读） |

```json
{
  "components": {
    "embedder": {"name": "langchain", "params": {
      "provider": "huggingface",
      "model": "sentence-transformers/all-MiniLM-L6-v2"
    }},
    "reranker": {"name": "cross_encoder", "params": {
      "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }}
  }
}
```

也可以直接注入对象，彻底绕开第三方 import：

```python
from cn.mcheat.rag.retrieval.rerank import CrossEncoderReranker
from cn.mcheat.rag.core.registry import ALL_REGISTRIES

reranker = CrossEncoderReranker(scorer=my_cross_encoder)  # 任何有 predict(pairs) 的对象
system = RAGSystem(overrides={"reranker": reranker})
```

> **换 embedding 模型 / 维度必须重建索引**：`VectorStore` 会抛
> `DimensionMismatchError` 并在消息里说明原因，防止新旧向量混用导致检索静默失真。

---

## 7. 配置管理

优先级：`configs/defaults.json` < 用户 JSON < `RAG_*` 环境变量 < `overrides` 字典。

```python
from cn.mcheat.rag import RAGConfig

cfg = RAGConfig.load(
    path="my_rag.json",                          # 可选
    overrides={"retrieval": {"coarse_top_k": 30, "rerank_top_n": 3}},
)
cfg.validate()                                  # 硬约束 + 建议区间软告警
```

`validate()` 会拒绝：`child_size > parent_size`（破坏父子子串不变量）、
`rerank_top_n > coarse_top_k`、两路召回全关、非法位置策略等；偏离文档建议区间
（粗召回 30~50 / 精排 3~5）时打 `WARNING`，方便做消融实验而不误调参。

高频环境变量：`RAG_COARSE_TOP_K`、`RAG_RERANK_TOP_N`、`RAG_MIN_RECALL_HITS`、
`RAG_FALLBACK_ENABLED`、`RAG_RERANKER`、`RAG_EMBEDDER`、`RAG_VECTOR_STORE`、
`RAG_CONTEXT_PLACEMENT`、`RAG_PARENT_SIZE`、`RAG_CHILD_SIZE`、
`RAG_CHILD_OVERLAP`、`RAG_LOG_LEVEL`。完整映射见 `config.ENV_OVERRIDES`。

> JSON 不支持注释，允许用 `_` 前缀的键写说明（会被忽略）。

---

## 8. 错误处理

`RAGError` 为根，按阶段分层：`ConfigError`、`LoaderError` /
`UnsupportedSourceError`、`CleaningError`、`IndexingError`、
`VectorStoreError` / `DimensionMismatchError`、`RetrievalError` /
`EmptyRecallError`、`PersistenceError`、`DependencyMissingError`、
`EvaluationError`。

- 所有报错都带**上下文 + 修复建议**（例如缺依赖直接给 `pip install ...`）；
- `cleaning.on_error = "raise" | "skip"`：离线建索引默认严格暴露数据问题，
  脏数据源可切 `skip` 并从 `CleaningReport.errors` 排查；
- Loader 用 `encoding → fallback_encodings` 逐个尝试，中文 GBK 文件不会让整批索引中断；
- 单路召回异常自动降级，空结果交给兜底链路，二者语义严格区分；
- 库代码不配置日志 handler（只挂 `NullHandler`），由应用侧决定输出。

---

## 9. 测试

测试用标准库 `unittest` 编写，**不需要安装 pytest**：

```bash
python -m unittest discover -s cn/mcheat/rag/tests -t . -p "test_*.py"
# 或安装 pytest 后：
pytest cn/mcheat/rag/tests -q
```

`tests/test_10_end_to_end.py` 对 5 条注意事项逐条做了端到端断言：
精排截断、父子子串不变量与去重、U 形位置、分阶段指标、兜底轨迹、
持久化与幂等重建、以及通过 `overrides` / 配置名替换组件。

---

## 10. 已知限制

1. 示例语料很小，粗召回会覆盖全部子块，因此 demo 里 `context_recall` 恒为 1；
   真实语料（几百条以上）才有区分度。
2. 默认 `hashing` embedder 与 `lexical` reranker 不是语义模型，只保证链路可跑、
   结果可复现；上线前按第 6 节替换并**重新跑 Golden Set 基线**。
3. `OfflineIndexer.build()` 是全量重建 + 裁剪过期 ID（`prune_stale`），
   未实现单文档增量更新；语料很大时可分批构建后合并，或实现带 `delete_by_source`
   的自定义 `VectorStore`。
4. `HashingEmbedder` 与 `InMemoryVectorStore` 是 O(N·D) 暴力检索，
   适合中小语料 / 演示 / 单测；大规模场景请接入 ANN 向量库。
5. 框架只负责检索与上下文组装，**不调用 LLM**；`result.prompt` 交给哪个模型、
   如何解析引用由应用侧决定。
