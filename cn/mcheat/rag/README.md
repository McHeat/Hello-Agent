# 可插拔 RAG 框架（离线索引 + 在线检索）

一个**零硬依赖**、面向生产权衡的 RAG 框架，实现位置：`Hello-Agent/cn/mcheat/rag/`。

设计目标来自 [`doc/rag编码注意事项.md`](../../../doc/rag编码注意事项.md) 的 5 条权衡点；
框架把每条都做成**可验证、可量化、可替换**的具体组件，而不是写在注释里的口号。

- 核心链路只使用 Python 3.10+ **标准库**，不需要 API key、不需要 numpy；
- 所有关键组件都是 `abc.ABC` 接口 + 具名注册表，改配置或注入对象即可替换；
- 离线索引与在线检索可分成两个进程，通过 `IndexArtifact` 传递状态；
- 自带 Golden Set 回归评测、拒答指标与 CI 门禁、完整异常体系；
- `configs/defaults.json` 保持 `hashing` / `lexical` 零依赖默认档，新增
  `configs/production.json` 中文生产补丁档。

---

## 1. 快速开始

从**项目根目录**运行（保证 `cn.mcheat.rag` 可导入）：

```bash
python -m cn.mcheat.rag.examples.demo                  # 一站式演示，直接可跑
python -m cn.mcheat.rag.examples.offline_build         # 离线建索引 -> examples/.rag_index/
python -m cn.mcheat.rag.examples.online_query "父子切分解决什么问题？" --show-prompt
python -m cn.mcheat.rag.examples.run_evaluation --sweep-threshold  # 阈值标定
python -m cn.mcheat.rag.examples.run_evaluation --save-baseline    # 生成基线
python -m cn.mcheat.rag.examples.run_evaluation --gate              # 回归门禁
python -m cn.mcheat.rag.examples.production_demo       # 缺可选依赖时优雅跳过
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

### 2.1 第 1 条：两段式 Top-K 与分数域约定

```
query ──> dense(Embedder + VectorStore) ── Top-K ─┐
      └─> sparse(BM25)                 ── Top-K ─┴─> RRF 融合去重 ─> coarse (≤50)
                                                                      │
                                                      Reranker.rerank(top_n=5)
                                                                      │
                                                            final (≤5) -> prompt
```

`coarse_top_k` / `rerank_top_n` 是两个独立配置，且有 `rerank_top_n <= coarse_top_k`
的硬校验。粗召回结果保留在 `RetrievalResult.coarse` 中，**只用于诊断和评测**，
永远不会被拼进 prompt。

**分数域约定（有意为之，不可混用）**：

| 位置 | 值域 | 用途 |
|---|---|---|
| `final[].score` | `[0,1]` 归一化分 | `min_best_score` 质量门禁、答案级评测 |
| `final[].raw_score` | 精排器原始域 | 诊断 / 消融对比 |
| `coarse[].score` | RRF 融合域（≈0.016） | 粗召回诊断、Context Recall |

`Reranker.normalize_scores()` 是接口级契约：默认 clamp 到 `[0,1]`，
`CrossEncoderReranker` 用 sigmoid，`NoopReranker` 用 `1 - rank/n` 名次映射，
`LexicalOverlapReranker` 移除了 prior 地板（仅作同分 tie-break）。

> 默认 Reranker 是零依赖的 `LexicalOverlapReranker`（词面覆盖率 + 词项聚集度）。
> **它是占位实现，不是语义模型**；生产环境请使用第 6 节的中文生产档。

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
`SequentialContextBuilder`（`placement="sequential"`）是消融基线。预算裁剪
已改为**预计算块长度 + 增量扣减**，不再在循环中反复完整渲染所有块。

### 2.4 第 4 条：分阶段度量与拒答指标

| 指标 | 公式 | 度量阶段 |
|---|---|---|
| Context Recall | `\|检索到 ∩ 相关\| / \|相关\|` | `result.coarse`（粗召回） |
| Context Precision | `\|检索到 ∩ 相关\| / \|检索到\|` | `result.final`（真正进 prompt） |
| Average Precision / MRR / HitRate / F1 | 见 `evaluation/metrics.py` | 精排后的有序列表 |
| `false_refusal_rate` / `false_accept_rate` / `refusal_accuracy` | 见 `evaluation/metrics.py` | 全部样本（含负样本） |

负样本（`answerable=false`、`relevant_ids=[]`）**不参与 Recall / Precision 均值**；
拒答指标覆盖全部样本并纳入 `GATED_METRICS`（旧基线缺这些指标时自动跳过比较）。
`EvalReport.regression_check()` 对「越低越好」与「越高越好」的指标分别判定。

### 2.5 第 5 条：可观测的降级梯度

```python
result.fallback.triggers        # 为什么触发：召回不足 / 归一化精排分低 / 上下文为空 ...
result.fallback.steps           # 每一步动作：widen_top_k -> human_handoff -> honest_no_answer
result.diagnostics["retry_history"]  # [{from_top_k, to_top_k, reason}, ...]
result.diagnostics["stage_timings_ms"]  # transform/recall/fusion/rerank/expand/context/fallback
```

单路召回**抛异常**时不会中断链路：记录 `degraded` 后继续用另一路；
两路都异常时抛内部信号 `EmptyRecallError`，由兜底链路接管。

---

## 3. 目录结构

```
cn/mcheat/rag/
├── README.md                    本文档
├── requirements.txt             可选依赖清单（核心零依赖）
├── __init__.py                  公共 API + 启动时注册内置组件
├── api.py                       RAGSystem 门面：index/retrieve/evaluate/save/load
├── config.py                    配置管理（默认值→配置档→文件→环境变量→覆盖 + 校验）
├── models.py                    领域模型（Document / ScoredChunk / IndexStats / ...）
├── exceptions.py                分阶段异常体系
├── _utils.py                    稳定哈希、中英文分词、向量运算、日志
├── configs/
│   ├── defaults.json            零依赖默认档（唯一的参数入口）
│   └── production.json          中文 BGE 生产补丁档（NEW）
├── core/
│   ├── interfaces.py            全部 ABC / Protocol 扩展点
│   ├── registry.py              具名注册表 + 可选组件机制 + "@组件名" 依赖注入
│   └── bootstrap.py             内置实现自注册（幂等）
├── loaders/                     5 种 Document Loader + 目录组合器 + 后缀分发
├── cleaning/                    7 步可组合清洗流水线 + 清洗报告
├── indexing/
│   ├── splitters.py / parent_store.py
│   ├── manifest.py              SourceManifest / SourceRecord（NEW）
│   └── pipeline.py              IndexArtifact + OfflineIndexer（增量 / 单次 flush）
├── embeddings/                  HashingEmbedder（默认）+ LangChain BGE 适配器
├── vectorstores/                InMemory / FileBacked / SqliteVectorStore（NEW）
├── retrieval/
│   ├── recall.py / bm25.py / hybrid.py / rerank.py / context.py / fallback.py
│   ├── query_transform.py       Identity / 规则型 / LLM Protocol / 变换链（NEW）
│   ├── cache.py                 查询向量 LRU + 结果 TTL 缓存（NEW）
│   └── pipeline.py              OnlineRetriever（变换→缓存→召回→精排→兜底）
├── evaluation/
│   ├── golden.py / metrics.py / runner.py
│   └── answer_metrics.py        AnswerJudge / LexicalJudge / AnswerEvalRunner（NEW）
├── examples/                    示例语料 + 可执行脚本 + golden_set.json（含负样本）
└── tests/                       unittest 测试（test_01~test_17）
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
**`VectorStore`（可插拔向量库）**、`RecallRetriever`、`FusionStrategy`、
`Reranker`、`ParentStore`、`ContextBuilder`、`FallbackStrategy`、
`QueryTransformer`、`QueryRewriter`（Protocol）、`AnswerJudge`（Protocol）、
`CrossEncoderScorer`（Protocol）、`Stateful`（持久化 mixin）。

`Reranker.normalize_scores()` / `Embedder.fingerprint()` /
`VectorStore.get_many()` / `delete_by_source()` 都带默认实现，自定义组件
平滑升级；`Stateful` 让组件随 `RAGSystem.save()` 一起导出。

### 5.2 三种替换方式

**① 改配置名**（最常用，`configs/defaults.json` 或用户 JSON）：

```json
{
  "components": {
    "vector_store": {"name": "sqlite", "params": {"path": "./.rag_index/vectors.sqlite3"}},
    "parent_store": {"name": "sqlite", "params": {"path": "./.rag_index/parents.sqlite3"}},
    "fusion":       {"name": "weighted", "params": {}}
  }
}
```

> 切换 `name` 时框架会**自动丢弃旧实现继承来的 params**，只保留为新实现显式
> 写的参数，避免 `TypeError`；因此 `production.json` 为每个换实现的组件显式
> 写全 params（模型名 / 前缀 / batch_size / 路径）。

**② 注册自定义实现**：

```python
from cn.mcheat.rag.core.registry import VECTOR_STORE_REGISTRY
from cn.mcheat.rag.vectorstores import InMemoryVectorStore

@VECTOR_STORE_REGISTRY.register("my_store")
class MyVectorStore(InMemoryVectorStore):
    def search(self, query_vector, k, metadata_filter=None):
        ...  # 换成 FAISS / Milvus / pgvector / 自研 ANN
```

**③ 直接注入实例**：`RAGSystem(overrides={"vector_store": MyVectorStore()})`。

配置支持 `"@组件名"` 引用做依赖注入；构建顺序由 `core.registry.BUILD_ORDER` 保证。
`query_transformer` / `cache` 是**可选组件**：`BUILD_ORDER` 前置插入，但
`REQUIRED_COMPONENTS` 仍是原来的 9 个，`RAGConfig.from_dict()` 会为旧配置自动
注入默认规格（`identity` / `null`），旧用户配置、旧基线、旧索引产物不受影响。

---

## 6. 生产化：中文生产配置档（已重写）

默认实现是为了「零依赖、可复现、开箱即跑」，**不是语义最优**。原第 6 节推荐的
英文域 `all-MiniLM` / `ms-marco` 模型与中文语料错配，现改为中文优先的生产档：

| 组件 | 默认值 | 生产档（`production.json`） |
|---|---|---|
| `embedder` | `hashing` | `langchain` + `BAAI/bge-m3`（中英混合）+ BGE query/passage 前缀预设 |
| `reranker` | `lexical` | `cross_encoder` + `BAAI/bge-reranker-large`（中文）+ sigmoid 归一化 |
| `vector_store` | `memory` | `sqlite`（二进制 blob，避免 JSON 写放大） |
| `parent_store` | `memory` | `sqlite` |
| `parallel_recall` | `false` | `true`（仅 I/O-bound 的 API embedding 场景有收益） |

加载方式（二选一）：

```bash
RAG_PROFILE=production python -m cn.mcheat.rag.examples.production_demo
```

```python
from cn.mcheat.rag import RAGSystem, load_profile

config = load_profile("production")   # 等价于 RAGConfig.load(path="configs/production.json")
system = RAGSystem(config=config)
system.index(["your/corpus"], mode="full")
system.save("./.rag_index")
```

> **切换模型必须重建索引 + 重跑基线**：`Embedder.fingerprint()` =
> 类名 + 模型名 + 维度 + 前缀配置的稳定哈希；`IndexArtifact` 记录指纹与
> `index_version`，`RAGSystem.load()` 不一致即抛
> `IndexFingerprintMismatchError`（消息含「重建索引」），同维度换模型也能拦截。
> 旧产物缺这些字段**仅 WARN 不报错**。
>
> **阈值为暂行值**：`production.json` 的 `min_best_score` 沿用默认栈
> （`lexical + hashing`）的 `--sweep-threshold` 标定结果，与生产栈分数分布
> 不同、**不可迁移**；加载时会输出 WARNING，门禁在此之前不宣称已生效。
> 上线前须按方案 9.4 一次性人工流程用真实 BGE 模型单独扫描复标，回填
> `production.json` 并更新 `--save-baseline` 基线。
>
> **云端 provider 注记**：若使用 DashScope / OpenAI 等云端 embedding，
> 重试 / 限流 / 超时尚未实现（后续路线 #9），调用方需自行在 Embeddings
> 外层包一层重试，避免离线构建因网络抖动整体失败。
>
> 缺可选依赖时 `DependencyMissingError` 会给出 `pip install ...` 提示；
> `production_demo.py` 会优雅跳过，默认档不需要任何第三方依赖。

---

## 7. 配置管理

优先级：`configs/defaults.json` < `RAG_PROFILE` 配置档 < 用户 JSON <
`RAG_*` 环境变量 < `overrides` 字典。

```python
from cn.mcheat.rag import RAGConfig

cfg = RAGConfig.load(
    path="my_rag.json",
    overrides={"retrieval": {"coarse_top_k": 30, "rerank_top_n": 3}},
)
cfg.validate()  # 硬约束 + 建议区间软告警 + 门禁关闭/暂行阈值 WARNING
```

新增配置节（8.2 示意）：

```jsonc
{
  "retrieval": {
    "min_best_score": 0.30,       // 暂行值：0.35 起扫后回填，扩至 30~50 条复标
    "parallel_recall": false,     // D10：默认关闭，仅生产档/服务化显式开启
    "recall_timeout_s": 5.0,
    "max_query_variants": 3,
    "max_recall_calls": 6
  },
  "query_transform": { "name": "identity", "params": {}, "on_error": "use_original" },
  "cache": { "enabled": false, "query_lru_size": 512, "result_ttl_s": 300 },
  "indexing": { "mode": "auto" }
}
```

`validate()` 会拒绝：`child_size > parent_size`、`rerank_top_n > coarse_top_k`、
两路召回全关、非法位置策略 / 索引模式 / 降级策略等；门禁关闭
（`min_best_score=0.0`）与生产档暂行阈值未复标会输出 WARNING。

高频环境变量：`RAG_COARSE_TOP_K`、`RAG_RERANK_TOP_N`、`RAG_MIN_BEST_SCORE`、
`RAG_FALLBACK_ENABLED`、`RAG_RERANKER`、`RAG_EMBEDDER`、`RAG_VECTOR_STORE`、
`RAG_PROFILE`、`RAG_LOG_LEVEL`。完整映射见 `config.ENV_OVERRIDES`。

> JSON 不支持注释，允许用 `_` 前缀的键写说明（会被忽略）。

---

## 8. 错误处理与日志规范

`RAGError` 为根，按阶段分层；新增 `QueryTransformError` / `CacheError` /
`IndexFingerprintMismatchError`。所有报错都带**上下文 + 修复建议**。

- 统一 `get_logger(__name__)`；缓存命中、逐块裁剪用 `DEBUG` 避免刷屏；
- 门禁关闭、生产档暂行阈值未复标、指纹不匹配、查询变换降级用 `WARNING`，
  消息含修复建议；
- **查询文本、API key、向量内容不写入日志**；
- `cleaning.on_error = "raise" | "skip"`；Loader 编码回退；单路召回异常降级；
- 库代码不配置日志 handler（只挂 `NullHandler`），由应用侧决定输出。

---

## 9. 阈值标定与评测验收

### 9.1 阈值标定流程（P0-1 前置）

1. `examples/golden_set.json` 现有 **8 条正样本 + 5 条应拒答负样本**
   （`answerable=false`、空 `relevant_ids`）；旧正样本 JSON 无需改动即合法。
2. 以 **0.35 为待验证起点**跑：

   ```bash
   python -m cn.mcheat.rag.examples.run_evaluation --sweep-threshold
   ```

   输出 0.0~0.8 区间每个阈值下的误杀率（正样本被拒答）、误接受率
   （负样本被生成 prompt）、诚实拒绝率与拒答准确率。
3. 硬约束「既有 8 条正样本零误杀」，选误接受率最低的阈值写入
   `defaults.json`；框架已依据扫描结果写入**非零暂行阈值**。
4. `false_refusal_rate` / `false_accept_rate` / `refusal_accuracy` 已纳入
   `GATED_METRICS`，用 `--save-baseline` / `--gate` 锁定。
5. **复标计划**：8 正 + 5 负样本量下阈值置信区间很宽，所有阈值均标注
   **「暂行值，Golden Set 扩至 30~50 条后复标」**；复标后更新基线并重跑门禁。
6. `NoopReranker` 用 `1 - rank/n` 名次映射保证消融不被阈值团灭；如需完全
   排除阈值干扰，可临时 `min_best_score=0.0` 旁路（框架输出 WARNING），
   实验结束即恢复。

> 该阈值**仅适用于默认栈**；生产栈（`bge-reranker` + sigmoid）须按方案 9.4
> 一次性人工流程单独标定，不得复用。

### 9.2 答案级评测（框架边界仍止于 `result.prompt`）

`evaluation/answer_metrics.py` 定义 `AnswerJudge` Protocol 与零依赖的
`LexicalJudge` 兜底；`AnswerEvalRunner` **接收调用方生成的答案**，产出
Faithfulness / Answer Relevancy 聚合报告。框架不接 `llm.py`、不绑定厂商、
不需要 API key，CI 用假裁判即可验证。

```python
from cn.mcheat.rag import AnswerEvalRunner, LexicalJudge

report = AnswerEvalRunner(system, golden_set, answers, judge=LexicalJudge()).run()
print(report.summary)  # {"faithfulness": ..., "answer_relevancy": ...}
```

CLI 入口：`run_evaluation --answer-eval --answers answers.json`
（`answers.json` 为 `query -> 答案` 的 JSON 对象）。

### 9.3 测试与验收命令

```bash
python -m unittest discover -s cn/mcheat/rag/tests -t . -p "test_*.py"
python -m cn.mcheat.rag.examples.demo
python -m cn.mcheat.rag.examples.run_evaluation --save-baseline
python -m cn.mcheat.rag.examples.run_evaluation --gate
python -m cn.mcheat.rag.examples.production_demo   # 缺可选依赖时应优雅跳过
```

`tests/test_10_end_to_end.py` 含**可执行的等价性测试**：以阶段 3 结束时
`--save-baseline` 冻结的 `summary` 为准，在「全部可选组件关闭」配置下重跑
Golden Set 逐项比对相等，不依赖人工判断。

---

## 10. 缓存、增量索引与并行化

| 能力 | 开关 | 说明 |
|---|---|---|
| 结果 / 查询缓存 | `cache.enabled=false`（默认）或 `components.cache.name="null"` | `EmbeddingQueryCache` 是 LRU，`RetrievalResultCache` 是 TTL；两级 key 均含 `index_version`，索引更新后自然失效。同一 query 连续 100 次检索 embedding 调用 100 → 1 |
| 查询变换 | `components.query_transformer.name="identity"`（默认） | `identity` / `rule_based` / `llm`（仅 Protocol 注入）；多查询分别粗召回后复用 `FusionStrategy`，精排与 prompt 用主查询；失败降级为原查询 |
| 增量索引 | `RAGSystem.index(..., mode="auto")`（默认） | 无 `SourceManifest` 时全量、有则按 `content_hash` 只重建变更来源；`mode="full"` 强制全量；`prune_stale` 仅在全量模式生效 |
| 并行召回 | `retrieval.parallel_recall=false`（默认） | 默认栈两路均为纯 Python CPU-bound，受 GIL 限制线程并行无收益；生产档 / API embedding 的 I/O-bound 场景显式开启，单路超时自动降级 |
| 调用预算 | `max_query_variants=3` / `max_recall_calls=6` | 约束「多查询 × widen_top_k 重试」的成本放大 |

> **`auto_save` 语义变更（D7，请特别注意）**：`FileBackedVectorStore.auto_save`
> 默认值由 `True` 改为 **`False`**。构建期由 `OfflineIndexer.build()` 结束统一
> `flush()` 一次（10 万子块 / batch 32：改造前 3000+ 次全量写 → 改造后 1 次）；
> **直接使用该类的调用方必须显式调用 `flush()`**，否则写操作不会自动落盘。
> 需要旧语义可显式传 `auto_save=True`。`tests/test_05` 已同步适配。

`IndexArtifact` 新增 `embedding_fingerprint` / `index_version` / `manifest`；
`IndexStats` 新增 `index_version` / `mode` / `changed_sources` / `deleted_chunks`；
`diagnostics` 新增 `stage_timings_ms` / `query_variants` / `cache_hit` /
`index_version` / `recall_calls`，原 `elapsed_ms` 保留。内存组件与 SQLite
连接均已加 `RLock` / `check_same_thread=False` 保证并发安全。

---

## 11. 已知限制与后续路线

1. 示例语料很小，粗召回会覆盖全部子块，因此 demo 里 `context_recall` 恒为 1；
   真实语料（几百条以上）才有区分度。
2. 默认 `hashing` embedder 与 `lexical` reranker 不是语义模型；上线请使用
   第 6 节中文生产档并**重建索引 + 重跑 Golden Set 基线**。
3. **`min_best_score` 是暂行值**（8 正 + 5 负样本），置信区间宽；生产档阈值
   更是沿用默认栈结果、未经生产栈复标，上线前须按方案 9.4 完成一次性人工标定。
4. `HashingEmbedder` / `BM25Index` / `SqliteVectorStore` 仍是暴力检索 +
   BM25 线性扫描，适合中小语料；大规模请接 ANN / 真倒排（P2）。
5. 框架只负责检索与上下文组装，**不调用 LLM**；答案级评测接收调用方答案，
   `result.prompt` 交给哪个模型、如何解析引用由应用侧决定。
6. P2 与后续路线（本轮不做）：`ChildChunk.embed_text` 标题增强、BM25 真倒排
   + 停用词、token 预算、流式摄取、embedding API 重试/限流、OTel 聚合指标、
   引用后校验、多租户过滤安全加固、ANN 适配器。
7. `parallel_recall` 默认关闭；默认栈两路 CPU-bound，线程并行无收益甚至更慢，
   仅 I/O-bound（API embedding）时用 p95 延迟对比验收。

---

## 12. 问题归因（本次改造的四类根因）

| 类别 | 问题 | 修复 |
|---|---|---|
| A 默认配置路径缺陷 | `min_best_score=0.0` 门禁实际关闭；英文域模型错配；`auto_save=True` 写放大 | 统一归一化契约 + 非零暂行阈值；中文生产档 + 前缀 + 指纹；`auto_save=False` + 单次 `flush()` + SQLite 向量库 |
| B 跨组件契约缺陷 | 三个精排器分数域不同；`embed_query/embed_documents` 无前缀；同维度换模型静默混用；无负样本无法标定；`coarse` 分数域未说明 | `normalize_scores()` 接口契约 + `raw_score`；BGE/E5 预设；指纹 + `index_version`；负样本与拒答指标；`coarse` 保持 RRF 域并文档标注 |
| C 能力缺失 | 查询变换、两级缓存、增量索引、答案级评测、并行召回、阶段耗时、线程安全 | 全部以**可选组件 / 开关**实现，默认行为不变 |
| D 规模化性能 | 两路串行；预算裁剪重复渲染；`widen_top_k` 无预算；BM25 线性扫描 | 可选并行（默认关）、预计算长度、`max_recall_calls` 预算；BM25 倒排留待 P2 |
