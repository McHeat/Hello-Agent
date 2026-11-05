# RAG 实现全面评估报告

> 评估对象：`cn/mcheat/rag/`（可插拔 RAG 框架，零硬依赖，离线索引 + 在线检索）
> 评估日期：2026-11-05
> 评估范围：全部 58 个源文件（config / loaders / cleaning / indexing / embeddings /
> vectorstores / retrieval / evaluation / api），共 122 个单元测试。

---

## 一、架构合理性评估

### 1.1 总体判断

这是一个**设计成熟度明显高于平均水平的教学/工程两用 RAG 框架**。接口抽象
（`core/interfaces.py` 的 12 个 ABC + Protocol）、注册表 + `@组件名` 依赖注入、
离线/在线进程分离（`IndexArtifact`）、分阶段评测（Recall 在粗召回算、Precision 在
精排后算）+ CI 回归门禁，这些在生产级系统中都属于最佳实践。

**核心结论：架构骨架优秀，但默认实现是"占位"性质，且存在若干会在规模化场景下
暴露的真实缺陷（见 1.2 各环节评估）。**

### 1.2 各环节逐项评估

#### 数据摄取与清洗 —— 合理

5 种 Loader + `DirectoryLoader` 组合器 + 7 步清洗流水线的设计没有明显问题。
"先清洗再切分"保证偏移不变量的做法正确。两点小瑕疵：

- `OfflineIndexer.build()` 将全部文档一次性载入内存，无流式处理，
  大语料（10 万+ 文档）会成为内存瓶颈；
- 清洗去重是精确内容哈希，无近似去重（MinHash/SimHash），
  对"转载/微改"类冗余无效。

#### 分块策略 —— 合理且有亮点，但有一个隐性代价

Small-to-Big（父 1200 / 子 240 / 重叠 48）+ 子串不变量 + 按父块去重，
是教科书级实现。但存在一个**设计层面的隐性缺陷**：

子串不变量要求 `child.text == parent.text[start:end]`，这意味着子块的
embedding 文本**无法携带任何上下文增强**（如所属章节标题、文档标题）。
一篇长文中间某段"其缺点是……"的子块，脱离标题后 embedding 质量会显著下降。
当前架构下想给子块加标题前缀就必须破坏不变量——这是一个需要显式权衡的
设计死结（破解方案见 P2-9）。

另外，Markdown/HTML 的**结构信息在切分时被完全丢弃**（`TextLoader` 提取的
标题只进 metadata，不参与切分边界决策），切分退化为纯字符级启发式。

#### 嵌入模型 —— 最大短板（默认配置下）

默认 `HashingEmbedder`（256 维特征哈希）只捕捉词面重叠，**完全没有语义泛化
能力**，稠密通路实际上退化成了"又一个 BM25"。README 已诚实声明这是占位实现，
但有两个更深层的问题：

1. **LangChain 适配器的默认模型选错了语言域**。
   `all-MiniLM-L6-v2` 是英文模型，而这个框架的分词器、清洗规则、示例语料
   全是中文优先的。用户按 README 第 6 节"一键切换"后，中文检索质量会不升反降。

2. **query/document 非对称前缀未落地**。`Embedder` 接口专门区分了
   `embed_query` / `embed_documents`，但 `LangChainEmbedder` 直接透传，
   不支持 BGE/E5 系列要求的 `"query: "` / `"passage: "` 前缀——
   换上这些模型时精度会打折。

#### 向量存储与索引 —— 正确但有严重性能缺陷

- `InMemoryVectorStore` 纯 Python O(N·D) 暴力扫描，README 已声明限制，可接受；
- **`FileBackedVectorStore` 存在一个真实的 O(N²) 缺陷**：`auto_save=True`
  （默认）时**每个 batch（32 条）都全量序列化整个库并重写 JSON 文件**。
  索引 10 万子块 = 3000+ 次全量重写，越写越慢，磁盘 IO 彻底失控。
  这个默认值对"离线建索引"的主场景是反效果的；
- 无 ANN（HNSW/IVF），检索延迟随库规模线性增长；
- 持久化格式为 JSON 文本（向量以十进制字符串存储），体积比二进制格式
  膨胀 3~5 倍，加载解析也慢。

#### 检索策略 —— 骨架完整，细节有洞

混合召回 → RRF 融合 → 精排截断的两段式 Top-K 设计正确，
`rerank_top_n <= coarse_top_k` 硬校验、单路失败自动降级都很好。问题在：

1. **`min_best_score` 质量门禁在默认配置下形同虚设**。
   默认 `lexical` 精排器分数域约 [0, 1] 且恒 ≥ 0，而 `min_best_score`
   默认 `0.0`，`max(scores) < 0.0` 永远为假。也就是说：只要召回条数 ≥ 3，
   **任何垃圾结果都会被判定 `answer_ready=True` 并生成 prompt 交给 LLM**。
   幻觉控制实际上只剩"数量阈值"一道防线，没有"质量阈值"。

2. **切换精排器后阈值语义不一致（跨组件缺陷）**。
   `CrossEncoderReranker` 直接使用 `ms-marco-MiniLM-L-6-v2` 的原始 logit
   （域约 [-10, 10]，未归一化），而 `config.validate()` 强制
   `min_best_score ∈ [0, 1]`。切到 cross_encoder 后，分数 > 1 的结果永远过阈、
   负分结果可能误杀——**阈值机制在换实现后直接失效**，且这是框架自己强调的
   "可插拔"路径上会踩的坑。默认推荐模型同样又是英文域的 `ms-marco`，
   与中文场景不匹配。

3. **兜底重试成本放大**。`widen_top_k` 触发后回到循环头部**重跑完整的
   两路召回 + 精排**，最多 2 次。对 API 型 embedding / cross-encoder，
   这意味着单次查询最坏 3 倍调用成本，且没有任何熔断/预算控制。

4. **BM25 名不副实**：`BM25Index` 虽然注释写"倒排索引"，实际查询时
   **线性遍历所有 chunk 的 tf 字典**逐个打分，与暴力扫描无异。

5. 两路召回是**串行执行**的（`for source, retriever, k in paths`），无并行。

#### 上下文构建与提示组装 —— 设计有据，实现基本正确

U 形编排（edges）、预算裁剪优先砍中间、末尾重复问题，均有文献支撑，
且 `SequentialContextBuilder` 提供了消融对照，这是亮点。三个次要问题：

- 预算单位是**字符**不是 token，与 LLM 实际计费/限制单位脱节，
  中英混排时偏差可达 2~3 倍；
- `_total_length` 在裁剪 while 循环里被反复调用，且每次调用都对所有块
  做完整 `_block_text` 渲染，O(n²)（块数少影响小，属代码洁癖级问题）；
- 引用标注只靠 system instruction 要求 LLM"用 [编号] 标注"，
  **没有任何生成侧的引用校验机制**——框架到 `result.prompt` 为止是诚实的设计
  选择，但意味着幻觉控制的最后一环完全外包给了调用方。

---

## 二、缺失点清单

按重要性排序：

| # | 缺失能力 | 影响 | 现状证据 |
|---|---------|------|---------|
| 1 | **语义嵌入（默认配置）** | 稠密通路无语义能力，混合检索收益减半 | `defaults.json` → `hashing`, 256 维 |
| 2 | **质量阈值的跨精排器归一化** | 换精排器后 `min_best_score` 语义失效；默认 0.0 等于关闭质量门禁 | `rerank.py` raw logits 直通 |
| 3 | **查询变换层（Query Rewriting / 多查询扩展 / HyDE / 分解）** | 口语化、多跳、关键词稀疏的查询召回率低；这是现代 RAG 与朴素 RAG 的分水岭 | `OnlineRetriever.retrieve` 直接拿原始 query 用 |
| 4 | **缓存（查询缓存 / embedding 缓存 / 结果缓存）** | 高频重复查询全链路重算；API embedding 重复计费 | 全链路无任何 cache |
| 5 | **增量索引 / 文档级更新** | 改一个文件必须全量重建（README 已承认）；语料频繁变更场景不可用 | `prune_stale` 全量对账 |
| 6 | **生成侧评测（Faithfulness / Answer Relevancy / LLM-as-judge）** | 现有 7 项指标全部度量检索，无一度量最终答案质量；"幻觉控制"效果无法量化 | `evaluation/metrics.py` 全是检索指标 |
| 7 | **并发与线程安全** | 在线进程多线程调用 `retrieve` 时，内存 dict / BM25 无锁；服务化部署隐患 | 全部内置组件无同步原语 |
| 8 | **结构感知切分 / 标题上下文注入** | 长文档中部子块脱离标题后 embedding 语义漂移 | 子串不变量阻塞了标题增强 |
| 9 | **embedding API 的重试/限流/超时** | 网络抖动直接炸掉整个离线构建（`on_error` 只覆盖清洗阶段） | `adapters.py` 无任何重试 |
| 10 | **可观测性指标（延迟分位数、阶段耗时直方图）** | `diagnostics` 字段优秀但仅单次查询维度，无聚合/上报通道 | 无 OpenTelemetry / Prometheus 集成 |
| 11 | **引用后校验 / 答案忠实性守门** | prompt 组装后的幻觉风险完全依赖调用方 LLM 自觉 | 框架边界止于 prompt |
| 12 | **多租户过滤的安全性** | `metadata_filter` 由调用方传入后直接透传，无注入校验（对 SQLite 类后端有风险） | `VectorStore.search` 透传 |

---

## 三、改进建议（按优先级）

### P0 —— 上生产前必须解决

#### 1. 修正精排分数归一化 + 启用质量阈值（防御幻觉，改动小收益大）

在 `CrossEncoderReranker.rerank` 中对 logit 做 sigmoid 归一化，使其落入 [0, 1]
与 `min_best_score` 的校验域一致：

```python
import math
batch_scores = [1.0 / (1.0 + math.exp(-float(v))) for v in raw]
```

同时在 `defaults.json` 中把 `min_best_score` 从 `0.0` 调到一个经 Golden Set
标定的非零值（建议先用 0.3~0.5 做扫描实验，用 `evaluation` 门禁确认
"诚实拒绝率 vs 误杀率"的平衡点）。当前 0.0 默认值意味着质量门禁整个是关着的
——这是全框架最容易被忽视的幻觉入口。

#### 2. 替换默认语义嵌入为中文优先模型

`adapters.py` 与 README 的推荐从 `all-MiniLM-L6-v2` 改为
`BAAI/bge-large-zh-v1.5` 或 `BAAI/bge-m3`（中英混合语料选 m3）。
同时在 `LangChainEmbedder` 中实现 query/passage 前缀注入（BGE 需要
`"为这个句子生成表示以用于检索相关文章："` 查询前缀，E5 需要
`query:/passage:`），这是 `Embedder` 接口区分两个方法的本意，目前没有兑现。
替换后必须重建索引并重跑 Golden Set 基线（框架已有 `DimensionMismatchError`
防护，这点做得好）。

#### 3. 修复 `FileBackedVectorStore` 的 O(N²) 写放大

最小改动：`OfflineIndexer` 构建期间以 `auto_save=False` 运行，
构建结束统一 `flush()` 一次。更好的方案：向量落盘改用 SQLite
（框架已有 `SqliteParentStore` 先例，模式可复用），JSON 只留元数据。
若目标语料 > 10 万块，直接接 FAISS/HNSW 适配器——接口已就绪，
无需改上层代码。

### P1 —— 显著提升质量与效率

#### 4. 引入查询变换层（新组件）

在 `core/interfaces.py` 新增 `QueryTransformer` ABC，注册进 `BUILD_ORDER`
最前端，链式支持：

- 查询改写（LLM 把口语转检索式，如"那玩意儿为啥不好使" →
  "Top-K 参数过大的问题"）；
- 多查询扩展（1 query → 3 个变体，各自召回后 RRF 融合，
  天然复用现有 `FusionStrategy`）；
- HyDE（可选，用 LLM 生成假设答案再检索，对稠密通路收益明显）。

注意：查询改写引入 LLM 依赖后，需在 `FallbackConfig` 中增加
"改写失败则用原查询"的降级，保持框架"零依赖可跑"的底线。

#### 5. 增加两级缓存

- **embedding 查询缓存**：`EmbeddingDenseRetriever.search` 前 LRU 缓存
  `query → vector`（对 API embedding 直接省钱）；
- **结果缓存**：`(query, metadata_filter) → RetrievalResult` 的 TTL 缓存，
  命中时跳过全链路。缓存 key 必须包含索引版本号（可在 `IndexArtifact.created_at`
  基础上加单调递增 `index_version`），否则增量更新后会出现脏缓存。

#### 6. 文档级增量更新

给 `Document` 增加 `source` 级别的变更检测（`content_hash` 已有），
`OfflineIndexer.build` 支持只 reindex 变更文档：删除该 source 下所有旧 chunk
（需要 `delete_by_source`，对 BM25/向量库都可基于 metadata 实现），
再写入新块。`prune_stale` 保留作为全量模式。

#### 7. 补齐答案级评测

在 `evaluation/` 增加 `answer_metrics.py`：注入一个 LLM judge
（复用 `CrossEncoderScorer` 式的 Protocol 注入模式，不强依赖），度量
Faithfulness（答案是否每个论断都有上下文支撑）与 Answer Relevancy。
Golden Set 的 schema 扩展 `expected_answer` 字段（可选）。这样
"兜底/幻觉控制是否有效"第一次变得可量化——目前 `answer_ready_rate`
只能度量"框架是否按约定拒绝了"，不能度量"没拒绝时答案是否忠实"。

#### 8. 在线服务化基础

- 两路召回并行化（`concurrent.futures`，接口无共享可变状态，天然可并行）；
- 为 `InMemoryVectorStore` / `BM25Index` 加读写锁，或在文档中明确
  "单进程单线程"的契约；
- `diagnostics` 已经很全，建议增加阶段耗时分解
  （`recall_ms` / `rerank_ms` / `context_ms`），目前只有一个总 `elapsed_ms`。

### P2 —— 锦上添花

#### 9. 破解"子串不变量 vs 标题增强"的死结

给 `ChildChunk` 增加 `embed_text` 字段（= 标题前缀 + 原文），
`EmbeddingDenseRetriever.index` 用 `embed_text` 向量化、
`SmallToBigExpander` 还原时仍用原 `text`。不变量从
"子块文本是父块子串"放宽为"子块文本是父块子串，且 embed_text = heading + text"。
对长结构化文档（手册、法规）预计有可观 Recall 提升，用 Golden Set 验证。

#### 10. BM25 真倒排

建 `term → set[chunk_id]` 倒排表，查询时只访问命中 query term 的 chunk，
从 O(N) 降到 O(命中集)；顺带在 `tokenize` 中加轻量停用词过滤
（中英文各 50 词即可），当前"的/了/is/the"都在参与打分。

#### 11. token 预算

`max_context_chars` 增加 `max_context_tokens` 选项（实现一个可注入的
`TokenCounter`，不同 LLM 族可替换），避免按字符估预算在英文内容上爆 token。

#### 12. 流式摄取

`load_many` 改为生成器逐文件产出，清洗/切分按批消费，
缓解大语料内存峰值。

---

## 四、优先级总结

| 优先级 | 事项 | 一句话理由 |
|-------|------|-----------|
| **P0** | sigmoid 归一化 + 非零 `min_best_score` | 当前质量门禁实际是关闭的，幻觉风险 |
| **P0** | 中文语义嵌入 + query 前缀 | 默认/推荐模型均为英文域，主场景失配 |
| **P0** | 修复 FileBackedVectorStore 写放大 | O(N²) IO，建索引越建越慢 |
| **P1** | 查询变换层 | 朴素 RAG → 现代 RAG 的最大增量 |
| **P1** | 缓存 + 增量索引 | 降本与语料保鲜 |
| **P1** | 答案级评测 | 让"幻觉控制"从口号变成指标 |
| **P2** | 标题增强 / BM25 倒排 / token 预算 / 流式 | 规模化后的长尾优化 |

---

## 五、结语

这套框架的**架构决策（两段式 Top-K、Small-to-Big、分阶段度量、可观测兜底）
全部正确且实现严谨**，主要风险集中在两处：

1. "默认占位实现的语言域错配"（英文模型 vs 中文语料）；
2. "质量阈值机制形同虚设"（`min_best_score=0.0` + 跨精排器分数域不一致）。

它们恰好都藏在**使用者最信任的默认配置路径**上，建议按 P0 → P1 → P2 顺序
逐项修复，每项修复后用 Golden Set 回归门禁验证收益（框架已内置该能力）。
