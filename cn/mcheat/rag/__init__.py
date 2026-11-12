"""可插拔 RAG 框架：离线索引 + 在线检索。

严格落实 ``doc/rag编码注意事项.md`` 的 5 条权衡点：

====== ======================================================== ==================================
条目   落实位置                                                  具体做法
====== ======================================================== ==================================
1      ``retrieval/hybrid.py`` / ``rerank.py`` / ``pipeline.py``  向量+BM25 混合召回 30~50 → RRF → 精排 3~5 才进 prompt
2      ``indexing/splitters.py`` / ``retrieval/context.py``       父子切分 Small-to-Big：小片段检索、返回大段落
3      ``retrieval/context.py``                                   U 形编排：高相关片段放首/尾，末尾重复问题
4      ``evaluation/``                                            Context Recall / Precision 分阶段度量 + Golden Set 回归门禁
5      ``retrieval/fallback.py``                                  扩 Top-K 重试 → 转人工 → 如实说没找到相关信息
====== ======================================================== ==================================

核心链路**零第三方依赖**（仅标准库）；LangChain / sentence-transformers /
pypdf 等为可选适配器，缺依赖时不影响 import 与默认链路运行。

快速开始::

    from cn.mcheat.rag import RAGSystem

    system = RAGSystem()
    system.index(["cn/mcheat/rag/examples/data"])
    system.save("./.rag_index")

    result = system.retrieve("父子切分解决什么问题？")
    if result.answer_ready:
        print(result.prompt)
    else:
        print(result.fallback.message)

从项目根目录运行，保证 ``cn.mcheat.rag`` 可被导入。完整说明见包内 ``README.md``。
"""
from __future__ import annotations

from .core.bootstrap import bootstrap

# import 本包即完成内置组件注册，用户代码可以直接用配置名替换组件
bootstrap()

from .api import RAGSystem  # noqa: E402  (必须在 bootstrap 之后)
from .config import RAGConfig, load_profile  # noqa: E402
from .core.interfaces import AnswerJudge, QueryRewriter, QueryTransformer  # noqa: E402
from .evaluation import (  # noqa: E402
    AnswerEvalReport,
    AnswerEvalRunner,
    EvalReport,
    EvalRunner,
    GoldenCase,
    GoldenSet,
    LexicalJudge,
    RegressionVerdict,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from .evaluation.metrics import GATED_METRICS  # noqa: E402
from .exceptions import (  # noqa: E402
    CacheError,
    ConfigError,
    DependencyMissingError,
    DimensionMismatchError,
    IndexFingerprintMismatchError,
    IndexingError,
    LoaderError,
    PersistenceError,
    QueryTransformError,
    RAGError,
    RetrievalError,
)
from .indexing import SourceManifest, SourceRecord  # noqa: E402
from .models import (  # noqa: E402
    BuiltContext,
    ChildChunk,
    Document,
    IndexStats,
    ParentChunk,
    RetrievalAction,
    RetrievalResult,
    ScoredChunk,
)
from .retrieval.cache import InMemoryCache, NullCache  # noqa: E402
from .retrieval.query_transform import (  # noqa: E402
    IdentityQueryTransformer,
    LLMQueryTransformer,
    QueryTransformPipeline,
    RuleBasedQueryTransformer,
)
from .vectorstores import SqliteVectorStore  # noqa: E402

__version__ = "2.0.0"

__all__ = [
    # 门面 / 配置
    "RAGSystem", "RAGConfig", "load_profile", "bootstrap", "__version__",
    # 领域模型
    "Document", "ParentChunk", "ChildChunk", "ScoredChunk", "BuiltContext",
    "IndexStats", "RetrievalResult", "RetrievalAction",
    # 接口与可选组件
    "QueryTransformer", "QueryRewriter", "AnswerJudge",
    "IdentityQueryTransformer", "RuleBasedQueryTransformer",
    "LLMQueryTransformer", "QueryTransformPipeline",
    "InMemoryCache", "NullCache", "SqliteVectorStore",
    "SourceManifest", "SourceRecord",
    # 评测
    "GoldenSet", "GoldenCase", "EvalRunner", "EvalReport",
    "RegressionVerdict", "context_recall", "context_precision",
    "GATED_METRICS", "AnswerEvalRunner", "AnswerEvalReport",
    "LexicalJudge", "faithfulness", "answer_relevancy",
    # 异常
    "RAGError", "ConfigError", "LoaderError", "IndexingError",
    "RetrievalError", "DimensionMismatchError", "DependencyMissingError",
    "PersistenceError", "QueryTransformError", "CacheError",
    "IndexFingerprintMismatchError",
]
