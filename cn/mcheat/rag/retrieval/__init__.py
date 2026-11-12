"""在线检索模块：查询变换 / 缓存 → 混合粗召回 → 融合 → 精排 → 上下文 → 兜底。"""
from .bm25 import BM25Index
from .cache import (
    EmbeddingQueryCache,
    InMemoryCache,
    NullCache,
    RetrievalResultCache,
)
from .context import (
    EdgePlacementContextBuilder,
    SequentialContextBuilder,
    SmallToBigExpander,
    render_prompt,
)
from .fallback import FallbackChain
from .hybrid import HybridRetriever, RRFFusion, WeightedSumFusion
from .pipeline import OnlineRetriever
from .query_transform import (
    IdentityQueryTransformer,
    LLMQueryTransformer,
    QueryTransformPipeline,
    RuleBasedQueryTransformer,
)
from .recall import (
    EmbeddingDenseRetriever,
    LangChainDenseRetriever,
    NullSparseRetriever,
)
from .rerank import CrossEncoderReranker, LexicalOverlapReranker, NoopReranker

__all__ = [
    "BM25Index", "RRFFusion", "WeightedSumFusion", "HybridRetriever",
    "EmbeddingDenseRetriever", "LangChainDenseRetriever", "NullSparseRetriever",
    "CrossEncoderReranker", "LexicalOverlapReranker", "NoopReranker",
    "SmallToBigExpander", "EdgePlacementContextBuilder",
    "SequentialContextBuilder", "render_prompt", "FallbackChain",
    "OnlineRetriever",
    "IdentityQueryTransformer", "RuleBasedQueryTransformer",
    "LLMQueryTransformer", "QueryTransformPipeline",
    "EmbeddingQueryCache", "RetrievalResultCache", "NullCache", "InMemoryCache",
]
