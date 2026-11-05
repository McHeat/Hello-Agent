"""在线检索模块：混合粗召回 → 融合 → 精排 → 父块还原 → 上下文编排 → 兜底。"""
from .bm25 import BM25Index
from .context import (
    EdgePlacementContextBuilder,
    SequentialContextBuilder,
    SmallToBigExpander,
    render_prompt,
)
from .fallback import FallbackChain
from .hybrid import HybridRetriever, RRFFusion, WeightedSumFusion
from .pipeline import OnlineRetriever
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
]
