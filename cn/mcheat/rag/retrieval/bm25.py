"""BM25 稀疏检索（Okapi BM25，纯标准库实现）。

《注意事项》第 1 条要求「向量 + BM25 混合召回」：稠密向量擅长语义近似，
BM25 擅长精确词/术语匹配（型号、错误码、专有名词），两者互补。这里不引入
``rank_bm25`` / Elasticsearch，用倒排统计自己算，保证核心零依赖。

评分公式：

.. math::

    score(q, d) = \\sum_{t \\in q} IDF(t) \\cdot
        \\frac{tf(t,d) \\cdot (k_1+1)}
             {tf(t,d) + k_1 \\cdot (1 - b + b \\cdot |d| / avgdl)}

其中 :math:`IDF(t) = \\ln\\left(1 + \\frac{N - df(t) + 0.5}{df(t) + 0.5}\\right)`
（加 1 平滑，保证 IDF 非负，避免罕见词把分数压成负数）。
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .._utils import get_logger, tokenize
from ..core.interfaces import RecallRetriever
from ..core.registry import SPARSE_RETRIEVER_REGISTRY
from ..exceptions import ConfigError
from ..models import ChildChunk, Hit

logger = get_logger(__name__)


@SPARSE_RETRIEVER_REGISTRY.register("bm25")
class BM25Index(RecallRetriever):
    """内存 BM25 倒排索引。

    Args:
        k1: 词频饱和参数，越大越偏向高频词（常用 1.2~2.0）。
        b: 文档长度归一化强度，0 表示不归一化，1 表示完全归一化（常用 0.75）。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0.0 <= b <= 1.0:
            raise ConfigError("BM25 参数需满足 k1 > 0 且 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, ChildChunk] = {}
        self._tf: dict[str, dict[str, int]] = {}   # chunk_id -> {term: count}
        self._lengths: dict[str, int] = {}         # chunk_id -> 词元总数
        self._df: dict[str, int] = {}              # term -> 包含该词的文档数
        self._avgdl: float = 0.0

    # -- 离线写入 -----------------------------------------------------------

    def index(self, chunks: Sequence[ChildChunk]) -> int:
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            old_terms = set(self._tf.get(chunk.id, {}))
            self._remove_df(old_terms)  # upsert：先撤销旧统计，避免 df 虚高
            self._chunks[chunk.id] = chunk
            self._tf[chunk.id] = counts
            self._lengths[chunk.id] = len(tokens)
            for term in counts:
                self._df[term] = self._df.get(term, 0) + 1
        self._recompute_avgdl()
        logger.debug("BM25 索引写入 %d 条，当前 %d 条", len(chunks), self.count())
        return len(chunks)

    # -- 在线检索 -----------------------------------------------------------

    def search(self, query: str, k: int,
               metadata_filter: dict[str, Any] | None = None) -> list[Hit]:
        if k <= 0 or not self._chunks:
            return []
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        total_docs = len(self._chunks)
        scores: list[tuple[float, str]] = []
        for chunk_id, counts in self._tf.items():
            chunk = self._chunks[chunk_id]
            if metadata_filter and not all(
                chunk.metadata.get(key) == value
                for key, value in metadata_filter.items()
            ):
                continue
            score = self._score(query_terms, counts, chunk_id, total_docs)
            if score > 0.0:
                scores.append((score, chunk_id))

        # 分数降序 + ID 兜底排序，结果可复现
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [
            Hit(chunk=self._chunks[chunk_id], score=float(score), source="sparse")
            for score, chunk_id in scores[:k]
        ]

    def _score(self, query_terms: set[str], counts: dict[str, int],
               chunk_id: str, total_docs: int) -> float:
        """计算单篇文档对查询的 BM25 分数。"""
        doc_len = max(1, self._lengths.get(chunk_id, 0))
        norm = 1.0 - self.b + self.b * doc_len / max(self._avgdl, 1.0)
        total = 0.0
        for term in query_terms:
            tf = counts.get(term)
            if not tf:
                continue
            df = self._df.get(term, 0)
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            total += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        return total

    # -- 维护 ---------------------------------------------------------------

    def delete(self, ids: Sequence[str]) -> int:
        removed = 0
        for chunk_id in ids:
            if chunk_id not in self._chunks:
                continue
            self._remove_df(set(self._tf.get(chunk_id, {})))
            self._chunks.pop(chunk_id, None)
            self._tf.pop(chunk_id, None)
            self._lengths.pop(chunk_id, None)
            removed += 1
        if removed:
            self._recompute_avgdl()
        return removed

    def ids(self) -> set[str]:
        return set(self._chunks)

    def count(self) -> int:
        return len(self._chunks)

    def _remove_df(self, terms: set[str]) -> None:
        for term in terms:
            remaining = self._df.get(term, 0) - 1
            if remaining <= 0:
                self._df.pop(term, None)
            else:
                self._df[term] = remaining

    def _recompute_avgdl(self) -> None:
        self._avgdl = (
            sum(self._lengths.values()) / len(self._lengths)
            if self._lengths else 0.0
        )

    # -- 持久化 -------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "documents": [
                {
                    "chunk": self._chunks[chunk_id].to_dict(),
                    "tf": self._tf[chunk_id],
                    "length": self._lengths[chunk_id],
                }
                for chunk_id in sorted(self._chunks)
            ],
        }

    def import_state(self, state: dict[str, Any]) -> None:
        self.k1 = float(state.get("k1", self.k1))
        self.b = float(state.get("b", self.b))
        self._chunks.clear()
        self._tf.clear()
        self._lengths.clear()
        self._df.clear()
        for record in state.get("documents", []):
            chunk = ChildChunk.from_dict(record["chunk"])
            counts = dict(record["tf"])
            self._chunks[chunk.id] = chunk
            self._tf[chunk.id] = counts
            self._lengths[chunk.id] = int(record["length"])
            for term in counts:
                self._df[term] = self._df.get(term, 0) + 1
        self._recompute_avgdl()


__all__: Sequence[str] = ["BM25Index"]
