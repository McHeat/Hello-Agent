"""零依赖的确定性 Embedding（Hashing Trick）。

**用途定位**：让整个框架在「没有模型、没有 API key、没有 numpy」的环境下
也能端到端跑通与测试，并保证结果可复现。它不是语义模型，只捕捉词面重叠；
生产环境请把 ``components.embedder.name`` 换成 ``"langchain"``（见
``embeddings/adapters.py``）或自行注册真正的 Embedding 实现。

实现要点：

* 用 :func:`_utils.tokenize` 切词（中英文一致），term 经 ``blake2b`` 哈希到
  ``dimension`` 维；用一位哈希值做符号（signed hashing）降低碰撞偏置；
* 对累加计数取 ``log1p`` 做次线性抑制，避免高频词主导向量；
* L2 归一化，使 ``InMemoryVectorStore`` 的内积即余弦相似度；
* 查询与文档共用同一编码函数，避免「索引侧 / 查询侧不一致」这类隐形 bug。
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

from .._utils import get_logger, l2_normalize, tokenize
from ..core.interfaces import Embedder
from ..core.registry import EMBEDDER_REGISTRY
from ..exceptions import ConfigError
from ..models import Vector

logger = get_logger(__name__)


@EMBEDDER_REGISTRY.register("hashing")
class HashingEmbedder(Embedder):
    """基于特征哈希的确定性向量化器。

    Args:
        dimension: 向量维度。维度越高碰撞越少，但内存 / 计算量线性增长。
            **换维度会导致已有向量库报错**（``DimensionMismatchError``），
            变更时请重建索引。
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ConfigError("embedding dimension 必须为正整数")
        self._dimension = dimension
        logger.debug("HashingEmbedder(dimension=%d)", dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_one(self, text: str) -> Vector:
        values = [0.0] * self._dimension
        for term in tokenize(text):
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            values[index] += sign
        # 次线性缩放：log1p(|v|) 并保留符号，弱化高频词
        damped = [math.copysign(math.log1p(abs(v)), v) for v in values]
        return l2_normalize(damped)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        # 哈希编码不区分 query/document 通道；真实模型可在此加 query 前缀
        return self._embed_one(text)


__all__: Sequence[str] = ["HashingEmbedder"]
