"""两级缓存（P1-1）：查询向量 LRU 缓存 + 检索结果 TTL 缓存。

两级 key 均含 ``index_version``（每次成功 ``index()`` 递增），因此索引
更新后旧缓存自然失效，不会脏读。默认 ``NullCache`` 全关，零依赖默认档
行为与改造前一致；日志规范：命中用 DEBUG，不写入查询文本 / 向量内容。
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Sequence

from .._utils import get_logger
from ..core.registry import CACHE_REGISTRY
from ..exceptions import CacheError
from ..models import Vector

logger = get_logger(__name__)


class EmbeddingQueryCache:
    """``(index_version, fingerprint, query) -> 向量`` 的线程安全 LRU 缓存。"""

    def __init__(self, max_size: int = 512) -> None:
        if max_size <= 0:
            raise CacheError("query_lru_size 必须为正整数")
        self.max_size = max_size
        self._entries: OrderedDict[tuple, Vector] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, fingerprint: str, query: str,
            index_version: int = 0) -> Vector | None:
        key = (index_version, fingerprint, query)
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            logger.debug("查询向量缓存命中")
            return self._entries[key]

    def put(self, fingerprint: str, query: str, index_version: int,
            vector: Vector) -> None:
        key = (index_version, fingerprint, query)
        with self._lock:
            self._entries[key] = tuple(float(x) for x in vector)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class RetrievalResultCache:
    """``(index_version, key) -> 结果`` 的线程安全 TTL 缓存。"""

    def __init__(self, ttl_s: float = 300.0) -> None:
        if ttl_s <= 0:
            raise CacheError("result_ttl_s 必须为正数")
        self.ttl_s = float(ttl_s)
        self._entries: dict[tuple, tuple[float, Any]] = {}
        self._lock = threading.RLock()

    def get(self, key: Any, index_version: int = 0) -> Any | None:
        composite = (index_version, _freeze(key))
        with self._lock:
            entry = self._entries.get(composite)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() >= expires_at:
                self._entries.pop(composite, None)
                return None
            logger.debug("检索结果缓存命中")
            return value

    def put(self, key: Any, result: Any, index_version: int = 0) -> None:
        composite = (index_version, _freeze(key))
        with self._lock:
            self._entries[composite] = (
                time.monotonic() + self.ttl_s, result
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _freeze(key: Any) -> Any:
    """把 key 统一转成可哈希结构（dict 按排序后的键值冻结）。"""
    if isinstance(key, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in key.items()))
    if isinstance(key, (list, tuple)):
        return tuple(_freeze(item) for item in key)
    if isinstance(key, set):
        return tuple(sorted(_freeze(item) for item in key))
    return key


class NullCache:
    """关闭态缓存：所有操作都是空操作，行为与未启用缓存完全一致。"""

    enabled = False

    def __init__(self, enabled: bool = False, query_lru_size: int = 512,
                 result_ttl_s: float = 300.0) -> None:
        self.index_version = 0

    def set_index_version(self, version: int) -> None:
        self.index_version = int(version)

    def get_query_vector(self, fingerprint: str, query: str) -> Vector | None:
        return None

    def put_query_vector(self, fingerprint: str, query: str,
                         vector: Vector) -> None:
        return None

    def get_result(self, key: Any) -> Any | None:
        return None

    def put_result(self, key: Any, result: Any) -> None:
        return None


@CACHE_REGISTRY.register("null")
class RegisteredNullCache(NullCache):
    """注册表名（与 :class:`NullCache` 行为一致）。"""


@CACHE_REGISTRY.register("memory")
class InMemoryCache(NullCache):
    """内存组合缓存：查询向量 LRU + 结果 TTL，key 均含 ``index_version``。"""

    def __init__(self, enabled: bool = True, query_lru_size: int = 512,
                 result_ttl_s: float = 300.0) -> None:
        super().__init__(enabled, query_lru_size, result_ttl_s)
        self.enabled = bool(enabled)
        self.index_version = 0
        self.query_cache = EmbeddingQueryCache(query_lru_size)
        self.result_cache = RetrievalResultCache(result_ttl_s)

    def set_index_version(self, version: int) -> None:
        self.index_version = int(version)

    def get_query_vector(self, fingerprint: str, query: str) -> Vector | None:
        if not self.enabled:
            return None
        return self.query_cache.get(fingerprint, query, self.index_version)

    def put_query_vector(self, fingerprint: str, query: str,
                         vector: Vector) -> None:
        if self.enabled:
            self.query_cache.put(fingerprint, query, self.index_version, vector)

    def get_result(self, key: Any) -> Any | None:
        if not self.enabled:
            return None
        return self.result_cache.get(key, self.index_version)

    def put_result(self, key: Any, result: Any) -> None:
        if self.enabled:
            self.result_cache.put(key, result, self.index_version)


__all__: Sequence[str] = [
    "EmbeddingQueryCache", "RetrievalResultCache", "NullCache",
    "RegisteredNullCache", "InMemoryCache",
]
