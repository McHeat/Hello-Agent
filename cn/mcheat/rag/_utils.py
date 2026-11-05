"""包内部通用工具（零第三方依赖）。

本模块位于依赖链最底层，**不得 import 本包内任何其他模块**，以避免循环导入。
提供：稳定哈希 ID、中英文混合分词、向量归一化 / 余弦相似度、统一日志入口。
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import unicodedata
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

_PACKAGE_LOGGER = "cn.mcheat.rag"
_ROOT_LOGGER = logging.getLogger(_PACKAGE_LOGGER)
# 库代码永远不自行配置 handler（不调用 basicConfig），由应用侧决定输出方式；
# 加 NullHandler 只是为了避免 "No handler could be found" 告警。
_ROOT_LOGGER.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """返回一个位于 ``cn.mcheat.rag`` 命名空间下的 logger。

    Args:
        name: 一般传 ``__name__``；会自动剥掉包前缀，避免重复。

    Returns:
        ``logging.Logger`` 实例。应用侧可用
        ``logging.getLogger("cn.mcheat.rag").setLevel(logging.DEBUG)`` 打开调试日志。
    """
    suffix = name.rsplit(_PACKAGE_LOGGER, 1)[-1].lstrip(".")
    return _ROOT_LOGGER.getChild(suffix) if suffix else _ROOT_LOGGER


# ---------------------------------------------------------------------------
# 稳定 ID
# ---------------------------------------------------------------------------


def stable_id(*parts: object, length: int = 16) -> str:
    """根据若干入参生成**跨进程稳定**的短哈希 ID。

    与 ``uuid4`` 不同，同样的输入永远得到同样的 ID。这对离线索引的幂等性
    至关重要：重复建索引时子块 ID 不变，向量库执行 upsert 而非累积重复数据。

    Args:
        *parts: 参与哈希的组成部分，按序以 ``\\x1f`` 拼接后转字符串。
        length: 截断长度（十六进制字符数），默认 16（64 bit，冲突概率可忽略）。

    Returns:
        十六进制摘要字符串，例如 ``"a3f19c02bd47e815"``。
    """
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:length]


def content_hash(text: str) -> str:
    """文本内容的 sha256 摘要，用于去重与变更检测。"""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# 分词：同时兼顾英文单词与中文（无 jieba / 无第三方依赖）
# ---------------------------------------------------------------------------

# 匹配「连续的 ASCII 字母数字串」或「连续的 CJK 表意文字串」
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u9fff\uf900-\ufaff]+")
_CJK_RE = re.compile(r"^[\u3400-\u9fff\uf900-\ufaff]")


def tokenize(text: str) -> list[str]:
    """把文本切成检索用的词元（term）。

    规则：

    * 先做 ``NFKC`` 归一化并转小写，让全角英文/数字与半角等价；
    * 连续的 ASCII 字母数字作为一个词元（``"RAG"`` → ``"rag"``）；
    * 连续的中文串切成**单字 + 相邻二元组**（``"向量库"`` →
      ``"向", "向量", "量", "量库", "库"``）。用字符 bigram 近似中文分词，
      不需要 jieba 等分词器，且对未登录词更鲁棒。

    BM25、HashingEmbedder、词面重叠精排器共用此函数，保证「索引侧」与
    「查询侧」的切分方式严格一致（否则会出现永远检不中的 bug）。

    Args:
        text: 任意文本，可以为空。

    Returns:
        词元列表；空文本返回 ``[]``。
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalized):
        segment = match.group()
        if _CJK_RE.match(segment):
            # 中文：单字 + bigram
            for i, char in enumerate(segment):
                tokens.append(char)
                if i + 1 < len(segment):
                    tokens.append(segment[i : i + 2])
        else:
            tokens.append(segment)
    return tokens


# ---------------------------------------------------------------------------
# 向量运算（纯 Python；小规模语料足够，生产环境请替换为 FAISS / Chroma）
# ---------------------------------------------------------------------------


def l2_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    """L2 归一化。零向量原样返回（全零），避免除零。"""
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if norm == 0.0:
        return tuple(float(x) for x in vector)
    return tuple(float(x) / norm for x in vector)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """计算两个**已归一化**向量的余弦相似度，结果落在 ``[-1, 1]``。

    调用方保证维度一致（``InMemoryVectorStore`` 会在写入时校验）。
    若存在零向量，返回 ``0.0``。
    """
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致：{len(a)} != {len(b)}")
    if not a:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def clamp(value: float, low: float, high: float) -> float:
    """把数值限制在 ``[low, high]`` 区间内。"""
    return max(low, min(high, value))


def batched(items: Sequence, size: int) -> Iterable[Sequence]:
    """按 ``size`` 切片迭代，用于批量 embedding，避免一次塞爆内存/接口。"""
    if size <= 0:
        raise ValueError("batch size 必须为正整数")
    for start in range(0, len(items), size):
        yield items[start : start + size]
