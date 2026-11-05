"""检索质量指标：Context Recall / Context Precision 等（纯函数，无副作用）。

《注意事项》第 4 条：「召回用 Context Recall（该检到的检到没）、精度用
Context Precision（检回的有多少真相关）**分开看**」。

这两个指标必须在不同阶段测，否则会互相掩盖：

* **Context Recall** 在**粗召回**阶段算：相关文档有没有被 30~50 条候选覆盖。
  粗召回丢了，后面精排再准也救不回来。
* **Context Precision** 在**精排后**算：真正进 prompt 的 3~5 条里有多少相关。
  这直接决定 token 是否被浪费、是否会误导 LLM。

所有函数对空集合都有明确定义（返回 0.0 而非抛错），避免评测跑到一半崩掉。
"""
from __future__ import annotations

from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def dedupe_preserve_order(ids: Iterable[str]) -> list[str]:
    """去重但保留首次出现顺序（Precision/AP 需要顺序语义）。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _as_set(ids: Iterable[str]) -> set[str]:
    return set(ids)


# ---------------------------------------------------------------------------
# 分开看的两个核心指标
# ---------------------------------------------------------------------------


def context_recall(retrieved: Iterable[str], relevant: Iterable[str]) -> float:
    """Context Recall = ``|检索到 ∩ 相关| / |相关|``。

    「该检到的检到没」。与顺序无关，也不受多召回噪声影响，因此适合在
    **粗召回阶段**度量。相关集合为空时返回 ``0.0``（Golden Set 加载时
    已禁止空标注，这里只是防御）。
    """
    relevant_set = _as_set(relevant)
    if not relevant_set:
        return 0.0
    return len(_as_set(retrieved) & relevant_set) / len(relevant_set)


def context_precision(retrieved: Iterable[str], relevant: Iterable[str]) -> float:
    """Context Precision = ``|检索到 ∩ 相关| / |检索到|``。

    「检回的有多少真相关」。检索结果为空返回 ``0.0``（空上下文不是高精度，
    而是不可用）。应在**精排后、真正进 prompt 的列表**上计算。
    """
    retrieved_list = dedupe_preserve_order(retrieved)
    if not retrieved_list:
        return 0.0
    relevant_set = _as_set(relevant)
    hits = sum(1 for item in retrieved_list if item in relevant_set)
    return hits / len(retrieved_list)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """前 ``k`` 条的 Context Precision。"""
    if k <= 0:
        return 0.0
    return context_precision(ranked[:k], relevant)


def average_precision(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """AP（Average Precision）：考虑**位置**的精度。

    对每个命中的相关项，计算到该位置为止的 Precision，再对相关项数取平均。
    高相关片段排得越靠前，AP 越高 —— 这正好能度量精排与上下文编排的收益。
    """
    ranked_list = dedupe_preserve_order(ranked)
    relevant_set = _as_set(relevant)
    if not relevant_set or not ranked_list:
        return 0.0
    hits = 0
    cumulative = 0.0
    for position, item in enumerate(ranked_list, start=1):
        if item in relevant_set:
            hits += 1
            cumulative += hits / position
    return cumulative / len(relevant_set)


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """MRR 单样本形式：第一个相关结果名次的倒数；无命中返回 0.0。"""
    relevant_set = _as_set(relevant)
    for position, item in enumerate(dedupe_preserve_order(ranked), start=1):
        if item in relevant_set:
            return 1.0 / position
    return 0.0


def hit_rate(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """是否至少命中一条相关结果（0/1）。"""
    return 1.0 if _as_set(ranked) & _as_set(relevant) else 0.0


def f1_score(precision: float, recall: float) -> float:
    """Precision / Recall 的调和平均；两者都为 0 时返回 0.0。"""
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_all(ranked: Sequence[str], relevant: Iterable[str]) -> dict[str, float]:
    """一次算出全部指标，供评测报告与回归门禁使用。"""
    recall = context_recall(ranked, relevant)
    precision = context_precision(ranked, relevant)
    return {
        "context_recall": recall,
        "context_precision": precision,
        "average_precision": average_precision(ranked, relevant),
        "precision_at_3": precision_at_k(ranked, relevant, 3),
        "reciprocal_rank": reciprocal_rank(ranked, relevant),
        "hit_rate": hit_rate(ranked, relevant),
        "f1": f1_score(precision, recall),
    }


# 回归门禁只比较这几项（其余指标用于人工分析，不设硬门槛）
GATED_METRICS: tuple[str, ...] = (
    "context_recall", "context_precision", "average_precision", "hit_rate",
)


__all__: Sequence[str] = [
    "dedupe_preserve_order", "context_recall", "context_precision",
    "precision_at_k", "average_precision", "reciprocal_rank", "hit_rate",
    "f1_score", "compute_all", "GATED_METRICS",
]
