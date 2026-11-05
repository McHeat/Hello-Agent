"""Golden Set 回归评测：Context Recall / Context Precision 分阶段度量。"""
from .golden import GoldenCase, GoldenSet
from .metrics import (
    average_precision,
    compute_all,
    context_precision,
    context_recall,
    f1_score,
    hit_rate,
    precision_at_k,
    reciprocal_rank,
)
from .runner import (
    CaseResult,
    EvalReport,
    EvalRunner,
    RegressionVerdict,
    extract_ids,
    resolve_id,
)

__all__ = [
    "GoldenCase", "GoldenSet", "context_recall", "context_precision",
    "precision_at_k", "average_precision", "reciprocal_rank", "hit_rate",
    "f1_score", "compute_all", "EvalRunner", "EvalReport", "CaseResult",
    "RegressionVerdict", "resolve_id", "extract_ids",
]
