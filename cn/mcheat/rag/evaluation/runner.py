"""Golden Set 回归评测执行器与报告（《注意事项》第 4 条）。

关键设计：**召回与精度分开、分阶段度量**。

对每条标注样本运行一次完整在线检索，然后：

* 用 ``result.coarse``（粗召回 Top-30~50）算 ``context_recall``；
* 用 ``result.final``（精排后进 prompt 的 Top-3~5）算 ``context_precision``
  与其它排序指标。

这样调参时一眼能看出问题出在哪一段：Recall 低 → 该改 embedding / BM25 /
chunk 切分；Precision 低 → 该改精排 / Top-N / 兜底阈值。而不是盯着一个
模糊的「准确率」拍脑袋。

报告可存为基线 JSON，``regression_check`` 会在 CI 里比对指标跌幅并给出
通过/失败结论，防止一次「优化」悄悄拖垮另一项指标。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .._utils import get_logger
from ..config import EvaluationConfig
from ..exceptions import EvaluationError, PersistenceError
from ..models import RetrievalResult, ScoredChunk
from .golden import GoldenSet
from .metrics import GATED_METRICS, compute_all, context_precision, context_recall

if TYPE_CHECKING:  # 仅类型标注，避免与 api.RAGSystem 形成运行时循环导入
    from ..api import RAGSystem

logger = get_logger(__name__)


def resolve_id(item: ScoredChunk, id_field: str) -> str:
    """按配置把一条检索结果映射成 Golden Set 里标注的 ID。

    Args:
        item: 精排后的候选。
        id_field: ``chunk_id`` / ``parent_id`` / ``doc_id``，或任意 metadata
            字段名（默认 ``source`` 会取 ``filename``）。

    Returns:
        ID 字符串；缺失时返回空串（评测时会被过滤掉）。
    """
    if id_field == "chunk_id":
        return item.chunk.id
    if id_field == "parent_id":
        return item.parent.id if item.parent is not None else item.chunk.parent_id
    if id_field == "doc_id":
        return item.chunk.doc_id
    metadata = item.chunk.metadata
    # 「source」标注通常用文件名更友好，故对 source 做特殊映射
    if id_field == "source" and metadata.get("filename"):
        return str(metadata["filename"])
    value = metadata.get(id_field)
    return "" if value is None else str(value)


def extract_ids(items: list[ScoredChunk], id_field: str) -> list[str]:
    """提取 ID 列表，保留顺序并过滤空值。"""
    result: list[str] = []
    for item in items:
        value = resolve_id(item, id_field)
        if value:
            result.append(value)
    return result


@dataclass
class CaseResult:
    """单条标注样本的评测明细。"""

    query: str
    relevant_ids: list[str]
    coarse_ids: list[str]
    final_ids: list[str]
    recall_metrics: dict[str, float]
    precision_metrics: dict[str, float]
    action: str
    answer_ready: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "relevant_ids": self.relevant_ids,
            "coarse_ids": self.coarse_ids,
            "final_ids": self.final_ids,
            "recall_metrics": self.recall_metrics,
            "precision_metrics": self.precision_metrics,
            "action": self.action,
            "answer_ready": self.answer_ready,
        }


@dataclass
class EvalReport:
    """一次回归评测的聚合报告。"""

    golden_set: str = "default"
    cases: list[CaseResult] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "golden_set": self.golden_set,
            "summary": self.summary,
            "case_count": len(self.cases),
            "config_snapshot": self.config_snapshot,
            "cases": [case.as_dict() for case in self.cases],
        }

    def save(self, path: str | Path) -> None:
        """保存为 JSON，可直接作为下一次回归的 baseline。"""
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise PersistenceError(f"保存评测报告失败：{target}：{exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "EvalReport":
        """从 JSON 恢复报告（主要用 summary 做基线比对）。"""
        target = Path(path)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"读取基线报告失败：{target}：{exc}") from exc
        return cls(
            golden_set=str(payload.get("golden_set", "default")),
            summary={str(k): float(v) for k, v in payload.get("summary", {}).items()},
            config_snapshot=payload.get("config_snapshot", {}),
        )

    def regression_check(self, baseline: "EvalReport | dict[str, Any]",
                         max_drop: float = 0.02) -> "RegressionVerdict":
        """与基线比对，判断是否发生了不可接受的指标回退。

        Args:
            baseline: :class:`EvalReport` 或含 ``summary`` 的 dict。
            max_drop: 允许的绝对跌幅（指标都在 ``[0, 1]``，默认 0.02）。

        Returns:
            :class:`RegressionVerdict`。
        """
        base_summary = (
            baseline.summary if isinstance(baseline, EvalReport)
            else dict(baseline.get("summary", {}))
        )
        deltas: dict[str, float] = {}
        failed: list[str] = []
        for metric in GATED_METRICS:
            if metric not in base_summary or metric not in self.summary:
                continue
            delta = self.summary[metric] - float(base_summary[metric])
            deltas[metric] = round(delta, 6)
            if delta < -abs(max_drop):
                failed.append(f"{metric}: {delta:+.4f}")
        passed = not failed
        message = (
            "回归通过：各项指标均未低于基线"
            if passed else
            "回归失败，以下指标相对基线下降超过阈值 "
            f"{max_drop}：{'；'.join(failed)}"
        )
        return RegressionVerdict(passed=passed, deltas=deltas,
                                 failed_metrics=failed, message=message)


@dataclass
class RegressionVerdict:
    """回归门禁结论（CI 可直接用 ``passed`` 决定退出码）。"""

    passed: bool
    deltas: dict[str, float] = field(default_factory=dict)
    failed_metrics: list[str] = field(default_factory=list)
    message: str = ""


class EvalRunner:
    """在 Golden Set 上批量运行检索并汇总指标。

    Args:
        system: :class:`RAGSystem`（或任何提供 ``retrieve(query)`` 与
            ``config`` 的对象，便于测试替身）。
        golden_set: 人工标注集合。
        config: 评测配置；默认取 ``system.config.evaluation``。
    """

    def __init__(self, system: "RAGSystem", golden_set: GoldenSet,
                 config: EvaluationConfig | None = None) -> None:
        self.system = system
        self.golden_set = golden_set
        self.config = config or system.config.evaluation

    def run(self) -> EvalReport:
        """逐条执行检索、计算分阶段指标、汇总均值。

        Raises:
            EvaluationError: 检索过程抛出异常（会带上具体 query）。
        """
        cases: list[CaseResult] = []
        id_field = self.config.id_field
        for case in self.golden_set:
            try:
                result: RetrievalResult = self.system.retrieve(case.query)
            except Exception as exc:
                raise EvaluationError(
                    f"评测 query={case.query!r} 时检索失败："
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            coarse_ids = extract_ids(result.coarse, id_field)
            final_ids = extract_ids(result.final, id_field)
            relevant = sorted(case.relevant_ids)
            stage_map = {"coarse": coarse_ids, "final": final_ids}

            # 注意：recall_stage / precision_stage 可配置为不同阶段，
            # 默认严格按注意事项第 4 条「分开看」执行。
            recall_ids = stage_map.get(self.config.recall_stage, coarse_ids)
            precision_ids = stage_map.get(self.config.precision_stage, final_ids)

            recall_metrics = {
                "context_recall": context_recall(recall_ids, relevant),
                "hit_rate": float(bool(set(recall_ids) & set(relevant))),
            }
            precision_metrics = compute_all(precision_ids, relevant)
            cases.append(CaseResult(
                query=case.query,
                relevant_ids=relevant,
                coarse_ids=coarse_ids,
                final_ids=final_ids,
                recall_metrics=recall_metrics,
                precision_metrics=precision_metrics,
                action=result.fallback.final_action,
                answer_ready=result.answer_ready,
            ))

        if not cases:
            raise EvaluationError("Golden Set 为空，无法生成评测报告")

        # 汇总：粗召回阶段的 Recall + 精排阶段的 Precision/AP/...
        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        summary: dict[str, float] = {
            "context_recall": mean([c.recall_metrics["context_recall"] for c in cases]),
            "recall_hit_rate": mean([c.recall_metrics["hit_rate"] for c in cases]),
        }
        for metric in ("context_precision", "average_precision",
                       "precision_at_3", "reciprocal_rank", "hit_rate", "f1"):
            summary[metric] = mean([c.precision_metrics[metric] for c in cases])
        summary["answer_ready_rate"] = mean(
            [1.0 if c.answer_ready else 0.0 for c in cases]
        )
        summary = {key: round(value, 6) for key, value in summary.items()}

        report = EvalReport(
            golden_set=self.golden_set.name,
            cases=cases,
            summary=summary,
            config_snapshot={
                "id_field": id_field,
                "recall_stage": self.config.recall_stage,
                "precision_stage": self.config.precision_stage,
                "coarse_top_k": self.system.config.retrieval.coarse_top_k,
                "rerank_top_n": self.system.config.retrieval.rerank_top_n,
                "context_placement": self.system.config.component(
                    "context_builder"
                ).params.get("placement", "unknown"),
            },
        )
        logger.info("Golden Set 评测完成：%d 条，汇总 %s", len(cases), summary)
        return report


__all__: Sequence[str] = [
    "CaseResult", "EvalReport", "RegressionVerdict", "EvalRunner",
    "resolve_id", "extract_ids",
]
