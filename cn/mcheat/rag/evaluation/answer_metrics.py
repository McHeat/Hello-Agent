"""答案级评测（P1-3）：Faithfulness / Answer Relevancy + 裁判注入。

**框架边界仍止于 ``result.prompt``**：本模块接收**调用方生成的答案**，
不代跑 LLM、不绑定厂商、不需要 API key。``AnswerJudge`` 是 Protocol，
默认提供零依赖的规则型 ``LexicalJudge`` 兜底（答案句与上下文的词面
支撑度）；CI 用假裁判即可完整验证链路。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._utils import clamp, get_logger, tokenize
from ..core.interfaces import AnswerJudge
from ..exceptions import EvaluationError, PersistenceError
from .golden import GoldenSet

logger = get_logger(__name__)


class LexicalJudge:
    """零依赖规则型裁判：用词元重叠近似「有依据 / 切题」。

    这是可跑、可测的兜底实现，不宣称等价于 LLM-as-judge；生产环境可注入
    任何满足 :class:`AnswerJudge` Protocol 的裁判。
    """

    def judge_faithfulness(self, question: str, answer: str,
                           context: str) -> float:
        """答案词元被上下文支撑的比例。"""
        tokens = tokenize(answer)
        if not tokens:
            return 0.0
        context_terms = set(tokenize(context))
        supported = sum(1 for term in tokens if term in context_terms)
        return clamp(supported / len(tokens), 0.0, 1.0)

    def judge_relevancy(self, question: str, answer: str) -> float:
        """问题词元在答案中出现的比例。"""
        question_terms = set(tokenize(question))
        if not question_terms:
            return 0.0
        answer_terms = set(tokenize(answer))
        return clamp(len(question_terms & answer_terms) / len(question_terms),
                     0.0, 1.0)


def faithfulness(question: str, answer: str, context: str,
                 judge: AnswerJudge | None = None) -> float:
    """单条 Faithfulness 分数（``[0,1]``）。"""
    score = (judge or LexicalJudge()).judge_faithfulness(
        question, answer, context
    )
    return clamp(float(score), 0.0, 1.0)


def answer_relevancy(question: str, answer: str,
                     judge: AnswerJudge | None = None) -> float:
    """单条 Answer Relevancy 分数（``[0,1]``）。"""
    score = (judge or LexicalJudge()).judge_relevancy(question, answer)
    return clamp(float(score), 0.0, 1.0)


@dataclass
class AnswerCaseResult:
    """单条样本的答案级评测明细。"""

    query: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    answerable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "answerable": self.answerable,
        }


@dataclass
class AnswerEvalReport:
    """一次答案级评测的聚合报告。"""

    golden_set: str = "default"
    cases: list[AnswerCaseResult] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "golden_set": self.golden_set,
            "summary": self.summary,
            "case_count": len(self.cases),
            "cases": [case.as_dict() for case in self.cases],
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise PersistenceError(f"保存答案级评测报告失败：{target}：{exc}") from exc


class AnswerEvalRunner:
    """在 Golden Set 上聚合调用方答案的 Faithfulness / Relevancy。

    Args:
        system: 提供 ``retrieve(query)`` 的对象（通常是 :class:`RAGSystem`）。
        golden_set: 人工标注集合。
        answers: ``query -> 调用方生成的答案``；缺省时回退到
            ``GoldenCase.expected_answer``。
        judge: 满足 ``AnswerJudge`` Protocol 的裁判；默认 ``LexicalJudge``。
    """

    def __init__(self, system: Any, golden_set: GoldenSet,
                 answers: Mapping[str, str] | None = None,
                 judge: AnswerJudge | None = None) -> None:
        self.system = system
        self.golden_set = golden_set
        self.answers = dict(answers or {})
        self.judge = judge or LexicalJudge()

    def run(self) -> AnswerEvalReport:
        """逐条评测并产出聚合报告。"""
        cases: list[AnswerCaseResult] = []
        for case in self.golden_set:
            try:
                result = self.system.retrieve(case.query)
            except Exception as exc:
                raise EvaluationError(
                    f"答案级评测 query={case.query!r} 时检索失败："
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            answer = self.answers.get(case.query, case.expected_answer)
            context = result.context.text if result.context is not None else ""
            cases.append(AnswerCaseResult(
                query=case.query,
                answer=answer,
                faithfulness=round(
                    faithfulness(case.query, answer, context, self.judge), 6
                ),
                answer_relevancy=round(
                    answer_relevancy(case.query, answer, self.judge), 6
                ),
                answerable=case.answerable,
            ))
        if not cases:
            raise EvaluationError("Golden Set 为空，无法生成答案级评测报告")

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        summary = {
            "faithfulness": round(mean([c.faithfulness for c in cases]), 6),
            "answer_relevancy": round(
                mean([c.answer_relevancy for c in cases]), 6
            ),
        }
        logger.info("答案级评测完成：%d 条，汇总 %s", len(cases), summary)
        return AnswerEvalReport(
            golden_set=self.golden_set.name, cases=cases, summary=summary,
        )


__all__: Sequence[str] = [
    "AnswerJudge", "LexicalJudge", "faithfulness", "answer_relevancy",
    "AnswerCaseResult", "AnswerEvalReport", "AnswerEvalRunner",
]
