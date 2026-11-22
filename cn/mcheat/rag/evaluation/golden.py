"""Golden Set：几十到几百条人工标注的回归集（《注意事项》第 4 条）。

原文：「用指标量化，别拍脑袋调……配合一个几十到几百条的人工标注 Golden Set
做回归」。

标注格式（JSON）::

    {
      "name": "rag-v1",
      "cases": [
        {
          "query": "父子切分解决什么问题？",
          "relevant_ids": ["rag_notes.md"],
          "notes": "标注的是来源文件名，与 evaluation.id_field 对应"
        },
        {
          "query": "帕米尔高原雪豹栖息地海拔",
          "relevant_ids": [],
          "answerable": false,
          "notes": "应拒答负样本：知识库里没有相关文档"
        }
      ]
    }

``relevant_ids`` 的语义由 ``evaluation.id_field`` 决定，默认是 ``source``
对应的 ``filename``。**负样本**（``answerable=false``）允许空 ``relevant_ids``，
用于阈值标定与拒答指标；旧的正样本 JSON 无需任何改动即合法。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..exceptions import EvaluationError, PersistenceError


@dataclass(frozen=True)
class GoldenCase:
    """一条标注样本。

    Attributes:
        query: 用户问题。
        relevant_ids: 被判定为「相关」的 ID 集合；可回答样本必须非空，
            负样本允许为空。
        notes: 标注备注，便于复核。
        answerable: 知识库是否应当能回答；默认 ``True``（兼容旧 JSON）。
        expected_answer: 参考回答（答案级评测用；默认空，框架不代跑 LLM）。
    """

    query: str
    relevant_ids: frozenset[str]
    notes: str = ""
    answerable: bool = True
    expected_answer: str = ""

    @property
    def should_refuse(self) -> bool:
        """是否应当拒答（由 ``answerable`` 派生）。"""
        return not self.answerable

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoldenCase":
        if not isinstance(data, Mapping):
            raise EvaluationError(f"GoldenCase 必须是对象，收到 {type(data).__name__}")
        query = data.get("query")
        if not isinstance(query, str) or not query.strip():
            raise EvaluationError(f"GoldenCase 缺少非空 query：{dict(data)!r}")
        answerable = bool(data.get("answerable", True))
        raw_ids = data.get("relevant_ids", [])
        if not isinstance(raw_ids, (list, tuple, set, frozenset)):
            raise EvaluationError(
                f"query={query!r} 的 relevant_ids 必须是列表/集合；"
                f"没有相关标注的可回答样本无法定义 Context Recall。"
            )
        ids = frozenset(str(item) for item in raw_ids)
        if answerable and not ids:
            raise EvaluationError(
                f"query={query!r} 是可回答样本，relevant_ids 不能为空；"
                f"无相关文档请显式标注 answerable=false。"
            )
        return cls(
            query=query.strip(), relevant_ids=ids,
            notes=str(data.get("notes", "")),
            answerable=answerable,
            expected_answer=str(data.get("expected_answer", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "relevant_ids": sorted(self.relevant_ids),
            "notes": self.notes,
            "answerable": self.answerable,
            "expected_answer": self.expected_answer,
        }


@dataclass
class GoldenSet:
    """一组人工标注样本。"""

    name: str = "default"
    cases: list[GoldenCase] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    @classmethod
    def from_json(cls, path: str | Path) -> "GoldenSet":
        """从 JSON 文件加载并逐条校验。"""
        target = Path(path)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise EvaluationError(f"Golden Set 文件不存在：{target}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"Golden Set 文件损坏：{target}：{exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoldenSet":
        if not isinstance(data, Mapping):
            raise EvaluationError("Golden Set 顶层必须是对象")
        raw_cases = data.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise EvaluationError("Golden Set 的 cases 必须是非空列表")
        queries: set[str] = set()
        cases: list[GoldenCase] = []
        for index, raw in enumerate(raw_cases):
            case = GoldenCase.from_dict(raw)
            if case.query in queries:
                raise EvaluationError(f"第 {index + 1} 条重复的 query：{case.query!r}")
            queries.add(case.query)
            cases.append(case)
        return cls(name=str(data.get("name", "default")), cases=cases)

    @classmethod
    def from_cases(cls, cases: Iterable[GoldenCase], name: str = "default") -> "GoldenSet":
        """从内存中的样本构造（便于测试）。"""
        return cls(name=name, cases=list(cases))

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "cases": [case.to_dict() for case in self.cases]}

    def save(self, path: str | Path) -> None:
        """保存为 JSON（UTF-8、缩进 2）。"""
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise PersistenceError(f"保存 Golden Set 失败：{target}：{exc}") from exc


__all__: Sequence[str] = ["GoldenCase", "GoldenSet"]
