"""测试共用构造器，避免每个用例重复拼装领域对象。"""
from __future__ import annotations

from cn.mcheat.rag.models import (
    ChildChunk,
    Document,
    FusedHit,
    Hit,
    ParentChunk,
    ScoredChunk,
)


def make_documents(n: int = 3, prefix: str = "文档") -> list[Document]:
    return [
        Document(
            page_content=f"{prefix}{i} 的内容。" * 20,
            metadata={"source": f"/tmp/doc{i}.txt", "filename": f"doc{i}.txt"},
        )
        for i in range(n)
    ]


def make_child(chunk_id: str, text: str, parent_id: str = "p0",
               **metadata) -> ChildChunk:
    return ChildChunk(
        id=chunk_id, parent_id=parent_id, doc_id="d0", text=text,
        metadata={"source": "/tmp/a.txt", "filename": "a.txt", **metadata},
        start=0, end=len(text),
    )


def make_hit(chunk_id: str, text: str, score: float,
             source: str = "dense") -> Hit:
    return Hit(chunk=make_child(chunk_id, text), score=score, source=source)


def make_fused(chunk_id: str, text: str, score: float,
               sources: dict | None = None) -> FusedHit:
    return FusedHit(
        chunk=make_child(chunk_id, text), score=score,
        sources=sources or {"dense": score}, ranks={"dense": 1},
    )


def make_scored(rank: int, chunk_id: str, text: str,
                parent: ParentChunk | None = None) -> ScoredChunk:
    score = 1.0 - rank * 0.1
    return ScoredChunk(
        chunk=make_child(chunk_id, text),
        score=score, rank=rank, raw_score=score, parent=parent,
    )


def make_normalized_scored(rank: int, chunk_id: str, text: str,
                           score: float | None = None,
                           raw_score: float | None = None,
                           parent: ParentChunk | None = None) -> ScoredChunk:
    """构造「score 已归一化、raw_score 保留精排原始分」的候选。"""
    normalized = 1.0 - rank * 0.1 if score is None else score
    return ScoredChunk(
        chunk=make_child(chunk_id, text),
        score=normalized, rank=rank,
        raw_score=normalized if raw_score is None else raw_score,
        parent=parent,
    )


def make_fake_judge(faithfulness: float = 1.0,
                    relevancy: float = 1.0):
    """返回一个满足 AnswerJudge Protocol 的零依赖假裁判。"""
    class _Judge:
        def judge_faithfulness(self, question, answer, context):
            return faithfulness

        def judge_relevancy(self, question, answer):
            return relevancy

    return _Judge()


def make_negative_golden_set(n: int = 3, name: str = "negatives"):
    """构造 n 条应拒答负样本（旧 JSON 格式中的正样本不受影响）。"""
    from cn.mcheat.rag.evaluation.golden import GoldenCase, GoldenSet

    cases = [
        GoldenCase(
            query=f"负样本无关问题编号 {i} 帕米尔高原雪豹栖息地考据",
            relevant_ids=frozenset(),
            answerable=False,
        )
        for i in range(n)
    ]
    return GoldenSet.from_cases(cases, name=name)


def make_parent(parent_id: str, text: str) -> ParentChunk:
    return ParentChunk(
        id=parent_id, doc_id="d0", text=text,
        metadata={"source": "/tmp/a.txt", "filename": "a.txt"},
    )
