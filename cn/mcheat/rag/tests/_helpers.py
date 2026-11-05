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
    return ScoredChunk(
        chunk=make_child(chunk_id, text),
        score=1.0 - rank * 0.1, rank=rank, parent=parent,
    )


def make_parent(parent_id: str, text: str) -> ParentChunk:
    return ParentChunk(
        id=parent_id, doc_id="d0", text=text,
        metadata={"source": "/tmp/a.txt", "filename": "a.txt"},
    )
