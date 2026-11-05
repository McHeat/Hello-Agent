"""离线索引任务（可在独立进程 / 定时任务中运行）。

运行::

    python -m cn.mcheat.rag.examples.offline_build
    python -m cn.mcheat.rag.examples.offline_build /path/to/your/corpus

产物写到 ``examples/.rag_index/index.json``，含配置快照与三份索引状态。
"""
from __future__ import annotations

import sys
from pathlib import Path

from cn.mcheat.rag import RAGSystem

DEFAULT_CORPUS = Path(__file__).resolve().parent / "data"
INDEX_DIR = Path(__file__).resolve().parent / ".rag_index"


def main(sources: list[str] | None = None) -> None:
    paths = sources or [str(DEFAULT_CORPUS)]
    system = RAGSystem()
    stats = system.index(paths)
    print("离线索引统计：", stats.as_dict())
    artifact = system.save(INDEX_DIR)
    print(f"索引产物：{artifact}")
    print(f"向量条数：{system.component('dense_retriever').count()}，"
          f"BM25 条数：{system.component('sparse_retriever').count()}，"
          f"父块条数：{system.component('parent_store').count()}")


if __name__ == "__main__":
    main(sys.argv[1:])
