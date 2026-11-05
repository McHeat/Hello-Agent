"""在线检索服务入口（只读索引产物，不重新建索引）。

运行::

    python -m cn.mcheat.rag.examples.offline_build          # 先建索引
    python -m cn.mcheat.rag.examples.online_query "兜底链路有哪些降级手段"
    python -m cn.mcheat.rag.examples.online_query "Lost in the Middle 怎么处理" --show-prompt

**重要**：只有 ``result.answer_ready`` 为真时才把 ``result.prompt`` 交给 LLM；
否则直接返回 ``result.fallback.message``（转人工 / 如实说没找到相关信息）。
"""
from __future__ import annotations

import sys
from pathlib import Path

from cn.mcheat.rag import RAGSystem

INDEX_DIR = Path(__file__).resolve().parent / ".rag_index"


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    show_prompt = "--show-prompt" in args
    args = [item for item in args if not item.startswith("--")]
    query = " ".join(args) or "父子切分解决什么问题？"

    system = RAGSystem.load(INDEX_DIR)
    result = system.retrieve(query)

    print(f"查询：{query}")
    print(f"粗召回 {len(result.coarse)} 条 → 精排后 {len(result.final)} 条进 prompt")
    d = result.diagnostics
    print(f"诊断：attempts={d['attempts']}，Top-K={d['final_coarse_top_k']}，"
          f"上下文 {d['context_chars']} 字符，耗时 {d['elapsed_ms']}ms")
    print(f"兜底：triggered={result.fallback.triggered}，action={result.fallback.final_action}")

    if result.answer_ready:
        print("\n最终引用的资料：")
        for item in result.final:
            source = item.chunk.metadata.get("filename", "?")
            print(f"  [{item.rank + 1}] {source}  score={item.score:.4f}")
        if show_prompt:
            print("\n--- prompt ---")
            print(result.prompt)
        else:
            print("\n（加 --show-prompt 查看完整 prompt；生产环境此处调用 LLM）")
    else:
        # 绝不把空/低质上下文交给 LLM
        print(f"\n直接返回：{result.fallback.message}")


if __name__ == "__main__":
    main()
