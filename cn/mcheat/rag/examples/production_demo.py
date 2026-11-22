"""中文生产配置档演示（缺可选依赖时优雅跳过）。

运行::

    python -m cn.mcheat.rag.examples.production_demo
    # 或设置 RAG_PROFILE=production 后直接构造 RAGSystem()

生产档使用 ``BAAI/bge-m3``（query/passage 前缀）+ ``BAAI/bge-reranker-large``
+ SQLite 父块 / 向量库。切换后**必须重建索引并重跑基线**：指纹校验会拦截
旧索引；阈值当前为暂行值，须按方案 9.4 用真实生产栈单独复标。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from cn.mcheat.rag import RAGSystem, load_profile
from cn.mcheat.rag.exceptions import DependencyMissingError

SAMPLE_TEXT = (
    "中文检索增强生成系统使用 BGE 嵌入模型与交叉编码器精排。"
    "查询侧与文档侧需要分别添加前缀，否则中文语义检索质量会下降。"
    "索引产物会记录模型指纹与版本号，换模型时必须重建索引。\n"
)


def main() -> int:
    print("生产配置档：configs/production.json（中文 BGE + SQLite）")
    try:
        config = load_profile("production", use_env=False)
        system = RAGSystem(config=config)
    except DependencyMissingError as exc:
        print("跳过生产档演示：缺少可选依赖（默认档仍零依赖可跑）。")
        print(f"  {exc}")
        print("  安装 langchain-huggingface / sentence-transformers 后，"
              "用真实 BGE 模型重建索引并执行方案 9.4 阈值复标。")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus.txt"
        corpus.write_text(SAMPLE_TEXT, encoding="utf-8")
        stats = system.index([corpus], mode="full")
        print("  索引统计：", stats.as_dict())
        result = system.retrieve("中文检索为什么需要查询前缀")
        print("  归一化分：", [round(item.score, 4) for item in result.final])
        print("  是否可回答：", result.answer_ready)
        artifact = system.save(Path(tmp) / "index")
        print("  索引产物：", artifact.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
