"""一站式演示：离线索引 → 在线检索 → 5 条注意事项逐项验证 → 回归评测。

运行（必须从项目根目录，保证 ``cn.mcheat.rag`` 可导入）::

    python -m cn.mcheat.rag.examples.demo

核心链路零第三方依赖、不需要任何 API key，直接可跑。
"""
from __future__ import annotations

import logging
from pathlib import Path

from cn.mcheat.rag import RAGSystem

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.json"


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    # 打开 INFO 日志可以看到每个阶段的诊断（生产环境由应用侧配置 handler）
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    system = RAGSystem()
    _section("离线索引：5 种 Loader + 清洗 + Small-to-Big 父子切分")
    stats = system.index([DATA_DIR])
    for key, value in stats.as_dict().items():
        if key != "skipped":
            print(f"  {key:>22}: {value}")
    print("  清洗步骤:", system.cleaning_pipeline.steps)
    print("  已注册 Loader:", [loader.name for loader in system.loader_registry.loaders])

    query = "父子切分 Small-to-Big 是怎么解决 chunk 大小两难的"
    result = system.retrieve(query)

    _section("注意事项 1：粗召回 30~50 → 精排 3~5，只有精排结果进 prompt")
    print(f"  粗召回候选数 : {len(result.coarse)}  (coarse_top_k={system.config.retrieval.coarse_top_k})")
    print(f"  精排后进入 prompt : {len(result.final)}  (rerank_top_n={system.config.retrieval.rerank_top_n})")
    print(f"  粗召回通路 : {result.diagnostics['recall']['per_path']}")

    _section("注意事项 2：Small-to-Big，用小片段检索、返回大段落")
    for item in result.final[:2]:
        child_len = len(item.chunk.text)
        parent_len = len(item.parent.text) if item.parent else 0
        print(f"  命中子块 {item.chunk.id[:14]}… 长度={child_len}")
        print(f"    → 还原父块 {item.chunk.parent_id[:14]}… 长度={parent_len}"
              f"（父块子串校验：{item.parent.text[item.chunk.start:item.chunk.end] == item.chunk.text if item.parent else 'N/A'}）")

    _section("注意事项 3：Lost in the Middle，高相关片段放首/尾")
    print("  展示位置 -> 相关性名次:",
          [(p.position, p.relevance_rank) for p in result.context.placements])
    print("  提示词末尾（近因位置）:")
    for line in result.prompt.splitlines()[-3:]:
        print("   ", line)

    _section("注意事项 5：兜底链路（空检索 → 扩 Top-K → 转人工 / 如实回答）")
    empty = RAGSystem()  # 全新系统、索引为空
    fallback_result = empty.retrieve("任何问题")
    print(f"  最终动作 : {fallback_result.fallback.final_action}")
    print(f"  触发原因 : {fallback_result.fallback.triggers}")
    print(f"  降级轨迹 : {[s.strategy for s in fallback_result.fallback.steps]}")
    print(f"  重试记录 : {fallback_result.diagnostics['retry_history']}")
    print(f"  回答内容 : {fallback_result.fallback.message}")
    print(f"  是否禁止调用 LLM : {not fallback_result.answer_ready}")

    _section("注意事项 4：Golden Set 回归，Recall / Precision 分阶段量化")
    report = system.evaluate(GOLDEN_SET)
    print(f"  Golden Set: {report.golden_set}，{len(report.cases)} 条标注")
    for metric, value in report.summary.items():
        print(f"  {metric:>22}: {value}")
    print("  说明：context_recall 在粗召回阶段算，context_precision 在精排后算。")
    print("  示例语料很小，粗召回会覆盖全部子块，Recall 恒为 1；真实语料才有区分度。")

    _section("索引持久化：离线写 / 在线读 可分进程")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = system.save(tmp)
        restored = RAGSystem.load(tmp)
        check = restored.retrieve(query)
        print(f"  保存到 {path.name}，恢复后 final={len(check.final)}，"
              f"动作={check.fallback.final_action}")

    print("\n演示结束。")


if __name__ == "__main__":
    main()
