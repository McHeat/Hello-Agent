"""Golden Set 回归评测 + 阈值扫描（--sweep-threshold）+ CI 指标门禁。

运行::

    python -m cn.mcheat.rag.examples.run_evaluation                  # 评测并打印报告
    python -m cn.mcheat.rag.examples.run_evaluation --save-baseline  # 保存为基线
    python -m cn.mcheat.rag.examples.run_evaluation --gate           # 与基线比对，失败退出码 1
    python -m cn.mcheat.rag.examples.run_evaluation --sweep-threshold  # 0.0~0.8 阈值标定
    python -m cn.mcheat.rag.examples.run_evaluation --answer-eval --answers answers.json

``--sweep-threshold`` 输出「诚实拒绝率 vs 误杀率」曲线：以 0.35 为待验证
起点，在含负样本的 Golden Set 上扫描 0.0~0.8，硬约束是「既有正样本零误杀」，
据此选取误接受率最低的阈值写入 ``configs/defaults.json``。该阈值仅适用于
默认栈，生产栈须按方案 9.4 单独复标。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from cn.mcheat.rag import RAGSystem
from cn.mcheat.rag.evaluation import (
    GoldenSet,
    false_accept_rate,
    false_refusal_rate,
    refusal_accuracy,
)

HERE = Path(__file__).resolve().parent
GOLDEN_SET = HERE / "golden_set.json"
BASELINE = HERE / "baseline.json"
DATA_DIR = HERE / "data"

SWEEP_LOW = 0.0
SWEEP_HIGH = 0.8
SWEEP_STEP = 0.05


def _option(args: list[str], name: str) -> str | None:
    """读取 ``--name value`` 形式的简单 CLI 选项。"""
    if name in args:
        index = args.index(name)
        if index + 1 < len(args):
            return args[index + 1]
    return None


def _sweep(system: RAGSystem, golden: GoldenSet) -> None:
    """扫描 min_best_score，输出误杀 / 误接受 / 诚实拒绝 / 拒答准确率。"""
    original = system.config.retrieval.min_best_score
    print(f"\n阈值扫描（{SWEEP_LOW:.2f}~{SWEEP_HIGH:.2f}，步长 {SWEEP_STEP:.2f}）：")
    print(f"{'阈值':>8} {'误杀率':>10} {'误接受率':>10} "
          f"{'诚实拒绝率':>12} {'拒答准确率':>12}")
    rows: list[tuple[float, float, float, float, float]] = []
    steps = int(round((SWEEP_HIGH - SWEEP_LOW) / SWEEP_STEP)) + 1
    for index in range(steps):
        threshold = round(SWEEP_LOW + index * SWEEP_STEP, 4)
        system.config.retrieval.min_best_score = threshold
        outcomes = []
        for case in golden:
            result = system.retrieve(case.query)
            outcomes.append((case.should_refuse, not result.answer_ready))
        false_refusal = false_refusal_rate(outcomes)
        false_accept = false_accept_rate(outcomes)
        honest_refusal = 1.0 - false_accept
        accuracy = refusal_accuracy(outcomes)
        rows.append((threshold, false_refusal, false_accept,
                     honest_refusal, accuracy))
        print(f"{threshold:>8.2f} {false_refusal:>10.4f} {false_accept:>10.4f} "
              f"{honest_refusal:>12.4f} {accuracy:>12.4f}")

    # 硬约束：正样本零误杀；在此前提下取误接受率最低（同分取更保守的高阈值）
    candidates = [row for row in rows if row[1] == 0.0]
    if candidates:
        best = min(candidates, key=lambda row: (row[2], -row[0]))
        print(f"\n推荐阈值：min_best_score = {best[0]:.2f}"
              f"（正样本零误杀，误接受率={best[2]:.4f}）")
        print("该值为暂行值：Golden Set 扩至 30~50 条后须复标并更新 --save-baseline 基线。")
    else:
        print("\n未找到满足「正样本零误杀」的阈值，请检查语料 / 精排分数域。")
    system.config.retrieval.min_best_score = original


def _answer_eval(system: RAGSystem, golden: GoldenSet,
                 answers_path: str | None) -> int:
    """答案级评测入口：接收调用方生成的答案，框架不代跑 LLM。"""
    if not answers_path:
        print("答案级评测需要 --answers <JSON 文件>（query -> 调用方生成的答案）")
        return 2
    path = Path(answers_path)
    if not path.is_file():
        print(f"答案文件不存在：{path}")
        return 2
    answers = json.loads(path.read_text(encoding="utf-8"))
    report = system.evaluate(golden, answers=answers)
    print("\n答案级评测（Faithfulness / Answer Relevancy）：")
    for metric, value in report.summary.items():
        print(f"  {metric:>22}: {value:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    save_baseline = "--save-baseline" in args
    gate = "--gate" in args
    sweep = "--sweep-threshold" in args
    answer_eval = "--answer-eval" in args

    system = RAGSystem()
    system.index([DATA_DIR])
    golden = GoldenSet.from_json(GOLDEN_SET)

    if sweep:
        _sweep(system, golden)
    if answer_eval:
        return _answer_eval(system, golden, _option(args, "--answers"))

    if gate and not BASELINE.is_file():
        print(f"基线不存在：{BASELINE}，请先运行 --save-baseline")
        return 2

    if gate:
        report, verdict = system.evaluate_with_gate(golden, BASELINE)
    else:
        report = system.evaluate(golden)
        verdict = None

    print(f"\nGolden Set: {report.golden_set}（{len(report.cases)} 条）")
    print("分阶段指标（Recall=粗召回，Precision=精排后进 prompt）：")
    for metric, value in report.summary.items():
        print(f"  {metric:>22}: {value:.4f}")

    if save_baseline:
        report.save(BASELINE)
        print(f"\n基线已保存：{BASELINE}")

    if verdict is not None:
        print(f"\n回归门禁：{'通过' if verdict.passed else '失败'}")
        print("  指标变化：", verdict.deltas)
        print(" ", verdict.message)
        return 0 if verdict.passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
