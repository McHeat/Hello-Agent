"""Golden Set 回归评测 + CI 指标门禁。

运行::

    python -m cn.mcheat.rag.examples.run_evaluation                 # 评测并打印报告
    python -m cn.mcheat.rag.examples.run_evaluation --save-baseline # 保存为基线
    python -m cn.mcheat.rag.examples.run_evaluation --gate          # 与基线比对，失败退出码 1

``--gate`` 会把 ``context_recall / context_precision / average_precision /
hit_rate`` 与基线逐项比较，跌幅超过 ``evaluation.max_regression_drop`` 就返回
非零退出码，可直接接入 CI —— 这就是「别拍脑袋调参」的工程闭环。
"""
from __future__ import annotations

import sys
from pathlib import Path

from cn.mcheat.rag import RAGSystem
from cn.mcheat.rag.evaluation import GoldenSet

HERE = Path(__file__).resolve().parent
GOLDEN_SET = HERE / "golden_set.json"
BASELINE = HERE / "baseline.json"
DATA_DIR = HERE / "data"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    save_baseline = "--save-baseline" in args
    gate = "--gate" in args

    system = RAGSystem()
    system.index([DATA_DIR])
    golden = GoldenSet.from_json(GOLDEN_SET)

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
