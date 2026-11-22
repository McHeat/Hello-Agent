"""CLI 多轮交互演示：Python 学习 Agent 的完整 human-in-the-loop 闭环。

从**项目根目录**运行（保证 ``cn.mcheat.agent`` 与根目录 ``llm`` 都可导入）::

    python -m cn.mcheat.agent.examples.cli
    python -m cn.mcheat.agent.examples.cli --thread-id user-42
    python -m cn.mcheat.agent.examples.cli --config my_agent.json

需要 ``.env`` 里配置好 API key（模型经根目录 ``llm.py`` 工厂注入，
切换 Anthropic / OpenAI / DeepSeek 等只改 .env，代码零改动）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从任意目录运行都能 import 根目录的 llm.py（项目根 = 本文件向上四级）
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from cn.mcheat.agent import (  # noqa: E402
    AgentAnswer,
    ClarifyInterrupt,
    GraphError,
    LlmError,
    PythonTutorAgent,
)

BANNER = """\
==============================================
 Python 学习 Agent（LangGraph 生产级框架）
----------------------------------------------
 我可以：
   1. 讲解 Python 语法（如「列表推导式怎么写」）
   2. 检测你的 Python 安装环境（版本 / pip / 虚拟环境）
   3. 对你的代码给出修改指导（直接粘贴代码）
 输入 exit / quit 退出；同一会话共享多轮记忆。
=============================================="""


def build_agent(config_path: str | None) -> PythonTutorAgent:
    """装配 Agent：模型用根目录 llm.py 工厂，配置可选用户 JSON。"""
    from llm import get_model

    from cn.mcheat.agent import AgentConfig

    config = AgentConfig.load(path=config_path) if config_path else None
    return PythonTutorAgent(config=config, llm=get_model())


def main() -> int:
    parser = argparse.ArgumentParser(description="Python 学习 Agent CLI")
    parser.add_argument("--thread-id", default="cli-session",
                        help="会话线程 ID（同一 ID 共享多轮记忆）")
    parser.add_argument("--config", default=None,
                        help="可选的用户 JSON 配置路径（覆盖 defaults.json）")
    args = parser.parse_args()

    print(BANNER)
    try:
        agent = build_agent(args.config)
    except Exception as exc:  # 装配期错误（key 缺失 / 配置非法）给出可读提示
        print(f"\n[启动失败] {exc}")
        return 1

    while True:
        try:
            question = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("再见！")
            return 0

        try:
            outcome = agent.ask(question, thread_id=args.thread_id)

            # human-in-the-loop：意图不明确时向用户发起澄清，确认后继续
            while isinstance(outcome, ClarifyInterrupt):
                print(f"\n🤖 {outcome.question}")
                for candidate in outcome.candidates:
                    print(f"   {candidate}")
                try:
                    reply = input("你> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n（会话被中断，本轮放弃）")
                    break
                outcome = agent.resume(reply, thread_id=args.thread_id)

            if isinstance(outcome, AgentAnswer):
                print(f"\n🤖 {outcome.content}")
        except LlmError as exc:
            print(f"\n[LLM 错误] {exc}")
        except GraphError as exc:
            print(f"\n[图执行错误] {exc}")
        except Exception as exc:  # 兜底：不让单个异常杀掉整个 CLI 进程
            print(f"\n[未知错误] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
