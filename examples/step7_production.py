"""Step 7：生产化 —— create_agent + 流式输出 + 运行配置。

前面几步用「手写图」讲原理；生产上推荐用 langchain.agents.create_agent。
它内部就是一张 LangGraph 图，但帮你处理好了工具循环、错误、中间件等细节。

本示例展示三件生产必备的事：
    1. create_agent：一行搭起一个带工具的 agent（且直接支持 checkpointer）。
    2. 流式输出：stream 边生成边打印，用户体验好，也避免长响应超时。
    3. 运行配置：recursion_limit 防死循环、thread_id 做会话隔离。

运行：
    python examples/step7_production.py
"""
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from llm import get_model

model = get_model()


@tool
def add(a: float, b: float) -> float:
    """两个数相加。"""
    return a + b


# 一行创建 agent：内部是 LangGraph 图，.invoke / .stream 都可直接调用。
agent = create_agent(
    model,
    tools=[add],
    system_prompt="你是一个精于计算的助手。",
    checkpointer=MemorySaver(),  # 直接传 checkpointer 即可获得记忆
)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "prod-001"}}
    question = "计算 3 加 4 等于多少？"

    print("=== 流式输出（stream_mode='updates'，每个节点完成吐一个 chunk）===")
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="updates",
    ):
        print(chunk)

    print("\n=== 最终结果 ===")
    final = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        # 防止 agent 在工具调用里死循环的上限
        recursion_limit=10,
    )
    final["messages"][-1].pretty_print()
