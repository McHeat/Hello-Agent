"""Step 1：LangGraph 基础 —— 图（Graph）如何组织「状态」和「节点」。

一个 LangGraph 应用由三样东西组成：

    1. State（状态）：贯穿整张图的数据结构，节点之间靠它传递数据。
    2. Node（节点）：一个个函数，接收 State、返回对 State 的增量更新。
    3. Edge（边）：连接节点的「导线」，决定执行顺序；可无条件、也可按条件分支。

本示例不调用任何 LLM，用纯 Python 函数把「图」这个核心机制讲清楚，
确保你在接模型之前先看明白图是怎么转起来的。

运行（无需任何 API key）：
    python examples/step1_basic_graph.py
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# 1. 定义 State —— 一个 TypedDict，声明图里会流动哪些字段。
class State(TypedDict):
    number: int      # 输入的数字
    parity: str      # 奇偶性：even / odd
    result: str      # 最终描述


# 2. 定义节点 —— 每个节点是「接收 state、返回部分更新」的普通函数。
#    返回的 dict 会和原 state 浅合并（只覆盖你写出来的字段）。
def classify(state: State) -> dict:
    n = state["number"]
    return {"parity": "even" if n % 2 == 0 else "odd"}


def handle_even(state: State) -> dict:
    n = state["number"]
    return {"result": f"{n} 是偶数，翻倍后是 {n * 2}"}


def handle_odd(state: State) -> dict:
    n = state["number"]
    return {"result": f"{n} 是奇数，三倍后是 {n * 3}"}


# 3. 条件路由函数 —— 决定下一步去哪个节点。
#    返回值必须是某个节点名，或 END（这里直接返回 parity 的值）。
def route(state: State) -> str:
    return state["parity"]


# 4. 搭图：StateGraph -> add_node -> add_edge -> compile
builder = StateGraph(State)

builder.add_node("classify", classify)
builder.add_node("handle_even", handle_even)
builder.add_node("handle_odd", handle_odd)

builder.add_edge(START, "classify")            # 图入口：从 START 到第一个节点
builder.add_conditional_edges(                 # 条件分支：按 route() 的返回值分叉
    "classify",                                #   从哪个节点出发
    route,                                     #   路由函数
    {"even": "handle_even", "odd": "handle_odd"},  # 路由值 -> 目标节点名
)
builder.add_edge("handle_even", END)           # 图出口
builder.add_edge("handle_odd", END)

graph = builder.compile()                      # 编译成可执行对象


if __name__ == "__main__":
    # 5. 调用：invoke 同步执行整张图，返回最终 state。
    print("=== 偶数输入 ===")
    print(graph.invoke({"number": 4}))

    print("\n=== 奇数输入 ===")
    print(graph.invoke({"number": 7}))
