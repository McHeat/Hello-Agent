"""Step 3：工具调用 —— 让智能体「动手」做事（ReAct 循环）。

这是 agent 的核心：模型不再只是聊天，还能调用工具（函数）获取信息或执行动作，
把结果带回模型，循环往复，直到模型认为可以给出最终答案。
这就是经典的 Reasoning（推理）+ Acting（行动）= ReAct 循环。

图结构：
    START -> agent（模型决定：直接回答 / 调用工具）
         -> 条件分支：有 tool_calls 就去 tools 节点，否则 END
    tools 节点执行完 -> 回到 agent，形成循环

运行：
    python examples/step3_tool_calling.py
"""
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

from llm import get_model


# 1. 定义工具：用 @tool 装饰普通函数即可。
#    函数的 docstring 就是给模型看的「使用说明书」，务必写清楚何时用、参数是什么。
@tool
def multiply(a: float, b: float) -> float:
    """两个数相乘。当需要计算乘法时使用。"""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """两个数相加。当需要计算加法时使用。"""
    return a + b


tools = [multiply, add]
# bind_tools：把工具列表「挂」到模型上，模型就知道自己有哪些工具可以调用。
model = get_model().bind_tools(tools)


# 2. agent 节点：把历史消息喂给模型。
#    返回的 AIMessage 里要么是文本答案，要么是 tool_calls（想调用哪些工具）。
def agent_node(state: MessagesState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


# 3. tools 节点：真正执行模型请求的工具调用，把结果打包成 ToolMessage 返回。
#    （生产上可用 langgraph.prebuilt.ToolNode 一行替代，这里手写是为了看清原理。）
def tools_node(state: MessagesState) -> dict:
    tool_calls = state["messages"][-1].tool_calls  # 最后一条消息里的工具调用
    name_to_fn = {t.name: t for t in tools}
    results = []
    for call in tool_calls:
        fn = name_to_fn[call["name"]]
        output = fn.invoke(call["args"])           # args 已经是解析好的 dict
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}


# 4. 路由：模型还要继续调工具吗？
def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")   # 工具执行完回到 agent，形成循环
graph = builder.compile()


if __name__ == "__main__":
    # 这个问题需要先算乘法、再算加法，会触发两轮工具调用。
    question = "先算 12 乘以 3，然后再加上 5，结果是多少？"
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    for m in result["messages"]:
        m.pretty_print()
