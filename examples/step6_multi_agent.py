"""Step 6：多智能体 —— 一个「主管」调度多个「专家」（Supervisor 模式）。

这是 langchain-ai/langgraph-supervisor（~1.7k stars）背后的核心思想：
    一个主管(supervisor)负责理解任务并分派，多个专家(worker)各司其职。

本示例用手写图展示这个模式，让「分派」这件事看得见：
    supervisor(主管决定找谁) -> researcher / math -> 回到主管 -> 直到 FINISH

生产上可直接用 langgraph-supervisor 库的 create_supervisor 一行搞定，
这里手写是为了把原理摊开。

新知识点：
    * add_messages reducer：自定义 State 里要「追加消息」而非覆盖时用。
    * create_agent：langchain.agents 提供的高层封装，一个函数就是一个子智能体，
      可以直接当成图里的节点来用。

运行：
    python examples/step6_multi_agent.py
"""
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from llm import get_model

model = get_model()


# ---- 专家 1：数学专家，拥有计算器工具 ----
@tool
def multiply(a: float, b: float) -> float:
    """两个数相乘。"""
    return a * b


math_agent = create_agent(
    model, tools=[multiply], system_prompt="你是数学专家，用计算器工具解答数学问题。"
)


# ---- 专家 2：研究员，拥有一个「知识库」检索工具（模拟 RAG）----
KNOWLEDGE = {
    "langgraph": "LangGraph 是 LangChain 推出的、用于构建有状态多智能体应用的图编排框架。",
    "react": "ReAct 是一种让模型交替进行推理(Reasoning)和行动(Acting)的范式。",
}


@tool
def lookup(topic: str) -> str:
    """在内部知识库中查找某个主题的介绍。topic 可选：langgraph / react。"""
    return KNOWLEDGE.get(topic.lower(), "知识库里没有这个主题。")


research_agent = create_agent(
    model, tools=[lookup], system_prompt="你是研究员，用 lookup 工具查资料回答问题。"
)


# ---- 主管状态：messages 累积整场对话，next 记录下一步找谁 ----
class SupervisorState(TypedDict):
    # add_messages 让多个节点返回的 messages 自动追加，而不是互相覆盖
    messages: Annotated[list, add_messages]
    next: str


# 主管节点：让模型根据当前对话决定下一步交给谁。
def supervisor_node(state: SupervisorState) -> dict:
    system = SystemMessage(content=(
        "你是主管。根据用户问题决定下一步交给谁：\n"
        "- 需要查资料/概念解释 -> researcher\n"
        "- 需要数学计算 -> math\n"
        "- 信息已足够，可以给出最终答案 -> FINISH\n"
        "只回复 researcher、math 或 FINISH，不要解释。"
    ))
    decision = model.invoke([system] + state["messages"]).content.strip().lower()

    if "finish" in decision:
        nxt = "FINISH"
    elif "math" in decision:
        nxt = "math"
    else:
        nxt = "researcher"
    return {"next": nxt}


# 专家节点：把整段对话交给对应子智能体，只把它的最后一条回复追加回 messages。
def research_node(state: SupervisorState) -> dict:
    result = research_agent.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


def math_node(state: SupervisorState) -> dict:
    result = math_agent.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


# 主管的收尾节点：基于前面专家给的结论，输出最终答案。
def final_node(state: SupervisorState) -> dict:
    answer = model.invoke(
        [SystemMessage(content="你是主管，基于前面的对话给用户一个完整最终答案。")]
        + state["messages"]
    )
    return {"messages": [answer]}


def route_after_supervisor(state: SupervisorState) -> str:
    return state["next"]


def route_after_worker(state: SupervisorState) -> str:
    return "supervisor"  # 专家干完活，永远回到主管再决策


builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_node)
builder.add_node("math", math_node)
builder.add_node("final", final_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {"researcher": "researcher", "math": "math", "FINISH": "final"},
)
builder.add_edge("researcher", "supervisor")
builder.add_edge("math", "supervisor")
builder.add_edge("final", END)

graph = builder.compile()


if __name__ == "__main__":
    # 一个既需要查资料、又需要计算的复合问题，看主管如何分派。
    question = "LangGraph 是什么？另外帮我算一下 7 乘以 8。"
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    print("=== 最终答案 ===")
    print(result["messages"][-1].content)
