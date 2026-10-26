"""Step 4：记忆 —— 用 Checkpointer 让智能体记住多轮对话。

默认情况下每次 invoke 都是「失忆」的：图不会记住上次的状态。
给图加上 checkpointer 后，每次执行的状态都会被持久化；
用同一个 thread_id 再 invoke，就能接着上次的对话继续。

关键点：
    * MemorySaver：内存版检查点（进程重启即丢失）。生产可换 SQLite/Postgres。
    * config={"configurable": {"thread_id": "..."}} 用来区分不同会话。

运行：
    python examples/step4_memory.py
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from llm import get_model

model = get_model()


def chatbot(state: MessagesState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)  # 关键：传入 checkpointer


if __name__ == "__main__":
    # 同一个 thread_id 视为同一个会话；换个 id 就是另一段独立对话。
    config = {"configurable": {"thread_id": "user-001"}}

    # 第一轮：告诉它一个信息
    graph.invoke(
        {"messages": [HumanMessage(content="我叫小明，是一名学生。")]},
        config=config,
    )

    # 第二轮：问一个「必须记住上一轮」才能回答的问题
    result = graph.invoke(
        {"messages": [HumanMessage(content="我叫什么名字？")]},
        config=config,
    )
    for m in result["messages"]:
        m.pretty_print()
