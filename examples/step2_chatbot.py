"""Step 2：把 LLM 接进图 —— 最简单的「聊天机器人」节点。

Step 1 里节点是纯函数；这里把其中一个节点换成「调用 LLM 的函数」，
你就得到了一个单节点图：输入一句话，模型回一句话。

新知识点：
    * MessagesState —— LangGraph 内置、专门存对话消息的 state。
      它自带 messages 字段，并做了「消息追加」处理（不用自己管历史合并）。
    * 节点返回 {"messages": [AIMessage(...)]}，会被自动追加到对话历史。

运行前准备（.env 里配好 key，见 .env.example）：
    ANTHROPIC_API_KEY=...     或  OPENAI_API_KEY=...
    LLM_PROVIDER=anthropic    或  openai

运行：
    python examples/step2_chatbot.py
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from llm import get_model

model = get_model()


def chatbot(state: MessagesState) -> dict:
    """聊天节点：把历史消息喂给模型，返回一条新消息。"""
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()


if __name__ == "__main__":
    # 通过 system 消息给模型设定人设/行为约束
    messages = [
        SystemMessage(content="你是一个简洁的中文助手，回答不超过三句话。"),
        HumanMessage(content="什么是 LangGraph？"),
    ]
    result = graph.invoke({"messages": messages})
    for m in result["messages"]:
        m.pretty_print()
