"""Step 5：人工介入 —— 用 interrupt 在关键节点停下来等人批准。

生产中很多动作不能全自动（发邮件、转账、发布、部署），需要执行前暂停、
让人类审批后再继续。LangGraph 的 interrupt() 就是干这个的。

关键点：
    * interrupt(payload)：暂停执行，把 payload 返回给调用方。
    * graph.invoke(...) 会在暂停点返回 {"__interrupt__": [...]}。
    * 用 Command(resume=...) 带着人类的决定继续跑，图会从暂停处接着走。

图结构（模拟「AI 写稿 -> 人工审稿 -> 发布/打回」）：
    draft(写稿) -> review(interrupt 暂停) -> publish 或 reject

运行：
    python examples/step5_human_in_the_loop.py
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from llm import get_model

model = get_model()


class State(TypedDict):
    draft: str      # 初稿
    approved: bool  # 是否通过


def draft_node(state: State) -> dict:
    """让模型写一句初稿。"""
    msg = model.invoke("写一句关于学习 AI 的励志口号（15 字以内，只给口号本身）。")
    return {"draft": msg.content}


def review_node(state: State) -> dict:
    """人工审稿：interrupt 会在这里暂停，把草稿交给人类看。"""
    decision = interrupt(
        {"draft": state["draft"], "question": "这稿子能发布吗？回复 y 或 n"}
    )
    # 人类用 Command(resume=...) 回复后，这里会收到那个字符串并继续往下走
    return {"approved": str(decision).strip().lower().startswith("y")}


def publish(state: State) -> dict:
    return {"draft": f"✅ 已发布：{state['draft']}"}


def reject(state: State) -> dict:
    return {"draft": "❌ 已打回重写"}


def route(state: State) -> str:
    return "publish" if state["approved"] else "reject"


builder = StateGraph(State)
builder.add_node("draft", draft_node)
builder.add_node("review", review_node)
builder.add_node("publish", publish)
builder.add_node("reject", reject)
builder.add_edge(START, "draft")
builder.add_edge("draft", "review")
builder.add_conditional_edges("review", route, {"publish": "publish", "reject": "reject"})
builder.add_edge("publish", END)
builder.add_edge("reject", END)

graph = builder.compile()


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "review-001"}}

    # 第一次 invoke：跑到 review 节点时 interrupt 暂停，返回暂停信息
    paused = graph.invoke({}, config=config)
    print("=== 已暂停，等待人工审批 ===")
    print(paused)

    # 人类看了草稿后决定：批准发布
    result = graph.invoke(Command(resume="y"), config=config)
    print("\n=== 继续执行后的最终结果 ===")
    print(result)
