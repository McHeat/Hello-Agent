"""
step9_callbacks.py
演示 LangChain 的回调机制（Callback）：在链运行的不同生命周期节点插入自定义逻辑。

核心概念：
- Callback Handler 是一组"钩子"方法（on_chain_start / on_chain_end / on_llm_start ...），
  链式调用会在对应时刻自动触发这些方法，用于打日志、埋点、监控等。
- 回调通过 config 在调用时注入，不污染模型本身。
"""

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

from llm import get_model

# 构造提示词模板：用 {age} 占位符接收运行时入参。
# 注意：from_messages 必须接收"消息列表"，每个元素是 (角色, 内容) 元组。
prompt = ChatPromptTemplate.from_messages([("human", "给我一个适宜{age}儿童的游戏建议")])


class CustomRequestCallBackHandler(BaseCallbackHandler):
    """自定义回调处理器：在链开始和结束时打印提示。"""

    # on_chain_start：链（prompt | model）刚开始执行时触发
    def on_chain_start(
        self,
        serialized: dict[str, Any],          # 链的序列化信息
        inputs: dict[str, Any],              # 本次调用的输入参数，如 {"age": "2岁"}
        *,
        run_id: UUID,                         # 本次运行的唯一 ID
        parent_run_id: UUID | None = None,    # 父级运行 ID（嵌套链时有值）
        tags: list[str] | None = None,        # 调用时传入的标签
        metadata: dict[str, Any] | None = None,  # 调用时传入的元数据
        **kwargs: Any,                        # 预留扩展字段
    ) -> Any:
        print("请求回调：开始运行")

    # on_chain_end：链执行完毕、拿到最终响应后触发
    def on_chain_end(
        self,
        response: Any,                        # 链的最终输出
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        print("请求回调：结束运行")


def construct_test():
    """构建"提示词 | 模型"链，并注入回调后执行一次调用。"""
    # 实例化回调处理器（注意是类的实例，而不是类本身）
    handler = CustomRequestCallBackHandler()

    model = get_model()
    # 用管道符把提示词模板和模型串成一条链：
    # 先渲染 prompt，再把渲染结果喂给模型。
    chain = prompt | model

    # 通过 config 在调用时注入回调，而不是修改模型属性。
    result = chain.invoke(
        {"age": "2岁"},
        config={"callbacks": [handler]},
    )
    result.pretty_print()


if __name__ == "__main__":
    construct_test()
