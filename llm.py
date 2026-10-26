"""模型工厂：一处切换 LLM 提供商（生产写法）。

所有示例都通过 `get_model()` 拿模型，业务代码只依赖 LangChain 的
`BaseChatModel` 抽象，不 import 具体厂商的 SDK。这样切换
Anthropic / OpenAI 只需要改环境变量，代码一行不动。

控制方式（见 .env.example）：
    LLM_PROVIDER = anthropic | openai   # 默认 anthropic
    LLM_MODEL    = 具体模型名            # 不填则用下面的默认值

用法：
    from llm import get_model
    model = get_model()                     # 用 .env 配置的 provider/model
    model = get_model("claude-sonnet-5-5")  # 临时指定某个模型
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# 显式加载项目根目录下的 .env（无论从哪个目录运行示例都能读到）
load_dotenv(Path(__file__).resolve().parent / ".env")

# 各 provider 的默认模型，与厂商当前推荐保持一致
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",  # Anthropic 当前默认旗舰模型
    "openai": "gpt-4o",              # OpenAI 常用模型
}


def get_model(model: Optional[str] = None) -> BaseChatModel:
    """返回一个 ChatModel 实例。

    Args:
        model: 可选，临时覆盖模型名。不传则按 LLM_MODEL -> provider 默认值 取值。
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()

    model_name = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider)
    if not model_name:
        raise ValueError(
            f"找不到 provider={provider!r} 的默认模型，请在 .env 里设置 LLM_MODEL"
        )

    if provider == "anthropic":
        # 延迟导入：只装了某个 provider 的 SDK 也不会在加载时出错。
        from langchain_anthropic import ChatAnthropic

        # 注意：2026-10 起 Anthropic 系列模型不再接受 temperature/top_p 等采样参数，
        # 这里不传任何采样参数，交给模型自带的 adaptive thinking 自行决策。
        return ChatAnthropic(model=model_name)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=0)

    raise ValueError(f"不支持的 LLM_PROVIDER={provider!r}，可选：anthropic / openai")


if __name__ == "__main__":
    m = get_model()
    print("已加载模型：", m)
