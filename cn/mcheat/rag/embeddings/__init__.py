"""可插拔 Embedding 实现。"""
from .hashing import HashingEmbedder

__all__ = ["HashingEmbedder", "LangChainEmbedder"]


def __getattr__(name: str):
    """延迟导入可选实现，避免未装 LangChain embeddings 时 import 本包就失败。"""
    if name == "LangChainEmbedder":
        from .adapters import LangChainEmbedder
        return LangChainEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
