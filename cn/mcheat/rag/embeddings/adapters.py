"""可选适配器：把 LangChain 的 ``Embeddings`` 接进本框架。

只在 ``components.embedder.name = "langchain"`` 时才会 import 相关 SDK
（延迟导入），因此核心包在没装这些依赖时依然可以正常 import / 运行。

支持的 provider：

* ``"huggingface"`` → ``langchain_huggingface.HuggingFaceEmbeddings``
* ``"openai"``      → ``langchain_openai.OpenAIEmbeddings``
* ``"dashscope"``   → ``langchain_community.embeddings.DashScopeEmbeddings``

也可以直接通过 ``embeddings=`` 注入任何实现了
``embed_documents / embed_query`` 的对象（包括测试替身），此时完全不触发
第三方 import —— 这是最推荐的生产接法。
"""
from __future__ import annotations

from typing import Any, Sequence

from .._utils import get_logger
from ..core.interfaces import Embedder
from ..core.registry import EMBEDDER_REGISTRY
from ..exceptions import DependencyMissingError, ConfigError
from ..models import Vector

logger = get_logger(__name__)


@EMBEDDER_REGISTRY.register("langchain")
class LangChainEmbedder(Embedder):
    """把 LangChain Embeddings 适配成本框架的 :class:`Embedder`。

    Args:
        embeddings: 已构造好的 LangChain Embeddings 对象（优先）。
        provider: 未提供 ``embeddings`` 时按此懒加载。
        model: 模型名。
        **kwargs: 传给底层 Embeddings 构造函数的额外参数。
    """

    def __init__(self, embeddings: Any | None = None,
                 provider: str = "huggingface",
                 model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 **kwargs: Any) -> None:
        self._embeddings = embeddings or _build_embeddings(provider, model, kwargs)
        if not (hasattr(self._embeddings, "embed_documents")
                and hasattr(self._embeddings, "embed_query")):
            raise ConfigError(
                "embeddings 对象需实现 embed_documents / embed_query "
                "(LangChain Embeddings 接口)"
            )
        # 探测维度：换模型后 vector store 会用这个维度做一致性校验
        self._dimension = len(self._embeddings.embed_query("dimension probe"))
        if self._dimension <= 0:
            raise ConfigError("Embeddings 返回了空向量，无法确定维度")
        logger.info("LangChainEmbedder 就绪：dimension=%d", self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        vectors = self._embeddings.embed_documents(list(texts))
        if len(vectors) != len(texts):
            raise ConfigError("Embeddings 返回数量与输入不一致")
        return [tuple(float(x) for x in vector) for vector in vectors]

    def embed_query(self, text: str) -> Vector:
        return tuple(float(x) for x in self._embeddings.embed_query(text))


def _build_embeddings(provider: str, model: str, kwargs: dict[str, Any]) -> Any:
    """按 provider 延迟 import 并构造底层 Embeddings。"""
    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise DependencyMissingError(
                "langchain-huggingface sentence-transformers",
                "HuggingFace Embeddings",
            ) from exc
        return HuggingFaceEmbeddings(model_name=model, **kwargs)

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise DependencyMissingError("langchain-openai", "OpenAI Embeddings") from exc
        return OpenAIEmbeddings(model=model, **kwargs)

    if provider == "dashscope":
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
        except ImportError as exc:
            raise DependencyMissingError(
                "langchain-community dashscope", "DashScope Embeddings"
            ) from exc
        return DashScopeEmbeddings(model=model, **kwargs)

    raise ConfigError(
        f"未知的 embeddings provider={provider!r}；"
        f"可选 huggingface/openai/dashscope，或直接注入 embeddings 对象。"
    )


__all__: Sequence[str] = ["LangChainEmbedder"]
