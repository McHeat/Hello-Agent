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

**query / passage 前缀**：BGE / E5 等非对称模型要求查询侧与文档侧使用
不同的输入前缀，否则中文检索质量会显著下降。本适配器按
``prefix_preset`` 或显式的 ``query_prefix`` / ``document_prefix`` 注入，
并把它纳入 :meth:`LangChainEmbedder.fingerprint`，避免换了前缀却复用了
旧索引。
"""
from __future__ import annotations

from typing import Any, Sequence

from .._utils import get_logger, stable_id
from ..core.interfaces import Embedder
from ..core.registry import EMBEDDER_REGISTRY
from ..exceptions import DependencyMissingError, ConfigError
from ..models import Vector

logger = get_logger(__name__)

# 常见非对称嵌入模型的 query / passage 前缀预设
PREFIX_PRESETS: dict[str, dict[str, str]] = {
    "none": {"query_prefix": "", "document_prefix": ""},
    "bge": {
        "query_prefix": "为这个句子生成表示以用于检索相关文章：",
        "document_prefix": "",
    },
    "e5": {"query_prefix": "query: ", "document_prefix": "passage: "},
}


def _infer_preset(model: str) -> str:
    """按模型名推断默认前缀预设；未知模型不加前缀。"""
    lowered = model.lower()
    if "bge" in lowered:
        return "bge"
    if "e5" in lowered:
        return "e5"
    return "none"


@EMBEDDER_REGISTRY.register("langchain")
class LangChainEmbedder(Embedder):
    """把 LangChain Embeddings 适配成本框架的 :class:`Embedder`。

    Args:
        embeddings: 已构造好的 LangChain Embeddings 对象（优先）。
        provider: 未提供 ``embeddings`` 时按此懒加载。
        model: 模型名（默认 ``BAAI/bge-m3``，中文优先）。
        query_prefix: 查询侧前缀；显式传入时覆盖预设。
        document_prefix: 文档侧前缀；显式传入时覆盖预设。
        prefix_preset: ``none`` / ``bge`` / ``e5``；默认按模型名推断。
        batch_size: 离线批量向量化提示（保留供适配器调用方对齐配置）。
        **kwargs: 传给底层 Embeddings 构造函数的额外参数。
    """

    def __init__(self, embeddings: Any | None = None,
                 provider: str = "huggingface",
                 model: str = "BAAI/bge-m3",
                 query_prefix: str | None = None,
                 document_prefix: str | None = None,
                 prefix_preset: str | None = None,
                 batch_size: int = 32,
                 **kwargs: Any) -> None:
        self._embeddings = embeddings or _build_embeddings(provider, model, kwargs)
        if not (hasattr(self._embeddings, "embed_documents")
                and hasattr(self._embeddings, "embed_query")):
            raise ConfigError(
                "embeddings 对象需实现 embed_documents / embed_query "
                "(LangChain Embeddings 接口)"
            )
        preset = prefix_preset or _infer_preset(model)
        if preset not in PREFIX_PRESETS:
            raise ConfigError(
                f"未知的 prefix_preset={preset!r}；可选：{sorted(PREFIX_PRESETS)}"
            )
        defaults = PREFIX_PRESETS[preset]
        self.prefix_preset = preset
        self.query_prefix = defaults["query_prefix"] if query_prefix is None \
            else str(query_prefix)
        self.document_prefix = defaults["document_prefix"] if document_prefix is None \
            else str(document_prefix)
        self.model = model
        self.provider = provider
        self.batch_size = batch_size
        # 探测维度：换模型后 vector store 会用这个维度做一致性校验
        self._dimension = len(self._embeddings.embed_query(
            self.query_prefix + "dimension probe"
        ))
        if self._dimension <= 0:
            raise ConfigError("Embeddings 返回了空向量，无法确定维度")
        logger.info("LangChainEmbedder 就绪：model=%s, preset=%s, dimension=%d",
                    model, preset, self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def fingerprint(self) -> str:
        """类名 + 模型名 + 维度 + 前缀配置的稳定哈希（同维度换模型也能拦截）。"""
        return stable_id(
            type(self).__name__, self.model, self._dimension,
            self.prefix_preset, self.query_prefix, self.document_prefix,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        prefixed = [self.document_prefix + text for text in texts]
        vectors = self._embeddings.embed_documents(prefixed)
        if len(vectors) != len(texts):
            raise ConfigError("Embeddings 返回数量与输入不一致")
        return [tuple(float(x) for x in vector) for vector in vectors]

    def embed_query(self, text: str) -> Vector:
        return tuple(float(x) for x in
                     self._embeddings.embed_query(self.query_prefix + text))


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


__all__: Sequence[str] = ["LangChainEmbedder", "PREFIX_PRESETS"]
