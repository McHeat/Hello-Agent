"""查询变换层（P1-1）：改写 / 多查询扩展 / HyDE 的可插拔入口。

边界：框架**不接真实 LLM**。LLM 能力只通过 ``QueryRewriter`` Protocol
注入，默认链路提供零依赖的规则型实现与 ``Identity`` 消融基线；任何失败
都按 ``FallbackConfig.on_transform_error`` 降级为原查询，保证零依赖默认档
可跑。多查询变体分别粗召回后复用 ``FusionStrategy`` 融合，精排与 prompt
始终使用主查询以控制交叉编码器成本。
"""
from __future__ import annotations

import re
from typing import Sequence

from .._utils import get_logger
from ..core.interfaces import QueryRewriter, QueryTransformer
from ..core.registry import QUERY_TRANSFORMER_REGISTRY
from ..exceptions import ConfigError, QueryTransformError

logger = get_logger(__name__)

_STRIP_RE = re.compile(r"[\s?？!！，,。、；;：:]+")


@QUERY_TRANSFORMER_REGISTRY.register("identity")
class IdentityQueryTransformer(QueryTransformer):
    """零变换：返回 ``[原查询]``，用于等价性与消融对照。"""

    def transform(self, query: str) -> list[str]:
        if not query or not query.strip():
            raise QueryTransformError("查询变换的输入不能为空")
        return [query]


@QUERY_TRANSFORMER_REGISTRY.register("rule_based")
class RuleBasedQueryTransformer(QueryTransformer):
    """规则型零依赖变换：去掉标点 / 空白，生成紧凑变体。

    变体只影响**粗召回**（由编排层融合），精排与 prompt 仍用主查询，
    因此开启后不会出现「主查询语义被改写」导致的负收益。
    """

    def transform(self, query: str) -> list[str]:
        if not query or not query.strip():
            raise QueryTransformError("查询变换的输入不能为空")
        cleaned = _STRIP_RE.sub(" ", query).strip()
        compact = cleaned.replace(" ", "")
        variants = [query]
        for candidate in (cleaned, compact):
            if candidate and candidate != query and candidate not in variants:
                variants.append(candidate)
        return variants


@QUERY_TRANSFORMER_REGISTRY.register("llm")
class LLMQueryTransformer(QueryTransformer):
    """仅按 Protocol 注入改写器，不绑定厂商、不要求 API key。

    Args:
        rewriter: 实现 ``rewrite(query) -> str`` 的对象；未注入时降级为原查询。
        max_variants: 返回的变体数上限。
    """

    def __init__(self, rewriter: QueryRewriter | None = None,
                 max_variants: int = 3) -> None:
        if max_variants <= 0:
            raise ConfigError("max_variants 必须为正整数")
        self.rewriter = rewriter
        self.max_variants = max_variants

    def transform(self, query: str) -> list[str]:
        if not query or not query.strip():
            raise QueryTransformError("查询变换的输入不能为空")
        if self.rewriter is None:
            logger.warning("LLMQueryTransformer 未注入 rewriter，规则型兜底为原查询")
            return [query]
        try:
            rewritten = str(self.rewriter.rewrite(query)).strip()
        except Exception as exc:
            raise QueryTransformError(
                f"查询改写失败：{type(exc).__name__}: {exc}"
            ) from exc
        variants = [query]
        if rewritten and rewritten != query:
            variants.append(rewritten)
        return variants[:self.max_variants]


class QueryTransformPipeline(QueryTransformer):
    """按序组合多个变换器，保证变体非空、主查询恒为首个元素。

    Args:
        transformers: 变换器链。
        max_variants: 变体数上限（来自 ``RetrievalConfig.max_query_variants``）。
        on_error: ``use_original`` 降级为 ``[原查询]``；``raise`` 抛
            :class:`QueryTransformError`。
    """

    def __init__(self, transformers: Sequence[QueryTransformer],
                 max_variants: int = 3,
                 on_error: str = "use_original") -> None:
        if on_error not in {"use_original", "raise"}:
            raise ConfigError("on_error 只能是 'use_original' 或 'raise'")
        if max_variants <= 0:
            raise ConfigError("max_variants 必须为正整数")
        self.transformers = list(transformers)
        self.max_variants = max_variants
        self.on_error = on_error

    def transform(self, query: str) -> list[str]:
        try:
            variants = [query]
            for transformer in self.transformers:
                produced = transformer.transform(variants[0])
                if not isinstance(produced, (list, tuple)) or not produced:
                    raise QueryTransformError(
                        f"{transformer.name} 返回了空的查询变体列表"
                    )
                for variant in produced:
                    text = str(variant).strip()
                    if not text or text in variants:
                        continue
                    if len(variants) >= self.max_variants:
                        break
                    variants.append(text)
            return variants
        except Exception as exc:
            if self.on_error == "raise":
                if isinstance(exc, QueryTransformError):
                    raise
                raise QueryTransformError(
                    f"查询变换失败：{type(exc).__name__}: {exc}"
                ) from exc
            logger.warning(
                "查询变换失败，按 on_error=%s 降级为原查询：%s",
                self.on_error, exc,
            )
            return [query]


__all__: Sequence[str] = [
    "IdentityQueryTransformer", "RuleBasedQueryTransformer",
    "LLMQueryTransformer", "QueryTransformPipeline",
]
