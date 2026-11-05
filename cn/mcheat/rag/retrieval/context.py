"""Small-to-Big 还原 + Lost-in-the-Middle 上下文编排 + Prompt 组装。

落实《rag编码注意事项》第 2、3 条：

* **第 2 条（chunk 大小两难）**：:class:`SmallToBigExpander` 把命中的子块
  还原成所在的大段落（父块）作为最终上下文，并按父块去重，避免同一段
  被多个子块重复塞进 prompt。
* **第 3 条（Lost in the Middle）**：:class:`EdgePlacementContextBuilder`
  默认使用 ``"edges"`` 策略，把最相关的片段放在上下文**开头与结尾**，
  最不相关的放在中间，形成 U 形相关性分布；同时在 prompt 末尾重复问题
  （近因效应更强），并用 ``max_context_chars`` 控制总长度，优先砍掉中间
  的低相关片段，而不是截断头尾。

``placement="sequential"`` 是消融基线（按相关性从头排到尾），可用 Golden Set
量化对比两种编排的实际收益。
"""
from __future__ import annotations

from typing import Sequence

from .._utils import get_logger
from ..config import PromptConfig
from ..core.interfaces import ContextBuilder, ParentStore
from ..core.registry import CONTEXT_BUILDER_REGISTRY
from ..exceptions import ConfigError
from ..models import BuiltContext, ContextPlacement, ScoredChunk

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Small-to-Big 父块还原
# ---------------------------------------------------------------------------


class SmallToBigExpander:
    """把「命中的子块」转换成「完整的父块上下文」。

    Args:
        parent_store: 离线阶段写入的父块存储。
        enabled: ``False`` 时直接返回子块（消融对照）。
        on_missing: 父块缺失时的行为。默认 ``"keep_child"`` —— 记录 warning
            并退回子块文本，**不让一次索引不一致导致整条检索失败**；
            ``"drop"`` 则丢弃该候选。
    """

    def __init__(self, parent_store: ParentStore, enabled: bool = True,
                 on_missing: str = "keep_child") -> None:
        if on_missing not in {"keep_child", "drop"}:
            raise ConfigError("on_missing 只能是 'keep_child' 或 'drop'")
        self.parent_store = parent_store
        self.enabled = enabled
        self.on_missing = on_missing

    def expand(self, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """按相关性顺序还原父块，并按 ``parent_id`` 去重。

        多个子块命中同一父块时只保留**最相关**的那条（输入已按相关性排序），
        其 ``chunk`` 仍保留原子块用于溯源与评测，``parent`` 字段才是真正
        进入上下文的文本。
        """
        if not self.enabled:
            return list(chunks)
        parent_ids = [item.chunk.parent_id for item in chunks]
        parents = self.parent_store.get_many(parent_ids)
        expanded: list[ScoredChunk] = []
        seen: set[str] = set()
        for item in chunks:
            parent = parents.get(item.chunk.parent_id)
            if parent is None:
                logger.warning(
                    "父块 %s 不存在（索引可能不一致），策略=%s",
                    item.chunk.parent_id, self.on_missing,
                )
                if self.on_missing == "drop":
                    continue
                item.parent = None
                expanded.append(item)
                continue
            if parent.id in seen:
                continue
            seen.add(parent.id)
            item.parent = parent
            expanded.append(item)
        return expanded


# ---------------------------------------------------------------------------
# 上下文编排
# ---------------------------------------------------------------------------


def _display_order(count: int, placement: str) -> list[int]:
    """返回「展示位置 -> 相关性名次」的映射。

    ``edges`` 示例（n=5）：展示顺序为 ``[0, 2, 4, 3, 1]``，即最相关(rank 0)
    在开头、次相关(rank 1) 在结尾、最不相关(rank 4) 落在正中间。
    """
    ranks = list(range(count))
    if placement in {"sequential", "head"}:
        return ranks
    if placement == "tail":
        return list(reversed(ranks))
    if placement == "edges":
        order = [0] * count
        left, right = 0, count - 1
        for rank in ranks:
            if rank % 2 == 0:
                order[left] = rank
                left += 1
            else:
                order[right] = rank
                right -= 1
        return order
    raise ConfigError(f"未知的上下文位置策略：{placement!r}")


def _block_body(item: ScoredChunk) -> str:
    """Small-to-Big：优先用父块全文，缺失时退回子块文本。"""
    return item.parent.text if item.parent is not None else item.chunk.text


def _block_text(item: ScoredChunk, label: int, body_limit: int | None = None) -> str:
    """渲染单个资料块，带来源与引用编号。"""
    body = _block_body(item).strip()
    if body_limit is not None and len(body) > body_limit:
        body = body[:body_limit].rstrip() + " …（已截断）"
    source = (
        item.chunk.metadata.get("filename")
        or item.chunk.metadata.get("source")
        or "unknown"
    )
    return f"[{label}] 来源：{source}\n{body}"


@CONTEXT_BUILDER_REGISTRY.register("edges")
class EdgePlacementContextBuilder(ContextBuilder):
    """按「首 / 尾放置高相关片段」编排上下文。

    Args:
        placement: ``"edges"``（默认 U 形）/ ``"sequential"`` / ``"tail"``。
        expand_to_parent: 是否用父块文本（由 SmallToBigExpander 预先完成，
            这里只做开关校验与日志，实际还原在 expander 中）。
        max_context_chars: 上下文总字符预算；超预算时**优先丢弃中间的
            低相关块**，而不是砍掉头尾关键片段。
    """

    def __init__(self, placement: str = "edges", expand_to_parent: bool = True,
                 max_context_chars: int = 4000) -> None:
        if max_context_chars <= 0:
            raise ConfigError("max_context_chars 必须为正整数")
        _display_order(3, placement)  # 构造期就校验策略名
        self.placement = placement
        self.expand_to_parent = expand_to_parent
        self.max_context_chars = max_context_chars

    def build(self, query: str, chunks: Sequence[ScoredChunk]) -> BuiltContext:
        """组装上下文文本，并给出每个块的位置记录。"""
        if not chunks:
            return BuiltContext(text="", placements=[], chunks=[], truncated_ids=[])

        # 输入按相关性排序（rank 升序），重排时也用这个顺序作为 rank 基准
        ranked = sorted(chunks, key=lambda item: (item.rank, item.chunk.id))
        # 1) 预算裁剪：先丢最不相关的（在 edges 策略下正好是中间位置）
        selected = list(ranked)
        truncated_ids: list[str] = []
        while len(selected) > 1 and self._total_length(selected) > self.max_context_chars:
            removed = selected.pop()
            truncated_ids.append(removed.chunk.id)

        # 2) 单块过长时等比截断正文，保证总预算不被突破
        body_limit: int | None = None
        if self._total_length(selected) > self.max_context_chars:
            overhead = self.max_context_chars // max(len(selected), 1)
            body_limit = max(200, overhead - 40)
            logger.warning(
                "单个资料块过长，按每块 %d 字符截断（max_context_chars=%d）",
                body_limit, self.max_context_chars,
            )

        # 3) U 形 / 顺序 / 尾部 位置编排
        order = _display_order(len(selected), self.placement)
        blocks: list[str] = []
        placements: list[ContextPlacement] = []
        display_chunks: list[ScoredChunk] = []
        for position, rank in enumerate(order):
            item = selected[rank]
            blocks.append(_block_text(item, position + 1, body_limit))
            placements.append(ContextPlacement(
                chunk_id=item.chunk.id,
                position=position,
                relevance_rank=item.rank,
            ))
            display_chunks.append(item)

        text = "\n\n".join(blocks)
        logger.debug(
            "上下文编排：placement=%s，%d 块，位置序列(rank)=%s，截断 %d 块",
            self.placement, len(selected), order, len(truncated_ids),
        )
        return BuiltContext(
            text=text, placements=placements,
            chunks=display_chunks, truncated_ids=truncated_ids,
        )

    def _total_length(self, items: Sequence[ScoredChunk]) -> int:
        return sum(len(_block_text(item, 1)) + 2 for item in items)


@CONTEXT_BUILDER_REGISTRY.register("sequential")
class SequentialContextBuilder(EdgePlacementContextBuilder):
    """消融基线：按相关性从头排到尾（不做首/尾重排）。

    与 ``EdgePlacementContextBuilder(placement="edges")`` 做 A/B 对比，
    用 Golden Set 上的答案质量 / Context Precision 验证「Lost in the Middle」
    是否真的存在（注意事项第 4 条）。
    """

    def __init__(self, max_context_chars: int = 4000,
                 expand_to_parent: bool = True) -> None:
        super().__init__(
            placement="sequential",
            expand_to_parent=expand_to_parent,
            max_context_chars=max_context_chars,
        )


# ---------------------------------------------------------------------------
# Prompt 组装
# ---------------------------------------------------------------------------


def render_prompt(query: str, context: BuiltContext,
                  config: PromptConfig) -> str:
    """把上下文包成可直接发给 LLM 的 prompt。

    结构刻意为「指令 → 资料（U 形排列）→ 问题 → 再次强调 + 问题」：
    末尾重复问题让关键指令与用户意图落在**近因位置**，降低长上下文中段
    被忽略的风险（注意事项第 3 条）。

    Args:
        query: 用户问题。
        context: :meth:`ContextBuilder.build` 的产物。
        config: prompt 配置。

    Returns:
        完整 prompt 字符串；调用方**只在** ``RetrievalResult.answer_ready``
        为真时把它交给 LLM。
    """
    question = config.question_template.format(query=query)
    lines = [
        config.system_instruction,
        "",
        config.context_header,
        "",
        context.text,
        "",
        question,
    ]
    if config.repeat_question_at_tail:
        lines.extend([
            "",
            "再次强调：只依据上文资料作答，并用 [编号] 标注引用；"
            "资料不足时直接说明「没有找到相关信息」。",
            question,
        ])
    return "\n".join(lines)


__all__: Sequence[str] = [
    "SmallToBigExpander", "EdgePlacementContextBuilder",
    "SequentialContextBuilder", "render_prompt",
]
