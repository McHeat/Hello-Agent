"""父子切分（Small-to-Big）—— 落实《rag编码注意事项》第 2 条。

问题：块太大 → 一个块混多个主题、检索粒度粗；块太小 → 上下文被切断、
丢失跨句依赖。

解法：

1. 先把文档切成**大段落**（父块，默认 1200 字符），尽量在段落 / 句子边界断开；
2. 再把每个父块切成**小片段**（子块，默认 240 字符、48 字符重叠），
   带句子边界对齐；
3. **只有子块**被 embedding 并写进向量库 / BM25 索引（命中准）；
4. 在线检索命中子块后，用 ``parent_id`` 从 ParentStore 取回整个父块
   作为上下文（信息完整）。见 ``retrieval/context.py``。

关键不变量（构造期就校验）：``child.text == parent.text[child.start:child.end]``。
这保证了 Small-to-Big 的还原是有据可依的，不会出现「检索到一个片段，
却返回一段不相干文本」的情况。
"""
from __future__ import annotations

from typing import Sequence

from .._utils import get_logger, stable_id
from ..core.interfaces import ParentChildSplitter
from ..core.registry import SPLITTER_REGISTRY
from ..exceptions import ConfigError
from ..models import ChildChunk, Document, ParentChunk, SplitResult

logger = get_logger(__name__)

# 优先级从高到低：先按段落，再按句末标点，最后按空格 / 单字。
# 末尾的 "" 表示「逐字符」这一最后手段。
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n", "\n", "。", "！", "？", "；", "！", "？",
    ". ", "! ", "? ", "; ", "，", ", ", " ", "",
)

# 用于在子块窗口右边界附近「吸附」到自然断句点
_BOUNDARY_CHARS = frozenset("\n。！？；.!?;，,、 ")


def _split_keep(text: str, separator: str) -> list[str]:
    """按分隔符切分并**保留分隔符在前一段末尾**，避免丢失标点。"""
    pieces = text.split(separator)
    return [
        piece + separator if i < len(pieces) - 1 else piece
        for i, piece in enumerate(pieces)
    ]


def _hard_cut(text: str, size: int) -> list[str]:
    """无任何可用分隔符时的兜底：定长硬切。"""
    return [text[i : i + size] for i in range(0, len(text), size)]


def _recursive_split(text: str, separators: Sequence[str], size: int) -> list[str]:
    """递归按分隔符切到每段不超过 ``size``。

    与 LangChain ``RecursiveCharacterTextSplitter`` 思路相同，但保留了分隔符
    且不依赖第三方库。若所有分隔符都用尽仍有超长段，则退化为定长硬切，
    保证任何输入都能被正确处理（不会静默丢数据）。
    """
    if len(text) <= size:
        return [text]
    if not separators:
        return _hard_cut(text, size)

    separator = separators[0]
    remaining = separators[1:]
    if separator and separator not in text:
        return _recursive_split(text, remaining, size)

    pieces = _split_keep(text, separator) if separator else list(text)
    result: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if len(piece) <= size:
            result.append(piece)
        else:
            result.extend(_recursive_split(piece, remaining, size))
    return result


def _merge_units(units: Sequence[str], size: int) -> list[str]:
    """把细碎单元贪心合并成不超过 ``size`` 的连续块。

    合并是**相邻拼接**，因此每块都是原文的连续子串，父子不变量天然成立。
    """
    chunks: list[str] = []
    buffer = ""
    for unit in units:
        if not unit.strip():
            continue
        if len(buffer) + len(unit) <= size:
            buffer += unit
        else:
            if buffer.strip():
                chunks.append(buffer)
            # 单元本身不应超过 size（_recursive_split 已保证），直接作为新块起点
            buffer = unit
    if buffer.strip():
        chunks.append(buffer)
    return chunks


@SPLITTER_REGISTRY.register("character")
class CharacterParentChildSplitter(ParentChildSplitter):
    """边界感知的父子切分器（默认）。

    Args:
        parent_size: 父块目标上限（字符数）。
        child_size: 子块目标上限，必须 ``<= parent_size``。
        child_overlap: 相邻子块重叠字符数，必须 ``< child_size``。
            重叠是为了不把跨句依赖硬切断。
        separators: 分隔符优先级；``None`` 用 :data:`DEFAULT_SEPARATORS`。
        respect_boundaries: ``False`` 时子块退化为纯定长滑窗（消融对照）。
    """

    def __init__(
        self,
        parent_size: int = 1200,
        child_size: int = 240,
        child_overlap: int = 48,
        separators: Sequence[str] | None = None,
        respect_boundaries: bool = True,
    ) -> None:
        if parent_size <= 0 or child_size <= 0:
            raise ConfigError("parent_size / child_size 必须为正整数")
        if child_size > parent_size:
            raise ConfigError("child_size 不能大于 parent_size（子块必须是父块的子串）")
        if not 0 <= child_overlap < child_size:
            raise ConfigError("child_overlap 需满足 0 <= overlap < child_size")
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.separators = list(separators if separators is not None else DEFAULT_SEPARATORS)
        self.respect_boundaries = respect_boundaries

    def split(self, documents: Sequence[Document]) -> SplitResult:
        """把一批文档切成父块 + 子块。

        Raises:
            ConfigError: 父子不变量被破坏（理论上不会发生，属于防御性断言）。
        """
        result = SplitResult()
        for document in documents:
            text = document.page_content
            if not text.strip():
                logger.debug("跳过空文档：%s", document.metadata.get("source", "?"))
                continue

            # --- 1) 父块：先递归细分，再贪心合并到 parent_size ---
            units = _recursive_split(text, self.separators, self.parent_size)
            parent_texts = _merge_units(units, self.parent_size) or [text[: self.parent_size]]

            for parent_index, parent_text in enumerate(parent_texts):
                parent_meta = dict(document.metadata)
                parent_meta.update({
                    "doc_id": document.id,
                    "parent_index": parent_index,
                })
                parent = ParentChunk(
                    id="par-" + stable_id(document.id, parent_index, parent_text),
                    doc_id=document.id,
                    text=parent_text,
                    metadata=parent_meta,
                    index=parent_index,
                )
                result.parents.append(parent)

                # --- 2) 子块：在父块文本上做带重叠的滑窗 ---
                for child_index, (start, end) in enumerate(
                    self._child_windows(parent_text)
                ):
                    child_text = parent_text[start:end]
                    if not child_text.strip():
                        continue
                    # 防御性断言：确保 Small-to-Big 可以无损还原
                    if parent_text[start:end] != child_text:  # pragma: no cover
                        raise ConfigError("父子切分不变量被破坏，请检查 splitter 实现")

                    child_meta = dict(parent_meta)
                    child_meta.update({
                        "parent_id": parent.id,
                        "child_index": child_index,
                        "char_start": start,
                        "char_end": end,
                    })
                    result.children.append(ChildChunk(
                        id="chi-" + stable_id(parent.id, child_index, child_text),
                        parent_id=parent.id,
                        doc_id=document.id,
                        text=child_text,
                        metadata=child_meta,
                        index=child_index,
                        start=start,
                        end=end,
                    ))

        logger.info(
            "父子切分：%d 篇文档 → %d 个父块 / %d 个子块"
            "（parent_size=%d, child_size=%d, overlap=%d）",
            len(documents), len(result.parents), len(result.children),
            self.parent_size, self.child_size, self.child_overlap,
        )
        return result

    def _child_windows(self, parent_text: str) -> list[tuple[int, int]]:
        """产出 ``(start, end)`` 区间列表。

        窗口以 ``child_size`` 为上限、``child_overlap`` 为步长回退；开启
        ``respect_boundaries`` 时，会把右边界向左吸附到最近的自然断句点
        （最多回退窗口的 20%），避免把一句话拦腰截断。
        """
        size = self.child_size
        overlap = self.child_overlap
        length = len(parent_text)
        windows: list[tuple[int, int]] = []
        start = 0
        while start < length:
            end = min(start + size, length)
            if end < length and self.respect_boundaries:
                # 只在本窗口尾部 20% 范围内找断点，避免切出过小的块
                lower = start + max(1, int(size * 0.8))
                snapped = -1
                for index in range(end - 1, lower - 1, -1):
                    if parent_text[index] in _BOUNDARY_CHARS:
                        snapped = index + 1
                        break
                if snapped > start:
                    end = snapped
            windows.append((start, end))
            if end >= length:
                break
            # 保证严格前进：配置已校验 overlap < child_size
            next_start = end - overlap
            start = next_start if next_start > start else end
        return windows


@SPLITTER_REGISTRY.register("fixed")
class FixedWindowParentChildSplitter(CharacterParentChildSplitter):
    """纯定长滑窗版本（关闭边界吸附）。

    用于消融实验：对比「边界感知」与「硬切」对 Context Recall 的影响，
    也用来验证「切分策略是否真的带来收益」，而不是拍脑袋决定（注意事项第 4 条）。
    """

    def __init__(self, parent_size: int = 1200, child_size: int = 240,
                 child_overlap: int = 48, **kwargs) -> None:
        super().__init__(
            parent_size=parent_size,
            child_size=child_size,
            child_overlap=child_overlap,
            respect_boundaries=False,
        )


__all__: Sequence[str] = [
    "DEFAULT_SEPARATORS", "CharacterParentChildSplitter",
    "FixedWindowParentChildSplitter",
]
