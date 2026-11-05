"""领域模型：贯穿「离线索引 → 在线检索 → 评测」全流程的数据结构。

所有对象都是纯 ``dataclass``，不含任何第三方依赖，也不持有资源句柄，
因此可以安全地 JSON 序列化（用于索引产物持久化）与跨进程传递。

命名约定：

* ``Document``   —— Loader 产出的原始文档；
* ``ParentChunk``—— 大段落（Small-to-Big 中「Big」，最终进 prompt 的完整上下文）；
* ``ChildChunk`` —— 小片段（Small-to-Big 中「Small」，只用于检索命中）；
* ``*Hit`` / ``ScoredChunk`` —— 检索各阶段带分数的中间结果；
* ``RetrievalResult`` —— 在线检索最终产物，含诊断信息，供评测与兜底使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ._utils import stable_id

# 向量统一用不可变 tuple，避免被调用方误改导致索引与查询不一致
Vector = tuple[float, ...]


# ---------------------------------------------------------------------------
# 检索动作常量
# ---------------------------------------------------------------------------


class RetrievalAction:
    """兜底链路的最终动作（见 ``retrieval/fallback.py``）。"""

    ANSWER = "answer"        # 正常：有足够证据，允许生成答案
    HANDOFF = "handoff"      # 转人工
    NO_ANSWER = "no_answer"  # 如实说明「没找到相关信息」，禁止 LLM 自由发挥


# ---------------------------------------------------------------------------
# 原始文档与父子块
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """Loader 产出的原始文档。

    Attributes:
        page_content: 正文。
        metadata: 至少包含 ``source``（来源路径/URL）；Loader 可补充
            ``title``、``row_index``、``doc_type`` 等字段。这些字段会被
            逐层透传到 ``ChildChunk.metadata``，用于过滤与溯源。
        id: 稳定 ID；为空时按 ``(source, page_content)`` 自动生成。
    """

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = "doc-" + stable_id(
                self.metadata.get("source", ""), self.page_content
            )

    def copy_with(self, page_content: str | None = None,
                  metadata: dict[str, Any] | None = None) -> "Document":
        """返回保留 ``id`` 的副本。Cleaner 用它替换正文，保证 ID 稳定。"""
        return Document(
            page_content=self.page_content if page_content is None else page_content,
            metadata=dict(self.metadata) if metadata is None else metadata,
            id=self.id,
        )


@dataclass
class ParentChunk:
    """大段落（父块）。

    父块**不参与向量检索**，只作为命中子块后返回的完整上下文。
    ``id`` 由 ``(doc_id, parent_index, text)`` 稳定哈希得到。
    """

    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    index: int = 0  # 在同一文档内的序号，用于还原阅读顺序


@dataclass
class ChildChunk:
    """小片段（子块）——唯一被 embedding 并写入向量库 / BM25 索引的单元。

    Attributes:
        parent_id: 所属父块 ID，Small-to-Big 还原的纽带。
        doc_id: 所属原始文档 ID。
        start / end: 子块文本在 **父块文本** 中的字符区间。
            由切分器保证 ``parent.text[start:end]`` 与 ``text`` 一致，
            这是父子切分正确性的不变量。
        index: 子块在同一父块内的序号。
    """

    id: str
    parent_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    index: int = 0
    start: int = 0
    end: int = 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供向量库 / 索引产物持久化使用。"""
        return {
            "id": self.id, "parent_id": self.parent_id, "doc_id": self.doc_id,
            "text": self.text, "metadata": self.metadata,
            "index": self.index, "start": self.start, "end": self.end,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChildChunk":
        """从 :meth:`to_dict` 的结果还原。"""
        return cls(**{k: data[k] for k in
                      ("id", "parent_id", "doc_id", "text", "metadata",
                       "index", "start", "end") if k in data})


@dataclass
class SplitResult:
    """切分器输出：一批父块 + 一批子块。"""

    parents: list[ParentChunk] = field(default_factory=list)
    children: list[ChildChunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 召回阶段
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    """单条通路的召回结果。

    Attributes:
        source: 来自哪条通路，``"dense"`` / ``"sparse"`` / 自定义通路名。
    """

    chunk: ChildChunk
    score: float
    source: str = "dense"


@dataclass
class Ranking:
    """交给融合器的一路排序结果。

    Attributes:
        weight: 该通路的权重，由配置 ``components.fusion.params.*_weight`` 提供。
    """

    source: str
    hits: list[Hit]
    weight: float = 1.0


@dataclass
class FusedHit:
    """融合后的候选，保留各通路的原始分数与名次，便于诊断与回归分析。"""

    chunk: ChildChunk
    score: float                       # 融合分（RRF 或加权归一化分）
    sources: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 精排与上下文
# ---------------------------------------------------------------------------


@dataclass
class ScoredChunk:
    """精排后的候选。只有进入 ``final`` 列表的这些块才会被放进 prompt。"""

    chunk: ChildChunk
    score: float                        # 精排分数
    rank: int = 0                       # 精排名次（相关性降序，0 最相关）
    recall_score: float = 0.0           # 融合分，保留供评测 / 调试
    recall_sources: tuple[str, ...] = ()  # 由哪些通路召回
    parent: ParentChunk | None = None   # Small-to-Big 还原出的父块


@dataclass
class ContextPlacement:
    """上下文编排的可观测记录：某个块被放在了第几个位置。"""

    chunk_id: str
    position: int          # 最终在 context 文本中的展示序号
    relevance_rank: int    # 精排相关性名次（0 最相关）


@dataclass
class BuiltContext:
    """组装好的上下文块（尚未拼成完整 prompt）。

    ``placements`` 用于验证「Lost in the Middle」策略：高相关片段应当出现在
    首 / 尾位置，而不是被埋在中间（见 ``retrieval/context.py``）。
    """

    text: str
    placements: list[ContextPlacement] = field(default_factory=list)
    chunks: list[ScoredChunk] = field(default_factory=list)
    truncated_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 兜底
# ---------------------------------------------------------------------------


@dataclass
class FallbackContext:
    """兜底策略做决策时能看到的完整现场快照（只读语义，不要修改）。"""

    query: str
    stage: str                       # "recall" | "rerank" | "context"
    coarse_count: int                # 粗召回候选数
    final_count: int                 # 精排后候选数
    best_score: float                # 最高精排分
    rerank_scores: list[float]
    context_text: str
    attempt: int                     # 当前是第几次（0 表示首次）
    current_top_k: int               # 本次使用的粗召回 Top-K
    max_retries: int
    widen_factor: float
    max_coarse_top_k: int
    handoff_enabled: bool
    no_answer_text: str
    handoff_text: str
    min_recall_hits: int
    min_best_score: float
    corpus_size: int = 0             # 语料规模，用于避免「小库也要求召回 3 条」的误判


@dataclass
class FallbackStep:
    """兜底链路中某一步的动作记录。"""

    strategy: str
    action: str                      # "retry" | "handoff" | "no_answer"
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackDecision:
    """一次检索的兜底结论，完整保留链路轨迹便于审计。"""

    triggered: bool = False
    steps: list[FallbackStep] = field(default_factory=list)
    final_action: str = RetrievalAction.ANSWER
    retry_top_k: int | None = None
    message: str = ""
    triggers: list[str] = field(default_factory=list)  # 触发兜底的具体原因

    @property
    def should_retry(self) -> bool:
        """是否需要用更大的 Top-K 重跑粗召回。"""
        return self.retry_top_k is not None


# ---------------------------------------------------------------------------
# 在线检索产物
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """在线检索的完整产物。

    刻意同时暴露 ``coarse`` 与 ``final`` 两阶段结果，这是为了落实
    「召回用 Context Recall、精度用 Context Precision **分开看**」：

    * ``coarse``：粗召回的 Top-30~50，用于算 Context Recall；
    * ``final`` ：精排后的 Top-3~5，用于算 Context Precision，也是**唯一**
      进入 ``prompt`` 的内容。
    """

    query: str
    coarse: list[ScoredChunk] = field(default_factory=list)
    final: list[ScoredChunk] = field(default_factory=list)
    context: BuiltContext | None = None
    prompt: str = ""
    fallback: FallbackDecision = field(default_factory=FallbackDecision)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def answer_ready(self) -> bool:
        """是否允许把 ``prompt`` 交给 LLM 生成答案。

        兜底判定为转人工 / 如实回答时返回 ``False``，调用方**必须**直接返回
        ``fallback.message``，不得调用 LLM，以避免无依据的幻觉回答。
        """
        return self.fallback.final_action == RetrievalAction.ANSWER

    def final_ids(self) -> list[str]:
        """最终进 prompt 的子块 ID（有序）。"""
        return [item.chunk.id for item in self.final]


# ---------------------------------------------------------------------------
# 离线索引产物
# ---------------------------------------------------------------------------


@dataclass
class IndexStats:
    """离线建索引的统计与跳过明细，用于可观测性与数据质量排查。"""

    documents_loaded: int = 0
    documents_kept: int = 0
    documents_dropped: int = 0
    parents: int = 0
    children: int = 0
    vectors_upserted: int = 0
    pruned: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的 dict。"""
        return {
            "documents_loaded": self.documents_loaded,
            "documents_kept": self.documents_kept,
            "documents_dropped": self.documents_dropped,
            "parents": self.parents,
            "children": self.children,
            "vectors_upserted": self.vectors_upserted,
            "pruned": self.pruned,
            "skipped": self.skipped,
            "duration_ms": round(self.duration_ms, 3),
        }


# ---------------------------------------------------------------------------
# 类型辅助
# ---------------------------------------------------------------------------


def hits_to_chunks(hits: Sequence[Hit]) -> list[ChildChunk]:
    """去掉分数，只保留子块（保持顺序）。"""
    return [hit.chunk for hit in hits]
