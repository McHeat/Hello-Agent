"""全部**抽象接口 / 扩展点**（``abc.ABC``）。

这是框架「高度抽象、可插拔」的核心：上层编排（``indexing.pipeline``、
``retrieval.pipeline``、``api.RAGSystem``）只依赖本文件里的接口，不依赖任何
具体实现。想替换组件时，二选一：

1. 继承对应 ABC 实现新方法，用注册表 ``register(...)`` 注册，然后在
   ``configs/defaults.json`` 里把 ``components.<key>.name`` 换成新名字；
2. 直接构造好对象，通过 ``RAGSystem(overrides={"vector_store": my_store})``
   注入（注册表会做 ``isinstance`` 校验，保证契约不被破坏）。

接口清单（按数据流顺序）::

    DocumentLoader -> Cleaner -> ParentChildSplitter -> Embedder
        -> VectorStore / DenseRetriever / SparseRetriever  [离线]
    DenseRetriever + SparseRetriever -> FusionStrategy -> Reranker
        -> ParentStore -> ContextBuilder -> FallbackStrategy   [在线]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from ..exceptions import PersistenceNotSupportedError
from ..models import (
    BuiltContext,
    ChildChunk,
    Document,
    FallbackContext,
    FallbackStep,
    FusedHit,
    Hit,
    ParentChunk,
    Ranking,
    ScoredChunk,
    SplitResult,
    Vector,
)

Source = str | Path


# ---------------------------------------------------------------------------
# 持久化能力（可选 mixin）
# ---------------------------------------------------------------------------


class Stateful:
    """让组件能随 ``RAGSystem.save()`` 一起写入索引产物。

    默认实现抛 :class:`PersistenceNotSupportedError`。自带持久化的组件
    （例如 Chroma 自己管理磁盘目录）可以覆写成返回 ``{"external": True, ...}``
    并在 ``import_state`` 里做空操作。
    """

    def export_state(self) -> dict[str, Any]:
        """导出可 JSON 序列化的全部状态。"""
        raise PersistenceNotSupportedError(type(self).__name__, "export_state")

    def import_state(self, state: dict[str, Any]) -> None:
        """从 ``export_state()`` 的结果恢复状态。"""
        raise PersistenceNotSupportedError(type(self).__name__, "import_state")

    @property
    def supports_persistence(self) -> bool:
        """是否实现了导出 / 导入（供 ``RAGSystem.save`` 判断）。"""
        return type(self).export_state is not Stateful.export_state


# ---------------------------------------------------------------------------
# 离线：加载 / 清洗 / 切分
# ---------------------------------------------------------------------------


class DocumentLoader(ABC):
    """把一种外部数据源转成一批 :class:`~models.Document`。"""

    @property
    def name(self) -> str:
        """组件名，默认取类名。"""
        return type(self).__name__

    @abstractmethod
    def supports(self, source: Source) -> bool:
        """是否能处理该来源（通常按文件后缀判断）。"""

    @abstractmethod
    def load(self, source: Source) -> list[Document]:
        """读取并返回文档列表；一个来源可以产出多篇文档（如 CSV 每行一篇）。

        Raises:
            LoaderError: 读取 / 解析失败时抛出，消息中需带来源路径。
        """


class Cleaner(ABC):
    """清洗流水线中的一环，输入输出都是 :class:`~models.Document`。

    返回 ``None`` 表示**丢弃该文档**（例如清洗后为空、与已有文档重复）。
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def clean(self, document: Document) -> Document | None:
        """清洗单篇文档；返回 ``None`` 表示丢弃。"""


class TextCleaner(Cleaner):
    """只改正文的 Cleaner 便捷基类。

    子类只需实现 :meth:`clean_text`，元数据与稳定 ID 由基类原样保留。
    """

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """对纯文本做变换。"""

    def clean(self, document: Document) -> Document | None:
        return document.copy_with(page_content=self.clean_text(document.page_content))


class ParentChildSplitter(ABC):
    """父子切分器（Small-to-Big）。

    契约：返回的每个 ``ChildChunk`` 都必须满足

    * ``parent_id`` 指向 ``parents`` 中确实存在的父块；
    * ``text == parent.text[start:end]``（子串不变量）。

    在线检索命中子块后，靠 ``parent_id`` 取回完整父块上下文。
    """

    @abstractmethod
    def split(self, documents: Sequence[Document]) -> SplitResult:
        """把一批文档切成父块与子块。"""


class Embedder(ABC):
    """文本 → 向量。索引侧与查询侧必须使用**同一个实例/模型**。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度；换模型会触发 :class:`DimensionMismatchError`。"""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """批量向量化（返回顺序与输入一致）。"""

    @abstractmethod
    def embed_query(self, text: str) -> Vector:
        """单条查询向量化。部分模型区分 query/document 前缀，故单独定义。"""


# ---------------------------------------------------------------------------
# 可插拔向量库
# ---------------------------------------------------------------------------


class VectorStore(Stateful, ABC):
    """**可插拔向量库接口**（低层）。

    只负责「按 ID 存向量 + 取回 Top-K」，不负责 embedding。
    默认实现 ``InMemoryVectorStore`` 为零依赖的精确余弦检索；接 FAISS /
    Chroma / Milvus / pgvector 时实现本接口即可，其余代码零改动。

    ``upsert`` 语义：ID 已存在则覆盖，保证重复建索引幂等。
    """

    @abstractmethod
    def upsert(self, chunks: Sequence[ChildChunk],
               vectors: Sequence[Vector]) -> int:
        """写入 / 覆盖一批子块及其向量，返回写入条数。

        Raises:
            DimensionMismatchError: 向量维度与库中已有维度不一致。
        """

    @abstractmethod
    def search(self, query_vector: Vector, k: int,
               metadata_filter: dict[str, Any] | None = None) -> list[Hit]:
        """按相似度返回最多 ``k`` 条命中（分数降序）。

        Args:
            metadata_filter: 精确匹配过滤，如 ``{"source": "a.md"}``；
                ``None`` 表示不过滤。实现必须支持该参数（可为空实现），
                因为在线检索会按租户 / 来源做过滤。
        """

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> int:
        """删除指定 ID，返回实际删除条数。"""

    @abstractmethod
    def ids(self) -> set[str]:
        """当前全部 ID（供离线索引裁剪过期数据）。"""

    @abstractmethod
    def count(self) -> int:
        """当前条数。"""

    def clear(self) -> None:
        """清空（默认基于 :meth:`delete` 实现）。"""
        self.delete(sorted(self.ids()))


class RecallRetriever(Stateful, ABC):
    """一条**召回通路**的统一接口（高层）。

    ``HybridRetriever`` 只认这个接口，于是「稠密通路」与「稀疏通路」可以
    自由组合/替换：默认 ``EmbeddingDenseRetriever``（Embedder + VectorStore）
    与 ``BM25Index``；也可以接 LangChain 的 VectorStore、Elasticsearch 等。
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def index(self, chunks: Sequence[ChildChunk]) -> int:
        """离线写入一批子块，返回写入条数。"""

    @abstractmethod
    def search(self, query: str, k: int,
               metadata_filter: dict[str, Any] | None = None) -> list[Hit]:
        """在线按查询文本召回最多 ``k`` 条，``hit.source`` 需标明通路名。"""

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> int:
        """删除指定 ID。"""

    @abstractmethod
    def ids(self) -> set[str]:
        """当前全部子块 ID。"""

    @abstractmethod
    def count(self) -> int:
        """当前条数。"""


class FusionStrategy(ABC):
    """多路召回结果的融合策略（RRF / 加权求和 / 自定义）。"""

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def fuse(self, rankings: Sequence[Ranking], limit: int) -> list[FusedHit]:
        """把多路排序融合成最多 ``limit`` 条候选（融合分降序）。

        必须按 ``chunk.id`` 去重，并把各通路原始分 / 名次写进
        ``FusedHit.sources`` / ``FusedHit.ranks``，供诊断使用。
        """


@runtime_checkable
class CrossEncoderScorer(Protocol):
    """交叉编码器打分器协议（结构化子类型，无需继承即可注入）。

    与 ``sentence_transformers.CrossEncoder`` 的 ``predict`` 签名兼容。
    """

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        """对一批 ``(query, passage)`` 返回相关度分数（越高越相关）。"""


class Reranker(ABC):
    """精排器：把 Top-30~50 粗候选压到进 prompt 的 Top-3~5。

    **这是《rag编码注意事项》第 1 条的关键落地点**：编排层会强制把
    :meth:`rerank` 的 ``top_n`` 设为 ``retrieval.rerank_top_n``，且只把
    返回结果放进 prompt，粗召回的几十条绝不整体塞进去。
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[FusedHit],
               top_n: int) -> list[ScoredChunk]:
        """对候选重打分并返回前 ``top_n`` 条（``rank`` 从 0 递增）。"""


class ParentStore(Stateful, ABC):
    """父块存储：离线写、在线读，是父子切分跨阶段的桥梁。

    默认 ``InMemoryParentStore``；``SqliteParentStore`` 基于标准库
    ``sqlite3``，适合父块量大 / 需要跨进程直接读取的场景。
    """

    @abstractmethod
    def put(self, parents: Sequence[ParentChunk]) -> int:
        """批量 upsert，返回条数。"""

    @abstractmethod
    def get(self, parent_id: str) -> ParentChunk | None:
        """按 ID 取单个父块；不存在返回 ``None``。"""

    @abstractmethod
    def get_many(self, ids: Sequence[str]) -> dict[str, ParentChunk]:
        """批量取父块；缺失的 ID 不出现在返回 dict 中。"""

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> int:
        """删除父块。"""

    @abstractmethod
    def ids(self) -> set[str]:
        """全部父块 ID。"""

    @abstractmethod
    def count(self) -> int:
        """父块条数。"""


# ---------------------------------------------------------------------------
# 在线：上下文编排 / 兜底
# ---------------------------------------------------------------------------


class ContextBuilder(ABC):
    """把精排结果组装成上下文文本。

    默认 ``EdgePlacementContextBuilder`` 落实《注意事项》第 3 条
    「Lost in the Middle」：高相关片段放首 / 尾，低相关放中间。
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def build(self, query: str, chunks: Sequence[ScoredChunk]) -> BuiltContext:
        """返回带 ``placements`` 记录的上下文，便于验证位置策略。"""


class FallbackStrategy(ABC):
    """兜底链路中的一环（《注意事项》第 5 条）。

    ``evaluate`` 返回 ``None`` 表示「本策略不介入」，交给下一环；
    返回 ``FallbackStep`` 则链路立即结束。策略按注册顺序组合成
    「扩 Top-K 重试 → 转人工 → 如实回答」的降级梯度。
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def evaluate(self, context: FallbackContext) -> FallbackStep | None:
        """根据现场快照决定是否介入。"""
