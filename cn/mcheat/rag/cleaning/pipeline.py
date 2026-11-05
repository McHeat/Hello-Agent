"""清洗流水线：责任链 + 统一错误策略 + 可观测报告。

``CleaningPipeline`` 自身也实现了 :class:`Cleaner` 接口（组合模式），因此
可以整条流水线作为另一个流水线的一步，便于分层复用。

错误策略（``cleaning.on_error``）：

* ``"raise"``：任一 Cleaner 抛错立即中断并抛 :class:`CleaningError`（默认，
  适合离线建索引，数据问题要暴露而不是悄悄吞掉）；
* ``"skip"``：丢弃出错文档、记录原因、继续处理其余文档（适合脏数据源）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .._utils import get_logger
from ..config import CleaningConfig
from ..core.interfaces import Cleaner
from ..core.registry import CLEANER_REGISTRY
from ..exceptions import CleaningError, ConfigError
from ..models import Document

logger = get_logger(__name__)


@dataclass
class CleaningReport:
    """一次清洗的可观测结果，可随索引产物一起归档。

    ``dropped_by_step`` 精确记录是哪个 Cleaner 返回了 ``None``，
    便于定位「为什么某篇文档没进索引」。
    """

    total_in: int = 0
    kept: int = 0
    dropped_by_step: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def dropped(self) -> int:
        return self.total_in - self.kept

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_in": self.total_in,
            "kept": self.kept,
            "dropped": self.dropped,
            "dropped_by_step": self.dropped_by_step,
            "errors": self.errors,
        }


class CleaningPipeline(Cleaner):
    """按配置顺序执行若干 Cleaner。

    Attributes:
        cleaners: 有序步骤列表。
        on_error: ``"raise"`` 或 ``"skip"``。
    """

    def __init__(self, cleaners: Sequence[Cleaner], on_error: str = "raise") -> None:
        if on_error not in {"raise", "skip"}:
            raise ConfigError("on_error 只能是 'raise' 或 'skip'")
        for cleaner in cleaners:
            if not isinstance(cleaner, Cleaner):
                raise ConfigError(
                    f"清洗步骤 {type(cleaner).__name__} 未实现 Cleaner 接口"
                )
        self.cleaners = list(cleaners)
        self.on_error = on_error

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def from_config(cls, config: CleaningConfig) -> "CleaningPipeline":
        """按 ``cleaning.steps`` 从注册表构造流水线。

        内置步骤的通用参数（``boilerplate_patterns`` / ``min_length``）会自动
        注入；``step_params[步骤名]`` 可覆盖任意构造参数；未注册的步骤名会
        给出包含可用值列表的报错。
        """
        cleaners: list[Cleaner] = []
        for step in config.steps:
            params: dict[str, Any] = dict(config.step_params.get(step, {}))
            if step == "boilerplate":
                params.setdefault("patterns", config.boilerplate_patterns)
            if step == "drop_empty":
                params.setdefault("min_length", config.min_length)
            cleaners.append(CLEANER_REGISTRY.create(step, **params))
        return cls(cleaners=cleaners, on_error=config.on_error)

    @property
    def steps(self) -> list[str]:
        """各步骤名字（用于日志与报告）。"""
        return [cleaner.name for cleaner in self.cleaners]

    def __len__(self) -> int:
        return len(self.cleaners)

    # -- 执行 ---------------------------------------------------------------

    def _reset_stateful(self) -> None:
        """重置带批次状态的 Cleaner（如去重器），保证 run 之间互不污染。"""
        for cleaner in self.cleaners:
            reset = getattr(cleaner, "reset", None)
            if callable(reset):
                reset()

    def clean(self, document: Document) -> Document | None:
        """清洗单篇文档（不重置批次状态），用于在线单文档场景。

        Raises:
            CleaningError: ``on_error="raise"`` 且某步骤失败。
        """
        current: Document | None = document
        for cleaner in self.cleaners:
            if current is None:
                return None
            try:
                current = cleaner.clean(current)
            except Exception as exc:
                if self.on_error == "raise":
                    raise CleaningError(
                        f"清洗步骤 {cleaner.name} 处理 "
                        f"{document.metadata.get('source', '?')} 失败："
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                logger.warning("清洗步骤 %s 失败，丢弃该文档：%s", cleaner.name, exc)
                return None
        return current

    def run(self, documents: Sequence[Document]) -> tuple[list[Document], CleaningReport]:
        """批量清洗，返回保留下来的文档与报告。

        与 :meth:`clean` 的区别：会先重置批次状态，并按 ``on_error`` 决定
        「抛错」还是「记录并跳过」，适合离线索引一次处理成百上千篇文档。
        """
        self._reset_stateful()
        report = CleaningReport(total_in=len(documents))
        kept: list[Document] = []

        for document in documents:
            current: Document | None = document
            drop_step: str | None = None
            for cleaner in self.cleaners:
                if current is None:
                    break
                try:
                    current = cleaner.clean(current)
                except Exception as exc:
                    report.errors.append({
                        "source": str(document.metadata.get("source", "?")),
                        "step": cleaner.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    if self.on_error == "raise":
                        raise CleaningError(
                            f"清洗步骤 {cleaner.name} 处理 "
                            f"{document.metadata.get('source', '?')} 失败："
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    logger.warning("跳过清洗失败的文档 %s：%s",
                                   document.metadata.get("source", "?"), exc)
                    drop_step = cleaner.name
                    current = None
                    break
                if current is None:
                    drop_step = cleaner.name
                    break

            if current is None:
                reason = drop_step or "unknown"
                report.dropped_by_step[reason] = report.dropped_by_step.get(reason, 0) + 1
                continue
            kept.append(current)

        report.kept = len(kept)
        logger.info(
            "清洗完成：输入 %d，保留 %d，丢弃 %d，明细 %s",
            report.total_in, report.kept, report.dropped, report.dropped_by_step,
        )
        return kept, report


__all__: Sequence[str] = ["CleaningPipeline", "CleaningReport"]
