"""Loader 基础设施：文件读取兜底、按后缀分发、目录组合加载。

设计要点：

* **5 种格式 Loader** 各自只负责一种数据形态，互不耦合；
* :class:`LoaderRegistry` 按后缀自动选 Loader，新增格式只需 ``register``；
* :class:`DirectoryLoader` 是**组合器**（不是第 6 种格式），递归遍历目录并
  复用注册表，让「离线索引」可以用一个路径吃下混合语料；
* 所有 IO 失败统一抛 :class:`LoaderError`，并在消息里带上来源与修复建议。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .._utils import get_logger
from ..config import LoaderConfig
from ..core.interfaces import DocumentLoader, Source
from ..exceptions import LoaderError, UnsupportedSourceError
from ..models import Document

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 文件读取
# ---------------------------------------------------------------------------


def resolve_file(source: Source, config: LoaderConfig | None = None) -> Path:
    """校验来源路径存在、是文件、且大小未超限。

    Raises:
        LoaderError: 路径不存在 / 不是普通文件 / 超过大小上限。
    """
    cfg = config or LoaderConfig()
    path = Path(source)
    if not path.exists():
        raise LoaderError(f"来源不存在：{path}")
    if not path.is_file():
        raise LoaderError(f"来源不是文件（目录请用 DirectoryLoader）：{path}")
    limit = cfg.max_file_size_mb * 1024 * 1024
    size = path.stat().st_size
    if size > limit:
        raise LoaderError(
            f"文件 {path} 大小 {size / 1024 / 1024:.1f}MB 超过上限 "
            f"{cfg.max_file_size_mb}MB；请拆分文件或调高 loader.max_file_size_mb。"
        )
    return path


def read_text_file(source: Source, config: LoaderConfig | None = None) -> str:
    """读取文本，带**编码回退**与大小限制。

    中文语料常见 GBK/GB18030 编码，直接按 UTF-8 读会抛
    ``UnicodeDecodeError``。这里按 ``encoding -> fallback_encodings`` 逐个
    尝试，全部失败才报错，避免一份乱码文件让整批索引中断。

    Raises:
        LoaderError: IO 失败或所有候选编码都解不开。
    """
    cfg = config or LoaderConfig()
    path = resolve_file(source, cfg)
    encodings = [cfg.encoding] + [e for e in cfg.fallback_encodings if e != cfg.encoding]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            logger.debug("%s 用 %s 解码失败，尝试下一个编码", path, encoding)
        except OSError as exc:
            raise LoaderError(f"读取 {path} 失败：{exc}") from exc
    raise LoaderError(
        f"无法用候选编码 {encodings} 解码 {path}（{last_error}）；"
        f"请在 loader.fallback_encodings 中补充正确编码。"
    )


def base_metadata(path: Path, doc_type: str, **extra) -> dict:
    """统一的来源元数据，保证每个 Chunk 都能溯源到原始文件。"""
    metadata: dict = {
        "source": str(path.resolve()),
        "filename": path.name,
        "suffix": path.suffix.lower(),
        "doc_type": doc_type,
    }
    metadata.update({k: v for k, v in extra.items() if v is not None})
    return metadata


class BaseFileLoader(DocumentLoader):
    """按后缀工作的文件 Loader 基类，封装路径与编码处理。"""

    suffixes: frozenset[str] = frozenset()
    doc_type: str = "file"

    def __init__(self, config: LoaderConfig | None = None) -> None:
        self.config = config or LoaderConfig()

    def supports(self, source: Source) -> bool:
        """后缀匹配且不是目录。后注册的 Loader 优先（见 :class:`LoaderRegistry`）。"""
        path = Path(source)
        return not path.is_dir() and path.suffix.lower() in self.suffixes

    def _read(self, source: Source) -> tuple[Path, str]:
        path = resolve_file(source, self.config)
        return path, read_text_file(path, self.config)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


class LoaderRegistry:
    """按后缀把来源分发给合适的 :class:`DocumentLoader`。

    后注册者优先，因此用户可以 ``registry.register(MyTxtLoader())`` 覆盖
    内置的 ``.txt`` 实现，而无需改动框架代码。
    """

    def __init__(self, config: LoaderConfig | None = None) -> None:
        self.config = config or LoaderConfig()
        self._loaders: list[DocumentLoader] = []

    def register(self, loader: DocumentLoader) -> None:
        """注册（或覆盖）一个 Loader。"""
        if not isinstance(loader, DocumentLoader):
            raise LoaderError(
                f"注册的 Loader 必须继承 DocumentLoader，收到 {type(loader).__name__}"
            )
        self._loaders.append(loader)
        logger.debug("注册 Loader：%s", loader.name)

    def for_source(self, source: Source) -> DocumentLoader:
        """找到能处理该来源的 Loader。

        Raises:
            UnsupportedSourceError: 无匹配 Loader。
        """
        for loader in reversed(self._loaders):
            if loader.supports(source):
                return loader
        raise UnsupportedSourceError(
            f"没有 Loader 能处理来源：{source}（后缀 {Path(source).suffix!r}）。"
            f"已注册：{[item.name for item in self._loaders]}；"
            f"可自定义 DocumentLoader 并调用 registry.register(...)。"
        )

    def load(self, source: Source) -> list[Document]:
        """加载单个来源；异常附上来源信息以便定位。"""
        loader = self.for_source(source)
        try:
            documents = loader.load(source)
        except LoaderError:
            raise
        except Exception as exc:
            raise LoaderError(
                f"{loader.name} 加载 {source} 失败：{type(exc).__name__}: {exc}"
            ) from exc
        logger.debug("%s 从 %s 产出 %d 篇文档", loader.name, source, len(documents))
        return documents

    def load_many(self, sources: Iterable[Source]) -> list[Document]:
        """批量加载多个来源（文件 / 目录混合）。"""
        documents: list[Document] = []
        for source in sources:
            documents.extend(self.load(source))
        return documents

    @property
    def loaders(self) -> Sequence[DocumentLoader]:
        """已注册 Loader（只读视图）。"""
        return tuple(self._loaders)


class DirectoryLoader(DocumentLoader):
    """组合器：递归遍历目录，把每个文件交给 :class:`LoaderRegistry` 分发。

    不是新的「文档类型」，而是复用已有 Loader 的批量入口。未知后缀默认跳过
    （记录 debug 日志），``strict=True`` 时改为抛错，便于数据质量卡点。
    """

    def __init__(self, registry: LoaderRegistry, config: LoaderConfig | None = None,
                 strict: bool = False) -> None:
        self._registry = registry
        self.config = config or LoaderConfig()
        self.strict = strict

    @property
    def name(self) -> str:
        return "DirectoryLoader"

    def supports(self, source: Source) -> bool:
        try:
            return Path(source).is_dir()
        except OSError:
            return False

    def load(self, source: Source) -> list[Document]:
        root = Path(source)
        if not root.is_dir():
            raise LoaderError(f"不是目录：{root}")
        pattern = "**/*" if self.config.recursive else "*"
        documents: list[Document] = []
        # 排序保证不同机器上建索引顺序一致 -> 稳定 ID / 可复现
        for path in sorted(p for p in root.glob(pattern) if p.is_file()):
            # 只检查相对目录的部分：绝对路径里若恰有 .xxx 组件，不应把整个目录跳过
            if self._is_hidden(path.relative_to(root)):
                continue
            try:
                # 不经过 registry.for_source，避免 DirectoryLoader 自己被再次选中
                candidates = [loader for loader in reversed(self._registry.loaders)
                              if loader is not self and loader.supports(path)]
                if not candidates:
                    if self.strict:
                        raise UnsupportedSourceError(f"目录中存在未支持的文件：{path}")
                    logger.debug("跳过未支持的文件：%s", path)
                    continue
                documents.extend(candidates[0].load(path))
            except (LoaderError, UnsupportedSourceError) as exc:
                if self.strict:
                    raise
                logger.warning("跳过无法加载的文件 %s：%s", path, exc)
        logger.info("目录 %s 共加载 %d 篇文档", root, len(documents))
        return documents

    @staticmethod
    def _is_hidden(path: Path) -> bool:
        """跳过隐藏文件 / 常见缓存目录，避免把 .DS_Store 之类的东西建进索引。"""
        return any(part.startswith(".") for part in path.parts)


def build_default_loader_registry(config: LoaderConfig | None = None) -> LoaderRegistry:
    """构造装有 5 种内置 Loader + 目录组合器的注册表。

    延迟 import 各实现模块，避免 ``loaders`` 包自身出现循环导入。
    """
    from .csv_loader import CsvLoader
    from .html_loader import HtmlLoader
    from .json_loader import JsonLoader
    from .pdf_loader import PdfLoader
    from .text_loader import TextLoader

    cfg = config or LoaderConfig()
    registry = LoaderRegistry(cfg)
    registry.register(TextLoader(cfg))    # .txt/.md/.log
    registry.register(JsonLoader(cfg))    # .json/.jsonl/.ndjson
    registry.register(CsvLoader(cfg))     # .csv/.tsv
    registry.register(HtmlLoader(cfg))    # .html/.htm
    registry.register(PdfLoader(cfg))     # .pdf（可选依赖 pypdf）
    registry.register(DirectoryLoader(registry, cfg))
    return registry


__all__ = [
    "BaseFileLoader", "LoaderRegistry", "DirectoryLoader",
    "build_default_loader_registry", "read_text_file", "resolve_file",
    "base_metadata",
]
