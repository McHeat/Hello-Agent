"""Loader #5：PDF（``.pdf``）—— 展示**可选依赖**与错误处理的写法。

核心框架零第三方依赖，但 PDF 是二进制格式，必须借助解析库。这里：

* 默认走 ``pypdf``，采用**延迟导入**：不装 pypdf 也能正常 import 整个 RAG 包；
* 缺依赖时抛 :class:`DependencyMissingError`，消息里直接给出安装命令，
  并说明可退回到零依赖的默认配置；
* 通过 ``text_extractor`` 可注入任意解析器（``pymupdf`` / OCR / 测试替身），
  因此本 Loader 在**没有 pypdf 的环境里依然可完整测试**，也是「可插拔」
  在 Loader 层的体现。

产出：一页 = 一篇 :class:`Document`，metadata 带 ``page_number``，
方便在答案里给出「第几页」的引用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .base import BaseFileLoader, base_metadata, resolve_file
from .._utils import get_logger
from ..config import LoaderConfig
from ..exceptions import DependencyMissingError, LoaderError
from ..models import Document

logger = get_logger(__name__)

# 注入点：(pdf_path) -> 每页文本列表
PdfTextExtractor = Callable[[Path], Sequence[str]]


def _pypdf_extractor(path: Path) -> Sequence[str]:
    """默认的 pypdf 解析器；仅在真正加载 .pdf 时才 import。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DependencyMissingError(
            "pypdf", "PDF 文档加载", "或向 PdfLoader 注入 text_extractor"
        ) from exc
    try:
        reader = PdfReader(str(path))
        return [
            (page.extract_text() or "") for page in reader.pages
        ]
    except Exception as exc:
        raise LoaderError(f"pypdf 读取 {path} 失败：{type(exc).__name__}: {exc}") from exc


class PdfLoader(BaseFileLoader):
    """PDF 加载器。

    Args:
        config: 通用读取配置（用于大小上限）。
        text_extractor: 可选的自定义页文本解析器；传入后完全不依赖 pypdf。
        skip_empty_pages: 是否跳过抽取不到文本的页（扫描件常见）。
    """

    suffixes = frozenset({".pdf"})
    doc_type = "pdf"

    def __init__(self, config: LoaderConfig | None = None,
                 text_extractor: PdfTextExtractor | None = None,
                 skip_empty_pages: bool = True) -> None:
        super().__init__(config)
        self.text_extractor = text_extractor or _pypdf_extractor
        self.skip_empty_pages = skip_empty_pages

    def load(self, source) -> list[Document]:
        path = resolve_file(source, self.config)
        try:
            pages = self.text_extractor(path)
        except (DependencyMissingError, LoaderError):
            raise
        except Exception as exc:
            raise LoaderError(
                f"PDF 解析失败：{path}：{type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
            raise LoaderError(
                f"text_extractor 必须返回「每页文本」序列，"
                f"收到 {type(pages).__name__}：{path}"
            )

        documents: list[Document] = []
        for page_index, text in enumerate(pages):
            if not isinstance(text, str):
                raise LoaderError(f"第 {page_index + 1} 页不是字符串：{path}")
            if self.skip_empty_pages and not text.strip():
                logger.debug("跳过无文本页：%s 第 %d 页", path, page_index + 1)
                continue
            metadata = base_metadata(
                path, "pdf",
                page_number=page_index + 1,
                total_pages=len(pages),
            )
            documents.append(Document(page_content=text, metadata=metadata))
        return documents


__all__: Sequence[str] = ["PdfLoader"]
