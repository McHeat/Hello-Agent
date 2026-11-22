"""Loader #5：PDF（``.pdf``）—— 委托 LangChain + 可选依赖与错误处理范式。

默认页文本抽取委托 ``langchain_community.document_loaders.PyPDFLoader``；
``text_extractor`` 注入点无 LC 对应，传入自定义解析器时走原路径（因此本
Loader 在**没有 pypdf / langchain 的环境里依然可完整测试**）。LC 不可用时
回退 ``pypdf`` 直读；两者都缺时抛 :class:`DependencyMissingError`，消息带
安装命令与修复建议。

产出：一页 = 一篇 :class:`Document`，metadata 带 ``page_number``，
方便在答案里给出「第几页」的引用。
"""
from __future__ import annotations

import importlib.util
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
    """零依赖 pypdf 直读（LC 不可用时的回退路径），仅在真正加载时才 import。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DependencyMissingError(
            "pypdf", "PDF 文档加载", "或向 PdfLoader 注入 text_extractor"
        ) from exc
    try:
        reader = PdfReader(str(path))
        return [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise LoaderError(
            f"pypdf 读取 {path} 失败：{type(exc).__name__}: {exc}"
        ) from exc


def _lc_pdf_pages(path: Path) -> Sequence[str]:
    """委托 LC ``PyPDFLoader`` 抽取每页文本（延迟导入 + 依赖提示）。"""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as exc:
        raise DependencyMissingError(
            "langchain-community", "PDF 文档加载", "或注入 text_extractor"
        ) from exc
    try:
        documents = PyPDFLoader(str(path)).load()
    except ImportError as exc:
        raise DependencyMissingError(
            "pypdf", "PDF 文档加载", f"原始错误：{exc}"
        ) from exc
    except Exception as exc:
        raise LoaderError(
            f"LC PyPDFLoader 读取 {path} 失败：{type(exc).__name__}: {exc}"
        ) from exc
    return [doc.page_content for doc in documents]


class PdfLoader(BaseFileLoader):
    """PDF 加载器。

    Args:
        config: 通用读取配置（用于大小上限）。
        text_extractor: 可选的自定义页文本解析器；传入后完全不依赖第三方库。
        skip_empty_pages: 是否跳过抽取不到文本的页（扫描件常见）。
    """

    suffixes = frozenset({".pdf"})
    doc_type = "pdf"

    def __init__(self, config: LoaderConfig | None = None,
                 text_extractor: PdfTextExtractor | None = None,
                 skip_empty_pages: bool = True) -> None:
        super().__init__(config)
        self.text_extractor = text_extractor
        self.skip_empty_pages = skip_empty_pages

    def load(self, source) -> list[Document]:
        path = resolve_file(source, self.config)
        pages = (self._run_extractor(self.text_extractor, path)
                 if self.text_extractor is not None
                 else self._pages(path))

        if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
            raise LoaderError(
                f"页文本解析器必须返回「每页文本」序列，"
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

    @staticmethod
    def _pages(path: Path) -> Sequence[str]:
        """优先走 LangChain，失败回退 pypdf 直读。"""
        if importlib.util.find_spec("langchain_community") is not None:
            try:
                return _lc_pdf_pages(path)
            except (DependencyMissingError, LoaderError) as exc:
                logger.debug("%s 的 LC 解析失败，回退 pypdf 直读：%s", path, exc)
        return _pypdf_extractor(path)

    @staticmethod
    def _run_extractor(extractor: PdfTextExtractor, path: Path) -> Sequence[str]:
        try:
            return extractor(path)
        except (DependencyMissingError, LoaderError):
            raise
        except Exception as exc:
            raise LoaderError(
                f"PDF 解析失败：{path}：{type(exc).__name__}: {exc}"
            ) from exc


__all__: Sequence[str] = ["PdfLoader"]
