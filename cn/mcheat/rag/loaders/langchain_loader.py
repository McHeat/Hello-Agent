"""新增格式加载层：Word / Excel / PowerPoint / 网页 URL。

统一约定（见 ``doc/loader_refactor_contract.md``）：

* 全部实现 :class:`DocumentLoader` 的 ``name`` / ``supports`` / ``load``；
* 第三方依赖**延迟导入**，缺依赖抛 :class:`DependencyMissingError`，
  任意未知异常包装成带来源的 :class:`LoaderError`；
* 元数据经 :func:`base_metadata` 归一 ``source``，保证稳定 ID 幂等；
* ``WebLoader`` 处理 http/https 来源，不走 :func:`resolve_file`。

Excel / PPT 采用 ``openpyxl`` / ``python-pptx`` 轻量路径：LangChain 的
``UnstructuredExcelLoader`` 等需要 ``unstructured``（依赖树过重），
故按「无法直接映射 → 轻量替代方案」处理，语义与 CsvLoader / PdfLoader 对齐。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

from .base import BaseFileLoader, base_metadata, resolve_file
from .._utils import get_logger
from ..config import LoaderConfig
from ..core.interfaces import DocumentLoader, Source
from ..exceptions import DependencyMissingError, LoaderError
from ..models import Document

logger = get_logger(__name__)


def _require_module(module: str, package: str, feature: str,
                    extra: str = "") -> None:
    """探测依赖但不执行包代码；缺失时给出带安装命令的提示。"""
    if importlib.util.find_spec(module) is None:
        raise DependencyMissingError(package, feature, extra)


def _merge_lc_metadata(metadata: dict[str, Any],
                       lc_document: Any) -> dict[str, Any]:
    """合并 LC metadata，但不允许覆盖已归一的基础溯源字段。"""
    for key, value in dict(getattr(lc_document, "metadata", {}) or {}).items():
        if value is not None and key not in metadata:
            metadata[key] = value
    return metadata


def _to_documents(lc_documents: Sequence[Any], path: Path,
                  doc_type: str, **extra) -> list[Document]:
    """LC Document → 框架 Document，基础元数据由 ``base_metadata`` 统一补齐。"""
    documents: list[Document] = []
    for lc_document in lc_documents:
        text = getattr(lc_document, "page_content", "") or ""
        metadata = _merge_lc_metadata(
            base_metadata(path, doc_type, **extra), lc_document
        )
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


class WordLoader(BaseFileLoader):
    """Word 文档（``.docx``，``.doc`` 尽力支持）。"""

    suffixes = frozenset({".docx", ".doc"})
    doc_type = "word"

    def load(self, source: Source) -> list[Document]:
        path = resolve_file(source, self.config)
        _require_module("langchain_community", "langchain-community",
                        "Word 文档加载", "或把 .doc 转换为 .docx")
        _require_module("docx2txt", "docx2txt", "Word 文档加载",
                        "或把 .doc 转换为 .docx")
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            lc_documents = Docx2txtLoader(str(path)).load()
        except Exception as exc:
            raise LoaderError(
                f"Word 加载失败：{path}：{type(exc).__name__}: {exc}"
            ) from exc
        return _to_documents(lc_documents, path, "word")


class ExcelLoader(BaseFileLoader):
    """Excel 工作簿（``.xlsx`` / ``.xls``），一行 = 一篇文档。"""

    suffixes = frozenset({".xlsx", ".xls"})
    doc_type = "excel"

    def __init__(self, config: LoaderConfig | None = None,
                 sheet_name: str | None = None) -> None:
        super().__init__(config)
        self.sheet_name = sheet_name

    def load(self, source: Source) -> list[Document]:
        path = resolve_file(source, self.config)
        _require_module("openpyxl", "openpyxl", "Excel 文档加载",
                        "或把 .xls 另存为 .xlsx")
        try:
            from openpyxl import load_workbook
            workbook = load_workbook(str(path), read_only=True, data_only=True)
            try:
                sheet = (workbook[self.sheet_name] if self.sheet_name
                         else workbook.active)
                rows = [tuple(row) for row in sheet.iter_rows(values_only=True)]
                sheet_title = sheet.title
            finally:
                workbook.close()
        except Exception as exc:
            raise LoaderError(
                f"Excel 加载失败：{path}：{type(exc).__name__}: {exc}"
            ) from exc

        if not rows:
            return []
        headers = [
            str(cell).strip() if cell is not None else f"col_{index}"
            for index, cell in enumerate(rows[0])
        ]
        documents: list[Document] = []
        for row_index, row in enumerate(rows[1:]):
            values = ["" if cell is None else str(cell) for cell in row]
            text = "\n".join(
                f"{header}: {value}" for header, value in zip(headers, values)
            )
            if not text.strip():
                continue
            metadata = base_metadata(
                path, "excel", row_index=row_index,
                sheet=sheet_title, columns=list(headers),
            )
            documents.append(Document(page_content=text, metadata=metadata))
        return documents


class PptLoader(BaseFileLoader):
    """PowerPoint 幻灯片（``.pptx``），一页 = 一篇文档。"""

    suffixes = frozenset({".pptx"})
    doc_type = "pptx"

    def load(self, source: Source) -> list[Document]:
        path = resolve_file(source, self.config)
        _require_module("pptx", "python-pptx", "PowerPoint 文档加载",
                        "请执行 pip install python-pptx")
        try:
            from pptx import Presentation
            presentation = Presentation(str(path))
            slides: list[str] = []
            for slide in presentation.slides:
                parts: list[str] = []
                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False):
                        text = "\n".join(
                            paragraph.text
                            for paragraph in shape.text_frame.paragraphs
                        ).strip()
                        if text:
                            parts.append(text)
                slides.append("\n".join(parts))
        except Exception as exc:
            raise LoaderError(
                f"PowerPoint 加载失败：{path}：{type(exc).__name__}: {exc}"
            ) from exc

        documents: list[Document] = []
        for index, text in enumerate(slides):
            if not text.strip():
                continue
            metadata = base_metadata(
                path, "pptx", page_number=index + 1, total_pages=len(slides)
            )
            documents.append(Document(page_content=text, metadata=metadata))
        return documents


class WebLoader(DocumentLoader):
    """网页 URL（``http://`` / ``https://``），委托 LC ``WebBaseLoader``。

    不是文件路径，因此不继承 :class:`BaseFileLoader`，也不调用
    :func:`resolve_file`。
    """

    def __init__(self, config: LoaderConfig | None = None) -> None:
        self.config = config or LoaderConfig()

    def supports(self, source: Source) -> bool:
        return str(source).startswith(("http://", "https://"))

    def load(self, source: Source) -> list[Document]:
        url = str(source)
        if not self.supports(url):
            raise LoaderError(f"WebLoader 只支持 http/https 来源：{url}")
        _require_module("langchain_community", "langchain-community", "网页加载")
        _require_module("bs4", "beautifulsoup4", "网页加载")
        try:
            from langchain_community.document_loaders import WebBaseLoader
            lc_documents = WebBaseLoader(url).load()
        except Exception as exc:
            raise LoaderError(
                f"WebBaseLoader 加载 {url} 失败：{type(exc).__name__}: {exc}"
            ) from exc

        documents: list[Document] = []
        for index, lc_document in enumerate(lc_documents):
            metadata: dict[str, Any] = {
                "source": url,
                "filename": "",
                "suffix": "",
                "doc_type": "web",
                "page_number": index + 1,
                "total_pages": len(lc_documents),
            }
            _merge_lc_metadata(metadata, lc_document)
            documents.append(Document(
                page_content=getattr(lc_document, "page_content", "") or "",
                metadata=metadata,
            ))
        return documents


__all__: Sequence[str] = ["WordLoader", "ExcelLoader", "PptLoader", "WebLoader"]
