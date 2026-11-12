"""Loader #1：纯文本 / Markdown（``.txt`` ``.log`` ``.md`` ``.markdown`` ``.rst``）。

原始文本读取委托 ``langchain_community.document_loaders.TextLoader``；YAML front
matter、首个 ``#`` 标题与按空行分段保留为**自有后处理**，因此产出的
:class:`Document` 集合与元数据与重构前完全一致。

LC ``TextLoader`` 只接受单一 encoding，覆盖不了中文语料常见的 GBK/GB18030
编码回退链；故 LC 读取失败或依赖缺失时回退 :func:`read_text_file`。
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Sequence

from .base import BaseFileLoader, base_metadata, read_text_file, resolve_file
from .._utils import get_logger
from ..exceptions import DependencyMissingError, LoaderError
from ..models import Document

logger = get_logger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def _lc_read_text(path: Path, encoding: str) -> str:
    """委托 LC ``TextLoader`` 读入原始文本（延迟导入 + 依赖提示）。"""
    try:
        from langchain_community.document_loaders import TextLoader as LCTextLoader
    except ImportError as exc:
        raise DependencyMissingError(
            "langchain-community", "纯文本加载", "或回退零依赖读取"
        ) from exc
    try:
        documents = LCTextLoader(str(path), encoding=encoding).load()
    except UnicodeDecodeError:
        raise  # 交给调用方走编码回退链
    except Exception as exc:
        raise LoaderError(
            f"LC TextLoader 读取 {path} 失败：{type(exc).__name__}: {exc}"
        ) from exc
    return "\n".join(doc.page_content for doc in documents)


class TextLoader(BaseFileLoader):
    """纯文本加载器（LangChain 委托 + 自有后处理）。

    Args:
        config: 通用读取配置（编码回退、大小上限）。
        split_paragraphs: ``True`` 时按空行分段，每段一篇文档。
    """

    suffixes = frozenset({".txt", ".log", ".md", ".markdown", ".rst"})
    doc_type = "text"

    def __init__(self, config=None, split_paragraphs: bool = False) -> None:
        super().__init__(config)
        self.split_paragraphs = split_paragraphs

    def _read(self, source) -> tuple[Path, str]:
        """优先走 LangChain；依赖缺失 / 解码失败时回退零依赖读取。"""
        path = resolve_file(source, self.config)
        if importlib.util.find_spec("langchain_community") is not None:
            try:
                return path, _lc_read_text(path, self.config.encoding)
            except (
                DependencyMissingError, LoaderError, UnicodeDecodeError, OSError
            ) as exc:
                logger.debug("%s 的 LC 读取失败，回退零依赖读取：%s", path, exc)
        return path, read_text_file(path, self.config)

    def load(self, source) -> list[Document]:
        path, raw = self._read(source)
        is_markdown = path.suffix.lower() in {".md", ".markdown"}
        front_matter: dict[str, str] = {}
        text = raw

        if is_markdown:
            match = _FRONTMATTER_RE.match(text)
            if match:
                front_matter = self._parse_front_matter(match.group(1))
                text = text[match.end():]

        metadata = base_metadata(
            path,
            "markdown" if is_markdown else "text",
            title=self._extract_title(text, is_markdown),
            char_count=len(text),
            **front_matter,
        )

        if not self.split_paragraphs:
            return [Document(page_content=text, metadata=metadata)]

        # 每段单独成文：``paragraph_index`` 记入元数据；``Document.id`` 仍按
        # ``(source, page_content)`` 生成，保证重复建索引不变
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        documents: list[Document] = []
        for index, paragraph in enumerate(paragraphs):
            para_meta = dict(metadata)
            para_meta["paragraph_index"] = index
            documents.append(Document(page_content=paragraph, metadata=para_meta))
        return documents

    @staticmethod
    def _parse_front_matter(block: str) -> dict[str, str]:
        """解析极简 ``key: value`` front matter（不引入 PyYAML 依赖）。

        只保留简单标量；列表/嵌套 YAML 原样忽略，避免解析错误。
        """
        result: dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line or line.lstrip().startswith("#"):
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and value and not key.startswith("-"):
                result[f"fm_{key}"] = value
        return result

    @staticmethod
    def _extract_title(text: str, is_markdown: bool) -> str | None:
        if not is_markdown:
            return None
        match = _HEADING_RE.search(text)
        return match.group(1) if match else None


__all__: Sequence[str] = ["TextLoader"]
