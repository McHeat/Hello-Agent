"""Loader #1：纯文本 / Markdown（``.txt`` ``.log`` ``.md`` ``.markdown`` ``.rst``）。

最朴素的形态：一个文件 = 一篇文档。额外做了两件轻量增强：

* 识别 Markdown 的 YAML front matter（``---`` 包裹），把简单键值写进 metadata；
* 取首个 ``# 标题`` 作为 ``title``，方便后续按来源过滤与展示引用。

可选 ``split_paragraphs=True`` 把一个文件按空行拆成多篇文档（适合 FAQ 文本）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from .base import BaseFileLoader, base_metadata
from ..models import Document

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


class TextLoader(BaseFileLoader):
    """纯文本加载器。

    Args:
        config: 通用读取配置（编码回退、大小上限）。
        split_paragraphs: ``True`` 时按空行分段，每段一篇文档。
    """

    suffixes = frozenset({".txt", ".log", ".md", ".markdown", ".rst"})
    doc_type = "text"

    def __init__(self, config=None, split_paragraphs: bool = False) -> None:
        super().__init__(config)
        self.split_paragraphs = split_paragraphs

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

        # 每段单独成文：``index`` 参与稳定 ID，保证重复建索引不变
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
