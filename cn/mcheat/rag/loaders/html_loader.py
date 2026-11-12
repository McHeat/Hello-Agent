"""Loader #4：HTML 网页（``.html`` ``.htm`` ``.xhtml``）。

HTML 读取与初步抽取委托 ``langchain_community.document_loaders.BSHTMLLoader``；
正文的**结构化**（丢弃非正文标签、块级标签转换换行、``<li>`` 项目符号）保留为
自有后处理，因为 LC ``BSHTMLLoader`` 只做 ``soup.get_text()``，无法保证与
重构前一致的阅读结构。LC 结果用于 ``title`` 兜底与抽取为空时的诊断记录，
不改写正文，保证产出与重构前完全一致。

抽取后仍会经过清洗流水线（去残留标签、压缩空白），两道防线互不重复。
"""
from __future__ import annotations

import html
import importlib.util
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Sequence

from .base import BaseFileLoader, base_metadata, resolve_file
from .._utils import get_logger
from ..exceptions import DependencyMissingError, LoaderError
from ..models import Document

logger = get_logger(__name__)

_SKIP_TAGS = frozenset({
    "script", "style", "noscript", "nav", "header", "footer", "aside",
    "form", "svg", "template", "iframe", "button", "select", "option",
})
_BLOCK_TAGS = frozenset({
    "p", "div", "section", "article", "main", "h1", "h2", "h3", "h4",
    "h5", "h6", "blockquote", "pre", "ul", "ol", "table", "tr", "br",
    "hr", "dl", "dt", "dd", "figure", "figcaption",
})
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")


class _TextExtractor(HTMLParser):
    """把 HTML 转成「带换行结构的纯文本」。

    用 ``_skip_stack`` 而不是布尔标记，是为了正确处理 skip 标签嵌套：
    ``<nav><div><nav>...</nav></div></nav>`` 不会提前退出跳过状态。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._skip_stack: list[str] = []
        self._in_title = False
        self.description: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_stack.append(tag)
            return
        if self._skip_stack:
            return
        attrs_dict = {k: (v or "") for k, v in attrs}
        if tag == "meta" and attrs_dict.get("name", "").lower() == "description":
            self.description = attrs_dict.get("content", "").strip() or None
        if tag == "title":
            self._in_title = True
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # <br/>、<hr/> 等自闭合标签
        self.handle_starttag(tag, attrs)
        if tag not in _SKIP_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._skip_stack:
            # 支持 skip 标签嵌套：遇到栈中任意一层就把它连同内层一起弹出
            if tag in self._skip_stack:
                index = len(self._skip_stack) - 1 - self._skip_stack[::-1].index(tag)
                del self._skip_stack[index:]
            return
        if tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_stack or not data:
            return
        if self._in_title:
            self._title_parts.append(data)
        else:
            self._parts.append(data)

    @property
    def title(self) -> str | None:
        value = " ".join("".join(self._title_parts).split())
        return value or None

    @property
    def text(self) -> str:
        raw = "".join(self._parts)
        raw = html.unescape(raw)
        raw = _TRAILING_SPACES.sub("\n", raw)
        raw = _MULTI_BLANK_LINES.sub("\n\n", raw)
        return raw.strip()


def _lc_html(path: Path) -> tuple[str, str | None]:
    """委托 LC ``BSHTMLLoader`` 做初步抽取（延迟导入 + 依赖提示）。"""
    try:
        from langchain_community.document_loaders import BSHTMLLoader
    except ImportError as exc:
        raise DependencyMissingError("langchain-community", "HTML 加载") from exc
    try:
        documents = BSHTMLLoader(str(path)).load()
    except ImportError as exc:
        raise DependencyMissingError(
            "beautifulsoup4", "HTML 加载", "BSHTMLLoader 还可能需要 lxml"
        ) from exc
    except Exception as exc:
        raise LoaderError(
            f"LC BSHTMLLoader 解析 {path} 失败：{type(exc).__name__}: {exc}"
        ) from exc
    text = "\n".join(doc.page_content for doc in documents)
    title = documents[0].metadata.get("title") if documents else None
    return text, title


class HtmlLoader(BaseFileLoader):
    """HTML 正文抽取加载器（一个文件 = 一篇文档）。"""

    suffixes = frozenset({".html", ".htm", ".xhtml"})
    doc_type = "html"

    def load(self, source) -> list[Document]:
        path = resolve_file(source, self.config)
        lc_text, lc_title = "", None
        if importlib.util.find_spec("langchain_community") is not None:
            try:
                lc_text, lc_title = _lc_html(path)
            except (DependencyMissingError, LoaderError) as exc:
                logger.debug("%s 的 LC 抽取失败，使用标准库解析：%s", path, exc)

        _, raw = self._read(source)
        extractor = _TextExtractor()
        try:
            extractor.feed(raw)
            extractor.close()
        except Exception as exc:  # HTMLParser 对畸形 HTML 一般很宽容，仍需兜底
            raise LoaderError(f"HTML 解析失败：{path}：{exc}") from exc

        # 正文只用自有结构化抽取，避免 LC get_text() 把 script/nav 原文带回，
        # 保证与重构前「正文为空即返回空」的行为完全一致；LC 结果仅作诊断。
        text = extractor.text
        if not text:
            if lc_text.strip():
                logger.debug("%s 自有抽取为空（LC 抽取到 %d 字符），保持空正文",
                             path, len(lc_text))
            logger.warning("HTML 未抽取到正文：%s", path)
        metadata = base_metadata(
            path, "html",
            title=extractor.title or lc_title,
            description=extractor.description,
        )
        return [Document(page_content=text, metadata=metadata)]


__all__: Sequence[str] = ["HtmlLoader"]
