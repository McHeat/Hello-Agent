"""Loader #4：HTML 网页（``.html`` ``.htm`` ``.xhtml``）。

用**标准库** ``html.parser`` 做正文抽取，不依赖 BeautifulSoup：

* 丢弃 ``script/style/noscript/nav/header/footer/aside/form/svg`` 等非正文标签；
* 块级标签转换成换行，``<li>`` 加项目符号，保留基本阅读结构；
* 抽取 ``<title>`` 与 ``<meta name="description">`` 写入 metadata；
* ``convert_charrefs=True`` 自动还原 ``&nbsp;`` 等实体。

抽取后仍会经过清洗流水线（去残留标签、压缩空白），两道防线互不重复。
"""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Sequence

from .base import BaseFileLoader, base_metadata
from .._utils import get_logger
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


class HtmlLoader(BaseFileLoader):
    """HTML 正文抽取加载器（一个文件 = 一篇文档）。"""

    suffixes = frozenset({".html", ".htm", ".xhtml"})
    doc_type = "html"

    def load(self, source) -> list[Document]:
        path, raw = self._read(source)
        extractor = _TextExtractor()
        try:
            extractor.feed(raw)
            extractor.close()
        except Exception as exc:  # HTMLParser 对畸形 HTML 一般很宽容，仍需兜底
            from ..exceptions import LoaderError
            raise LoaderError(f"HTML 解析失败：{path}：{exc}") from exc

        text = extractor.text
        if not text:
            logger.warning("HTML 未抽取到正文：%s", path)
        metadata = base_metadata(
            path, "html",
            title=extractor.title,
            description=extractor.description,
        )
        return [Document(page_content=text, metadata=metadata)]


__all__: Sequence[str] = ["HtmlLoader"]
