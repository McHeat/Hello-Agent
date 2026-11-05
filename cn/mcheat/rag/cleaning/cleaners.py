"""内置 Cleaner：每一步只做一件事，按顺序组合成清洗流水线。

清洗的目标不是「把文本改得好看」，而是**减少 embedding 与 BM25 的噪声**：
全角/半角不统一会让同一个词被切成两个 term；零宽字符与控制字符会污染
哈希；残留标签、页码、版权行会贡献无意义的相似度；重复文档会让 Top-K
被同一段内容占满（直接损害 Context Precision）。

所有 Cleaner 都注册进 ``CLEANER_REGISTRY``，配置项 ``cleaning.steps`` 用
名字引用；自定义 Cleaner 只要 ``@CLEANER_REGISTRY.register("xxx")`` 装饰
并把 ``"xxx"`` 加进 steps 即可，无需改动编排代码。
"""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Pattern, Sequence

from .._utils import content_hash, get_logger
from ..core.interfaces import Cleaner, TextCleaner
from ..core.registry import CLEANER_REGISTRY
from ..models import Document

logger = get_logger(__name__)


@CLEANER_REGISTRY.register("unicode_normalize")
class UnicodeNormalizer(TextCleaner):
    """Unicode 归一化（默认 NFKC）。

    NFKC 会把全角英数 ``ＲＡＧ`` 转成 ``RAG``、把兼容汉字统一，
    保证索引侧与查询侧的分词结果一致。
    """

    def __init__(self, form: str = "NFKC") -> None:
        if form not in {"NFC", "NFKC", "NFD", "NFKD"}:
            from ..exceptions import CleaningError
            raise CleaningError(f"不支持的 Unicode 归一化形式：{form!r}")
        self.form = form

    def clean_text(self, text: str) -> str:
        return unicodedata.normalize(self.form, text)


# 控制字符：保留 \n \t \r，其余 C0/C1、零宽字符、BOM、方向控制符全部剔除
_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"
    r"\u200b-\u200f\u202a-\u202e\u2060\u2066\u2067\ufeff]"
)


@CLEANER_REGISTRY.register("control_chars")
class ControlCharacterCleaner(TextCleaner):
    """剔除零宽字符、BOM、方向控制符与不可打印控制字符。"""

    def clean_text(self, text: str) -> str:
        return _CONTROL_RE.sub("", text)


# 只匹配「像标签」的片段：< 后跟字母或 /，避免误伤 ``a < b`` 这样的正文
_TAG_RE = re.compile(r"</?\s*[a-zA-Z][^<>]*>")


@CLEANER_REGISTRY.register("html_strip")
class HtmlTagCleaner(TextCleaner):
    """去掉残留 HTML 标签并还原实体。

    ``HtmlLoader`` 已做过正文抽取，这一步是给「从别处复制来的文本」兜底。
    """

    def clean_text(self, text: str) -> str:
        return html.unescape(_TAG_RE.sub(" ", text))


_SPACES_RE = re.compile(r"[ \t\u00a0\u3000]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_LINE_TRAILING_RE = re.compile(r"[ \t]+\n")


@CLEANER_REGISTRY.register("whitespace")
class WhitespaceCleaner(TextCleaner):
    """统一换行符、压缩空白、去掉行尾空格，保留段落间的空行。"""

    def clean_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _SPACES_RE.sub(" ", text)
        text = _LINE_TRAILING_RE.sub("\n", text)
        text = _MANY_NEWLINES_RE.sub("\n\n", text)
        return text.strip()


@CLEANER_REGISTRY.register("boilerplate")
class BoilerplateCleaner(TextCleaner):
    """按正则**逐行**删除样板文本（页码、版权声明等）。

    Args:
        patterns: 正则字符串列表，按整行匹配（``re.IGNORECASE``）。
            非法正则在构造期就报错，不会等到建索引 halfway 才炸。
    """

    def __init__(self, patterns: Sequence[str] | None = None) -> None:
        from ..exceptions import CleaningError

        self.patterns: list[Pattern[str]] = []
        for raw in patterns or []:
            try:
                self.patterns.append(re.compile(raw, re.IGNORECASE))
            except re.error as exc:
                raise CleaningError(f"非法的样板文本正则 {raw!r}：{exc}") from exc

    def clean_text(self, text: str) -> str:
        if not self.patterns:
            return text
        kept = [
            line for line in text.split("\n")
            if not any(pattern.match(line) for pattern in self.patterns)
        ]
        return "\n".join(kept)


@CLEANER_REGISTRY.register("drop_empty")
class DropEmptyCleaner(Cleaner):
    """清洗后正文过短的文档直接丢弃（返回 ``None``），不进入索引。

    空块 / 只剩标点的块会占用向量库名额，还可能在检索时被高分召回。
    """

    def __init__(self, min_length: int = 10) -> None:
        if min_length < 0:
            from ..exceptions import CleaningError
            raise CleaningError("min_length 不能为负")
        self.min_length = min_length

    def clean(self, document: Document) -> Document | None:
        return document if len(document.page_content.strip()) >= self.min_length else None


@CLEANER_REGISTRY.register("deduplicate")
class DeduplicateCleaner(Cleaner):
    """基于内容哈希的跨文档去重（同一批次内）。

    状态只在一次 ``run()`` 内有效：流水线开始时调用 :meth:`reset`，
    因此重复调用 ``index()`` 不会因历史哈希而误删文档。
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize
        self._seen: set[str] = set()

    def reset(self) -> None:
        """清空本轮去重状态（由 CleaningPipeline.run 调用）。"""
        self._seen.clear()

    def clean(self, document: Document) -> Document | None:
        text = document.page_content
        if self.normalize:
            text = " ".join(text.split())
        digest = content_hash(text)
        if digest in self._seen:
            return None
        self._seen.add(digest)
        return document


__all__: Sequence[str] = [
    "UnicodeNormalizer", "ControlCharacterCleaner", "HtmlTagCleaner",
    "WhitespaceCleaner", "BoilerplateCleaner", "DropEmptyCleaner",
    "DeduplicateCleaner",
]
