"""Loader #3：表格数据（``.csv`` ``.tsv``）。

一行记录 = 一篇文档。表格的关键在于**列名是有语义的**，所以默认用
``"列名: 值"`` 逐列拼接，而不是逗号原串；也可通过 ``text_template``
精确控制，例如 ``"{product} 的价格是 {price} 元，属于 {category}"``。

处理细节：BOM（``utf-8-sig``）、引号内换行、缺失列、空行。
"""
from __future__ import annotations

import csv
import io
from string import Formatter
from typing import Any, Sequence

from .base import BaseFileLoader, base_metadata
from .._utils import get_logger
from ..exceptions import LoaderError
from .json_loader import _SafeDict
from ..models import Document

logger = get_logger(__name__)


class CsvLoader(BaseFileLoader):
    """CSV / TSV 加载器。

    Args:
        config: 通用读取配置。
        text_template: 可选的 ``str.format`` 模板，字段名为表头。
        content_columns: 可选，只保留这些列（按给定顺序）。
        delimiter: 手动指定分隔符；``None`` 时按后缀自动（``\\t`` 或 ``,``）。
    """

    suffixes = frozenset({".csv", ".tsv"})
    doc_type = "csv"

    def __init__(self, config=None, text_template: str | None = None,
                 content_columns: Sequence[str] | None = None,
                 delimiter: str | None = None) -> None:
        super().__init__(config)
        self.text_template = text_template
        self.content_columns = list(content_columns) if content_columns else None
        self.delimiter = delimiter

    def load(self, source) -> list[Document]:
        path, raw = self._read(source)
        # utf-8-sig 读取时 BOM 已被剥掉；其他编码下手动去掉残留 BOM
        raw = raw.lstrip("\ufeff")
        delimiter = self.delimiter or ("\t" if path.suffix.lower() == ".tsv" else ",")

        try:
            reader = csv.DictReader(io.StringIO(raw, newline=""), delimiter=delimiter)
            rows = list(reader)
        except csv.Error as exc:
            raise LoaderError(f"CSV 解析失败：{path}（分隔符 {delimiter!r}）：{exc}") from exc

        if not reader.fieldnames:
            raise LoaderError(f"CSV 缺少表头：{path}")
        # DictReader 会把多余列塞进 None 键，剔除以免污染正文
        documents: list[Document] = []
        columns = self.content_columns or [c for c in reader.fieldnames if c]
        for row_index, row in enumerate(rows):
            clean_row: dict[str, Any] = {
                str(k).strip(): (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
                if k is not None
            }
            text = self._to_text(clean_row, columns)
            if not text.strip():
                logger.debug("跳过空行：%s 第 %d 行", path, row_index + 2)
                continue
            metadata = base_metadata(
                path, "csv", row_index=row_index,
                columns=list(reader.fieldnames),
            )
            for key, value in clean_row.items():
                if value and len(str(value)) <= 200:
                    metadata.setdefault(key, value)
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def _to_text(self, row: dict[str, Any], columns: Sequence[str]) -> str:
        if self.text_template is not None:
            try:
                return Formatter().vformat(self.text_template, (), _SafeDict(row))
            except (ValueError, KeyError, IndexError) as exc:
                raise LoaderError(f"text_template 渲染失败：{exc}") from exc
        return "\n".join(
            f"{column}: {row.get(column, '')}" for column in columns
        )


__all__: Sequence[str] = ["CsvLoader"]
