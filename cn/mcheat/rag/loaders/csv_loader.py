"""Loader #3：表格数据（``.csv`` ``.tsv``）。

行解析委托 ``langchain_community.document_loaders.CSVLoader``；其产出的
``列名: 值`` 行文本反解为行字典后，仍走自有的 ``text_template`` /
``content_columns`` 渲染逻辑，保证正文与 metadata 与重构前一致。

一行记录 = 一篇文档。表格的关键在于**列名是有语义的**，所以默认用
``"列名: 值"`` 逐列拼接，而不是逗号原串；也可通过 ``text_template``
精确控制，例如 ``"{product} 的价格是 {price} 元"``。

处理细节：BOM、引号内换行、缺失列、空行、tsv 分隔符与编码回退。
"""
from __future__ import annotations

import csv
import importlib.util
import io
from pathlib import Path
from string import Formatter
from typing import Any, Sequence

from .base import BaseFileLoader, base_metadata, resolve_file
from .._utils import get_logger
from ..exceptions import DependencyMissingError, LoaderError
from ..models import Document

logger = get_logger(__name__)


class _SafeDict(dict):
    """``format_map`` 用的容错字典：缺失键渲染成空串而不是抛 KeyError。"""

    def __missing__(self, key: str) -> str:
        return ""


def _lc_read_rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    """委托 LC ``CSVLoader`` 解析，并把行文本反解为字段字典。"""
    try:
        from langchain_community.document_loaders import CSVLoader
    except ImportError as exc:
        raise DependencyMissingError("langchain-community", "CSV 加载") from exc
    try:
        lc_documents = CSVLoader(
            file_path=str(path), csv_args={"delimiter": delimiter}
        ).load()
    except Exception as exc:
        raise LoaderError(
            f"LC CSVLoader 解析 {path} 失败：{type(exc).__name__}: {exc}"
        ) from exc

    rows: list[dict[str, str]] = []
    for lc_document in lc_documents:
        row: dict[str, str] = {}
        last_key: str | None = None
        for line in lc_document.page_content.split("\n"):
            key, separator, value = line.partition(": ")
            if separator and key.strip():
                last_key = key.strip().lstrip("\ufeff")
                row[last_key] = value.strip()
            elif last_key is not None:
                # LC 把引号内换行留在值里，这里拼回同一字段避免丢数据
                row[last_key] += "\n" + line
        rows.append(row)
    return rows


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
        path = resolve_file(source, self.config)
        delimiter = self.delimiter or ("\t" if path.suffix.lower() == ".tsv" else ",")
        rows, all_columns = self._rows(source, path, delimiter)
        columns = self.content_columns or [c for c in all_columns if c]

        documents: list[Document] = []
        for row_index, row in enumerate(rows):
            # DictReader 会把多余列塞进 None 键，剔除以免污染正文
            clean_row: dict[str, Any] = {
                str(k).strip(): (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
                if k is not None
            }
            try:
                text = self._to_text(clean_row, columns)
            except LoaderError as exc:
                raise LoaderError(f"{path}：{exc}") from exc
            if not text.strip():
                logger.debug("跳过空行：%s 第 %d 行", path, row_index + 2)
                continue
            metadata = base_metadata(
                path, "csv", row_index=row_index,
                columns=list(all_columns),
            )
            for key, value in clean_row.items():
                if value and len(str(value)) <= 200:
                    metadata.setdefault(key, value)
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    # -- 行解析 -------------------------------------------------------------

    def _rows(self, source, path: Path,
              delimiter: str) -> tuple[list[dict], list[str]]:
        """优先走 LangChain；依赖缺失 / 解析失败回退标准库（含编码回退）。"""
        if importlib.util.find_spec("langchain_community") is not None:
            try:
                rows = _lc_read_rows(path, delimiter)
                all_columns = list(rows[0].keys()) if rows else []
                return rows, all_columns
            except (DependencyMissingError, LoaderError) as exc:
                logger.debug("%s 的 LC 解析失败，回退标准库解析：%s", path, exc)
        return self._legacy_rows(source, path, delimiter)

    def _legacy_rows(self, source, path: Path,
                     delimiter: str) -> tuple[list[dict], list[str]]:
        _, raw = self._read(source)
        raw = raw.lstrip("\ufeff")
        try:
            reader = csv.DictReader(io.StringIO(raw, newline=""), delimiter=delimiter)
            rows = list(reader)
        except csv.Error as exc:
            raise LoaderError(
                f"CSV 解析失败：{path}（分隔符 {delimiter!r}）：{exc}"
            ) from exc
        if not reader.fieldnames:
            raise LoaderError(f"CSV 缺少表头：{path}")
        return rows, [c for c in reader.fieldnames if c]

    # -- 行转文本 -----------------------------------------------------------

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
