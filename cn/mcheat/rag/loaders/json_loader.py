"""Loader #2：结构化 JSON / JSONL（``.json`` ``.jsonl`` ``.ndjson``）。

与纯文本不同，这里要处理「结构 → 文本」的转换，是 RAG 落地最常见的坑之一：
直接 ``str(dict)`` 会把键名、引号、嵌套结构一起塞进 embedding，噪声很大。

支持的形态：

* ``.jsonl`` / ``.ndjson``：每行一个 JSON 对象 → 一篇文档；
* ``.json`` 顶层为 list → 每个元素一篇文档；
* ``.json`` 顶层为 dict → 若含 ``data/items/records/documents`` 列表则展开，
  否则整体当作一篇文档。

``text_template`` 可用 ``"{title}：{content}"`` 精确控制进入正文的字段；
不传时按 ``key: value`` 逐行拼接，嵌套结构用紧凑 JSON 表示。
"""
from __future__ import annotations

import json
from string import Formatter
from typing import Any, Sequence

from .base import BaseFileLoader, base_metadata
from .._utils import get_logger
from ..exceptions import LoaderError
from ..models import Document

logger = get_logger(__name__)

_LIST_FIELDS = ("data", "items", "records", "documents", "results")


class _SafeDict(dict):
    """``format_map`` 用的容错字典：缺失键渲染成空串而不是抛 KeyError。"""

    def __missing__(self, key: str) -> str:
        return ""


class JsonLoader(BaseFileLoader):
    """JSON / JSONL 加载器。

    Args:
        config: 通用读取配置。
        text_template: 可选的 ``str.format`` 模板。
        content_fields: 可选，只保留这些字段（按给定顺序）。
    """

    suffixes = frozenset({".json", ".jsonl", ".ndjson"})
    doc_type = "json"

    def __init__(self, config=None, text_template: str | None = None,
                 content_fields: Sequence[str] | None = None) -> None:
        super().__init__(config)
        self.text_template = text_template
        self.content_fields = list(content_fields) if content_fields else None

    def load(self, source) -> list[Document]:
        path, raw = self._read(source)
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            records = self._parse_jsonl(raw, path)
        else:
            records = self._parse_json(raw, path)

        documents: list[Document] = []
        for index, record in enumerate(records):
            text = self._to_text(record)
            if not text.strip():
                logger.debug("跳过空记录：%s#%d", path, index)
                continue
            metadata = base_metadata(path, "json", record_index=index)
            # 把短标量字段并入 metadata，供在线检索按 tenant/category 等过滤
            for key, value in self._scalar_items(record):
                if key not in {"source", "filename", "suffix", "doc_type"}:
                    metadata.setdefault(key, value)
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    # -- 解析 ---------------------------------------------------------------

    @staticmethod
    def _parse_jsonl(raw: str, path) -> list[Any]:
        records: list[Any] = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise LoaderError(
                    f"JSONL 解析失败：{path} 第 {line_no} 行：{exc.msg}"
                ) from exc
        return records

    @staticmethod
    def _parse_json(raw: str, path) -> list[Any]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LoaderError(
                f"JSON 解析失败：{path} 第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}"
            ) from exc
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for field in _LIST_FIELDS:
                nested = payload.get(field)
                if isinstance(nested, list):
                    logger.debug("%s 展开列表字段 %r（%d 条）", path, field, len(nested))
                    return nested
            return [payload]
        # 标量顶层：包成一条，避免静默丢数据
        return [{"value": payload}]

    # -- 结构转文本 ---------------------------------------------------------

    def _to_text(self, record: Any) -> str:
        if not isinstance(record, dict):
            return str(record)
        if self.text_template is not None:
            try:
                return Formatter().vformat(self.text_template, (), _SafeDict(record))
            except (ValueError, KeyError, IndexError) as exc:
                raise LoaderError(f"text_template 渲染失败：{exc}") from exc

        fields = self.content_fields or list(record.keys())
        lines: list[str] = []
        for key in fields:
            if key not in record:
                continue
            value = record[key]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _scalar_items(record: Any) -> list[tuple[str, Any]]:
        """挑出可安全放进 metadata 的短标量字段。"""
        if not isinstance(record, dict):
            return []
        result: list[tuple[str, Any]] = []
        for key, value in record.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                text = str(value)
                if len(text) <= 200:
                    result.append((str(key), value))
        return result


__all__: Sequence[str] = ["JsonLoader"]
