"""Document Loader 包：5 种格式 Loader + 目录组合器 + 注册表。

===========  ==================================  ============================
类型          后缀                                特点
===========  ==================================  ============================
TextLoader   .txt .log .md .markdown .rst        编码回退、front matter、标题
JsonLoader   .json .jsonl .ndjson               结构化字段 → 可控正文模板
CsvLoader    .csv .tsv                          一行一文档、列名语义拼接
HtmlLoader   .html .htm .xhtml                  标准库解析、剔除非正文标签
PdfLoader    .pdf                               可选 pypdf / 可注入解析器
===========  ==================================  ============================

``DirectoryLoader`` 是组合器（批量入口），不计入上述格式类型。
"""
from .base import (
    BaseFileLoader,
    DirectoryLoader,
    LoaderRegistry,
    base_metadata,
    build_default_loader_registry,
    read_text_file,
    resolve_file,
)
from .csv_loader import CsvLoader
from .html_loader import HtmlLoader
from .json_loader import JsonLoader
from .pdf_loader import PdfLoader
from .text_loader import TextLoader

__all__ = [
    "BaseFileLoader", "LoaderRegistry", "DirectoryLoader",
    "build_default_loader_registry", "read_text_file", "resolve_file",
    "base_metadata", "TextLoader", "JsonLoader", "CsvLoader",
    "HtmlLoader", "PdfLoader",
]
