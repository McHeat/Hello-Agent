"""Document Loader 包：9 种格式 Loader + 目录组合器 + 注册表。

===========  ==================================  ==================================
类型          后缀                                特点
===========  ==================================  ==================================
TextLoader   .txt .log .md .markdown .rst        LC TextLoader + 编码回退/front matter
JsonLoader   .json .jsonl .ndjson               LC JSONLoader(jq) + 可控正文模板
CsvLoader    .csv .tsv                          LC CSVLoader + 列名语义拼接
HtmlLoader   .html .htm .xhtml                  LC BSHTMLLoader + 结构化正文抽取
PdfLoader    .pdf                               LC PyPDFLoader / 可注入解析器
WordLoader   .docx .doc                         LC Docx2txtLoader
ExcelLoader  .xlsx .xls                         openpyxl，一行一文档
PptLoader    .pptx                              python-pptx，一页一文档
WebLoader    http:// https://                   LC WebBaseLoader，URL 在线抓取
===========  ==================================  ==================================

``DirectoryLoader`` 是组合器（批量入口），不计入上述格式类型。
第三方依赖一律延迟导入；未安装 LangChain 时内置格式自动回退零依赖实现。
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
from .langchain_loader import ExcelLoader, PptLoader, WebLoader, WordLoader
from .pdf_loader import PdfLoader
from .text_loader import TextLoader

__all__ = [
    "BaseFileLoader", "LoaderRegistry", "DirectoryLoader",
    "build_default_loader_registry", "read_text_file", "resolve_file",
    "base_metadata", "TextLoader", "JsonLoader", "CsvLoader",
    "HtmlLoader", "PdfLoader",
    "WordLoader", "ExcelLoader", "PptLoader", "WebLoader",
]
