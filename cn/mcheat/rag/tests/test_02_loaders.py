"""Loader 基线断言（15 个，逐字不变）+ 扩展格式与委托层补充用例。"""
from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from cn.mcheat.rag.config import LoaderConfig
from cn.mcheat.rag.core.interfaces import DocumentLoader
from cn.mcheat.rag.exceptions import (
    DependencyMissingError,
    LoaderError,
    UnsupportedSourceError,
)
from cn.mcheat.rag.loaders import (
    CsvLoader,
    DirectoryLoader,
    ExcelLoader,
    HtmlLoader,
    JsonLoader,
    LoaderRegistry,
    PdfLoader,
    PptLoader,
    TextLoader,
    WebLoader,
    WordLoader,
    build_default_loader_registry,
    read_text_file,
)
from cn.mcheat.rag.models import Document


class TestLoaders(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = build_default_loader_registry()

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, content, encoding: str = "utf-8") -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes)
                         else content.encode(encoding))
        return path

    def test_text_loader_and_markdown_front_matter(self):
        path = self._write("notes.md", "---\ntitle: 测试标题\ncategory: rag\n---\n\n# 一级标题\n\n正文内容足够长可以进索引。")
        docs = TextLoader().load(path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["doc_type"], "markdown")
        self.assertEqual(docs[0].metadata["title"], "一级标题")
        self.assertEqual(docs[0].metadata["fm_title"], "测试标题")
        self.assertNotIn("---", docs[0].page_content)

    def test_text_loader_split_paragraphs(self):
        path = self._write("p.txt", "第一段内容足够长。\n\n第二段内容也足够长。\n\n第三段内容同样长。")
        docs = TextLoader(split_paragraphs=True).load(path)
        self.assertEqual(len(docs), 3)
        self.assertEqual([d.metadata["paragraph_index"] for d in docs], [0, 1, 2])
        self.assertEqual(len({d.id for d in docs}), 3)  # 稳定且不重复

    def test_json_loader_list_and_template(self):
        path = self._write("data.json", '[{"title": "甲", "content": "内容甲"}, {"title": "乙", "content": "内容乙"}]')
        docs = JsonLoader().load(path)
        self.assertEqual(len(docs), 2)
        self.assertIn("title: 甲", docs[0].page_content)
        templated = JsonLoader(text_template="{title}：{content}（{missing}）").load(path)
        self.assertEqual(templated[0].page_content, "甲：内容甲（）")

    def test_json_loader_jsonl_and_nested_list(self):
        path = self._write("rows.jsonl", '{"q":"问题一","a":"答案一"}\n\n{"q":"问题二","a":"答案二"}\n')
        docs = JsonLoader().load(path)
        self.assertEqual(len(docs), 2)
        nested = self._write("nested.json", '{"data": [{"x": 1}, {"x": 2}]}')
        self.assertEqual(len(JsonLoader().load(nested)), 2)

    def test_csv_loader_column_semantics(self):
        path = self._write("p.csv", "name,price\nA,10\nB,20\n")
        docs = CsvLoader().load(path)
        self.assertEqual(len(docs), 2)
        self.assertIn("name: A", docs[0].page_content)
        self.assertEqual(docs[1].metadata["row_index"], 1)
        templated = CsvLoader(text_template="{name} 价格 {price}").load(path)
        self.assertEqual(templated[0].page_content, "A 价格 10")

    def test_tsv_loader_delimiter(self):
        path = self._write("p.tsv", "name\tprice\nA\t10\n")
        docs = CsvLoader().load(path)
        self.assertIn("price: 10", docs[0].page_content)

    def test_html_loader_strips_script_style_nav(self):
        path = self._write(
            "page.html",
            "<html><head><title>标题页</title><style>BODY{}</style>"
            "<script>alert('x')</script></head><body><nav>导航文字</nav>"
            "<main><h1>主标题</h1><p>第一段正文。</p><ul><li>要点一</li><li>要点二</li></ul>"
            "</main><footer>版权所有</footer></body></html>",
        )
        docs = HtmlLoader().load(path)
        self.assertEqual(len(docs), 1)
        text = docs[0].page_content
        self.assertEqual(docs[0].metadata["title"], "标题页")
        self.assertIn("第一段正文", text)
        self.assertIn("- 要点一", text)
        self.assertNotIn("alert", text)
        self.assertNotIn("导航文字", text)
        self.assertNotIn("BODY{}", text)

    def test_pdf_loader_with_injected_extractor(self):
        path = self.root / "fake.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        loader = PdfLoader(text_extractor=lambda p: ["第一页正文", "第二页正文", "   "])
        docs = loader.load(path)
        self.assertEqual(len(docs), 2)  # 空白页被跳过
        self.assertEqual(docs[0].metadata["page_number"], 1)
        self.assertEqual(docs[1].metadata["total_pages"], 3)
        self.assertEqual(docs[0].metadata["doc_type"], "pdf")

    def test_dependency_error_has_install_hint(self):
        error = DependencyMissingError("pypdf", "PDF 加载")
        self.assertIn("pip install pypdf", str(error))

    def test_directory_loader_dispatches_all_formats(self):
        self._write("a.txt", "纯文本内容足够长可以进索引。")
        self._write("b.md", "# 标题\n\nMarkdown 正文足够长。")
        self._write("c.jsonl", '{"x": "JSON 正文足够长"}\n')
        self._write("d.csv", "col\n值足够长可以进索引\n")
        self._write("e.html", "<html><body><p>HTML 正文足够长</p></body></html>")
        self._write("ignored.bin", b"\x00\x01")
        docs = self.registry.load(self.root)
        self.assertEqual(len(docs), 5)
        types = {d.metadata["doc_type"] for d in docs}
        self.assertEqual(types, {"text", "markdown", "json", "csv", "html"})

    def test_unsupported_suffix_raises(self):
        path = self._write("x.unknown", "内容")
        with self.assertRaises(UnsupportedSourceError) as ctx:
            self.registry.load(path)
        self.assertIn(".unknown", str(ctx.exception))

    def test_registry_override_last_wins(self):
        class UpperLoader(DocumentLoader):
            def supports(self, source):
                return Path(source).suffix == ".txt"

            def load(self, source):
                return [Document(page_content=Path(source).read_text().upper(),
                                 metadata={"source": str(source)})]

        registry = LoaderRegistry()
        registry.register(TextLoader())
        registry.register(UpperLoader())
        path = self._write("a.txt", "hello")
        self.assertEqual(registry.load(path)[0].page_content, "HELLO")

    def test_encoding_fallback(self):
        path = self._write("gbk.txt", "中文编码回退测试内容", encoding="gbk")
        cfg = LoaderConfig(fallback_encodings=["gb18030"])
        self.assertIn("中文编码回退测试内容", read_text_file(path, cfg))

    def test_directory_loader_strict_mode(self):
        self._write("x.bin", b"\x00")
        strict = DirectoryLoader(self.registry, strict=True)
        with self.assertRaises(UnsupportedSourceError):
            strict.load(self.root)

    def test_hidden_files_are_skipped(self):
        (self.root / ".hidden").mkdir()
        (self.root / ".hidden" / "a.txt").write_text("隐藏文件", encoding="utf-8")
        docs = self.registry.load(self.root)
        self.assertEqual(len(docs), 0)


class TestLangChainLoaders(unittest.TestCase):
    """扩展格式与委托层补充用例（无网络请求；缺解析库时自动跳过）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, data: bytes) -> Path:
        path = self.root / name
        path.write_bytes(data)
        return path

    def test_web_loader_only_matches_http_urls(self):
        loader = WebLoader()
        self.assertTrue(loader.supports("https://example.com/page"))
        self.assertTrue(loader.supports("http://example.com"))
        self.assertFalse(loader.supports("ftp://example.com"))
        self.assertFalse(loader.supports(str(self.root / "a.txt")))

    @unittest.skipUnless(
        importlib.util.find_spec("langchain_community") is not None,
        "langchain-community 未安装，扩展格式未注册",
    )
    def test_registry_dispatches_url_to_web_loader(self):
        registry = build_default_loader_registry()
        self.assertIsInstance(
            registry.for_source("https://example.com/page"), WebLoader
        )

    @unittest.skipUnless(
        all(importlib.util.find_spec(module) is not None
            for module in ("langchain_community", "docx2txt")),
        "Word 解析依赖未安装",
    )
    def test_word_loader_docx(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "word/document.xml",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                'wordprocessingml/2006/main"><w:body><w:p><w:r>'
                '<w:t>Word 正文内容</w:t></w:r></w:p></w:body></w:document>',
            )
        path = self._write("sample.docx", buffer.getvalue())
        documents = WordLoader().load(path)
        self.assertGreaterEqual(len(documents), 1)
        self.assertIn("Word 正文", documents[0].page_content)
        self.assertEqual(documents[0].metadata["doc_type"], "word")
        self.assertEqual(documents[0].metadata["source"], str(path.resolve()))

    @unittest.skipUnless(
        importlib.util.find_spec("openpyxl") is not None, "openpyxl 未安装"
    )
    def test_excel_loader_one_document_per_row(self):
        from openpyxl import Workbook
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["name", "price"])
        sheet.append(["甲", 10])
        sheet.append(["乙", 20])
        buffer = io.BytesIO()
        workbook.save(buffer)
        path = self._write("sample.xlsx", buffer.getvalue())
        documents = ExcelLoader().load(path)
        self.assertEqual(len(documents), 2)
        self.assertEqual([d.metadata["row_index"] for d in documents], [0, 1])
        self.assertIn("name: 甲", documents[0].page_content)

    @unittest.skipUnless(
        importlib.util.find_spec("pptx") is not None, "python-pptx 未安装"
    )
    def test_ppt_loader_one_document_per_slide(self):
        from pptx import Presentation
        presentation = Presentation()
        for text in ("第一页正文", "第二页正文"):
            slide = presentation.slides.add_slide(presentation.slide_layouts[6])
            slide.shapes.add_textbox(0, 0, 100, 100).text_frame.text = text
        buffer = io.BytesIO()
        presentation.save(buffer)
        path = self._write("sample.pptx", buffer.getvalue())
        documents = PptLoader().load(path)
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].metadata["page_number"], 1)
        self.assertEqual(documents[1].metadata["total_pages"], 2)

    def test_unknown_exception_is_wrapped_as_loader_error(self):
        def broken_extractor(_path):
            raise RuntimeError("解析器内部崩溃")

        path = self._write("broken.pdf", b"%PDF-1.4 fake")
        with self.assertRaises(LoaderError) as ctx:
            PdfLoader(text_extractor=broken_extractor).load(path)
        self.assertIn(str(path), str(ctx.exception))
        self.assertIn("RuntimeError", str(ctx.exception))

    def test_dependency_hint_for_new_formats(self):
        error = DependencyMissingError("docx2txt", "Word 文档加载")
        self.assertIn("pip install docx2txt", str(error))

    def test_metadata_source_is_normalized(self):
        self._write("a.txt", "纯文本内容足够长可以进索引。".encode("utf-8"))
        self._write("b.json", '[{"x": "JSON 正文足够长"}]'.encode("utf-8"))
        self._write("c.csv", "col\n值足够长可以进索引\n".encode("utf-8"))
        self._write(
            "d.html",
            "<html><body><p>HTML 正文足够长</p></body></html>".encode("utf-8"),
        )
        registry = build_default_loader_registry()
        for path in sorted(self.root.iterdir()):
            for document in registry.load(path):
                metadata = document.metadata
                self.assertEqual(metadata["source"], str(path.resolve()))
                self.assertEqual(metadata["filename"], path.name)
                self.assertEqual(metadata["suffix"], path.suffix)
                self.assertIn("doc_type", metadata)


if __name__ == "__main__":
    unittest.main()
