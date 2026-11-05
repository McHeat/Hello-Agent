"""清洗流水线：各步骤语义、去重、错误策略与报告。"""
from __future__ import annotations

import unittest

from cn.mcheat.rag.cleaning import (
    BoilerplateCleaner,
    CleaningPipeline,
    ControlCharacterCleaner,
    DeduplicateCleaner,
    DropEmptyCleaner,
    HtmlTagCleaner,
    UnicodeNormalizer,
    WhitespaceCleaner,
)
from cn.mcheat.rag.config import CleaningConfig
from cn.mcheat.rag.core.interfaces import Cleaner
from cn.mcheat.rag.core.registry import CLEANER_REGISTRY
from cn.mcheat.rag.exceptions import CleaningError, ConfigError
from cn.mcheat.rag.models import Document


def _doc(text: str, **meta) -> Document:
    return Document(page_content=text, metadata={"source": "a.txt", **meta})


class TestCleaners(unittest.TestCase):
    def test_unicode_normalizer_unifies_fullwidth(self):
        self.assertEqual(UnicodeNormalizer().clean_text("ＲＡＧ　向量"), "RAG 向量")

    def test_control_character_cleaner(self):
        dirty = "正常\u200b文本\ufeff结尾\x07"
        self.assertEqual(ControlCharacterCleaner().clean_text(dirty), "正常文本结尾")

    def test_html_tag_cleaner_keeps_plain_math(self):
        self.assertEqual(HtmlTagCleaner().clean_text("<p>甲 & 乙</p>"), " 甲 & 乙 ")
        self.assertEqual(HtmlTagCleaner().clean_text("a < b"), "a < b")

    def test_whitespace_cleaner_normalizes_newlines(self):
        self.assertEqual(
            WhitespaceCleaner().clean_text("甲  \r\n\r\n\r\n乙\t\t丙   "),
            "甲\n\n乙 丙",
        )

    def test_boilerplate_cleaner_removes_page_and_copyright(self):
        cleaner = BoilerplateCleaner([
            r"^\s*第?\s*\d+\s*页(\s*/\s*共\s*\d+\s*页)?\s*$",
            r"^\s*(copyright|版权所有).*$",
        ])
        text = "正文内容一\n第 1 页 / 共 1 页\n版权所有 2026\n正文内容二"
        self.assertEqual(cleaner.clean_text(text), "正文内容一\n正文内容二")

    def test_drop_empty_returns_none(self):
        self.assertIsNone(DropEmptyCleaner(min_length=10).clean(_doc("短")))
        self.assertIsNotNone(DropEmptyCleaner(min_length=10).clean(_doc("足够长的一段正文内容")))

    def test_deduplicate_and_reset(self):
        cleaner = DeduplicateCleaner()
        doc = _doc("重复内容 重复内容")
        self.assertIsNotNone(cleaner.clean(doc))
        self.assertIsNone(cleaner.clean(_doc("重复内容\n重复内容")))  # 归一化后同哈希
        cleaner.reset()
        self.assertIsNotNone(cleaner.clean(doc))

    def test_invalid_regex_fails_at_construction(self):
        with self.assertRaises(CleaningError):
            BoilerplateCleaner(["["])


class TestCleaningPipeline(unittest.TestCase):
    def test_from_config_builds_default_steps_in_order(self):
        cfg = CleaningConfig()
        pipeline = CleaningPipeline.from_config(cfg)
        self.assertEqual(pipeline.steps, [
            "UnicodeNormalizer", "ControlCharacterCleaner", "HtmlTagCleaner",
            "WhitespaceCleaner", "BoilerplateCleaner", "DropEmptyCleaner",
            "DeduplicateCleaner",
        ])
        self.assertEqual(len(pipeline), 7)

    def test_run_deduplicates_and_reports_step(self):
        cfg = CleaningConfig(steps=["deduplicate"], on_error="raise")
        pipeline = CleaningPipeline.from_config(cfg)
        kept, report = pipeline.run([
            _doc("相同内容相同内容"), _doc("相同内容相同内容"), _doc("不同内容不同内容"),
        ])
        self.assertEqual(len(kept), 2)
        self.assertEqual(report.total_in, 3)
        self.assertEqual(report.dropped, 1)
        self.assertEqual(report.dropped_by_step.get("DeduplicateCleaner"), 1)

    def test_state_is_reset_between_runs(self):
        pipeline = CleaningPipeline.from_config(CleaningConfig(steps=["deduplicate"]))
        docs = [_doc("内容甲内容甲")]
        self.assertEqual(len(pipeline.run(docs)[0]), 1)
        self.assertEqual(len(pipeline.run(docs)[0]), 1)  # 不应被上一轮误删

    def test_on_error_raise(self):
        class Boom(Cleaner):
            def clean(self, document):
                raise RuntimeError("boom")

        pipeline = CleaningPipeline([Boom()], on_error="raise")
        with self.assertRaises(CleaningError) as ctx:
            pipeline.run([_doc("内容")])
        self.assertIn("Boom", str(ctx.exception))

    def test_on_error_skip_records_error(self):
        class Boom(Cleaner):
            def clean(self, document):
                raise RuntimeError("boom")

        pipeline = CleaningPipeline([Boom()], on_error="skip")
        kept, report = pipeline.run([_doc("内容内容内容")])
        self.assertEqual(kept, [])
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.errors[0]["step"], "Boom")

    def test_custom_cleaner_can_be_registered(self):
        class Shout(UnicodeNormalizer):
            def clean_text(self, text):
                return super().clean_text(text).upper()

        name = "test_shout_cleaner"
        try:
            CLEANER_REGISTRY.register(name, Shout)
            pipeline = CleaningPipeline.from_config(
                CleaningConfig(steps=[name], on_error="raise")
            )
            kept, _ = pipeline.run([_doc("hello")])
            self.assertEqual(kept[0].page_content, "HELLO")
        finally:
            CLEANER_REGISTRY.unregister(name)

    def test_invalid_step_and_policy_are_rejected(self):
        with self.assertRaises(ConfigError):
            CleaningPipeline.from_config(CleaningConfig(steps=["does_not_exist"]))
        with self.assertRaises(ConfigError):
            CleaningPipeline([], on_error="whatever")


if __name__ == "__main__":
    unittest.main()
