"""父子切分（Small-to-Big）：子串不变量、边界、重叠与参数校验。"""
from __future__ import annotations

import unittest

from cn.mcheat.rag.exceptions import ConfigError
from cn.mcheat.rag.indexing.splitters import (
    CharacterParentChildSplitter,
    FixedWindowParentChildSplitter,
)
from cn.mcheat.rag.models import Document


class TestParentChildSplitting(unittest.TestCase):
    def _long_document(self) -> Document:
        paragraphs = [
            f"第{i}段讨论主题{i}。" + ("详细论述句子。" * 12)
            for i in range(8)
        ]
        return Document(
            page_content="\n\n".join(paragraphs),
            metadata={"source": "/tmp/long.md", "filename": "long.md"},
        )

    def test_small_to_big_invariant(self):
        splitter = CharacterParentChildSplitter(
            parent_size=400, child_size=120, child_overlap=24
        )
        result = splitter.split([self._long_document()])
        self.assertGreater(len(result.parents), 1)
        self.assertGreater(len(result.children), len(result.parents))
        parents = {p.id: p for p in result.parents}
        for child in result.children:
            self.assertIn(child.parent_id, parents)
            parent = parents[child.parent_id]
            # 关键不变量：子块是父块文本的连续子串
            self.assertEqual(parent.text[child.start:child.end], child.text)
            self.assertLessEqual(len(child.text), 120)
            self.assertEqual(child.metadata["parent_id"], parent.id)
            self.assertEqual(child.metadata["doc_id"], parent.doc_id)

    def test_children_overlap_within_parent(self):
        splitter = CharacterParentChildSplitter(
            parent_size=400, child_size=100, child_overlap=30
        )
        result = splitter.split([self._long_document()])
        by_parent: dict[str, list] = {}
        for child in result.children:
            by_parent.setdefault(child.parent_id, []).append(child)
        overlapping_pairs = 0
        for children in by_parent.values():
            ordered = sorted(children, key=lambda c: c.index)
            for left, right in zip(ordered, ordered[1:]):
                if right.start < left.end:
                    overlapping_pairs += 1
        self.assertGreater(overlapping_pairs, 0)

    def test_ids_are_stable_across_runs(self):
        splitter = CharacterParentChildSplitter(parent_size=300, child_size=90)
        first = splitter.split([self._long_document()])
        second = splitter.split([self._long_document()])
        self.assertEqual([c.id for c in first.children],
                         [c.id for c in second.children])

    def test_fixed_window_variant_keeps_invariant(self):
        splitter = FixedWindowParentChildSplitter(
            parent_size=300, child_size=100, child_overlap=20
        )
        result = splitter.split([self._long_document()])
        parents = {p.id: p for p in result.parents}
        for child in result.children:
            self.assertEqual(parents[child.parent_id].text[child.start:child.end],
                             child.text)

    def test_empty_documents_are_skipped(self):
        result = CharacterParentChildSplitter(
            parent_size=100, child_size=30, child_overlap=5
        ).split([Document(page_content="   ", metadata={"source": "e.txt"})])
        self.assertEqual(result.parents, [])
        self.assertEqual(result.children, [])

    def test_invalid_parameters(self):
        for kwargs in (
            {"parent_size": 100, "child_size": 200, "child_overlap": 10},
            {"parent_size": 100, "child_size": 50, "child_overlap": 50},
            {"parent_size": 0, "child_size": 50, "child_overlap": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ConfigError):
                    CharacterParentChildSplitter(**kwargs)

    def test_very_long_unbreakable_text_does_not_lose_content(self):
        text = "甲" * 5000
        splitter = CharacterParentChildSplitter(
            parent_size=300, child_size=80, child_overlap=10
        )
        result = splitter.split([Document(page_content=text,
                                          metadata={"source": "u.txt"})])
        # 硬切兜底保证长文本不会被静默丢弃
        self.assertGreaterEqual(sum(len(p.text) for p in result.parents), 4500)
        self.assertGreater(len(result.children), 10)


if __name__ == "__main__":
    unittest.main()
