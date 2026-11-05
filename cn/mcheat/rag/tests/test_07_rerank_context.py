"""精排、Small-to-Big 还原、Lost-in-the-Middle 编排与 prompt 组装。"""
from __future__ import annotations

import unittest

from cn.mcheat.rag.config import PromptConfig
from cn.mcheat.rag.indexing.parent_store import InMemoryParentStore
from cn.mcheat.rag.retrieval.context import (
    EdgePlacementContextBuilder,
    SmallToBigExpander,
    SequentialContextBuilder,
    _display_order,
    render_prompt,
)
from cn.mcheat.rag.retrieval.rerank import (
    CrossEncoderReranker,
    LexicalOverlapReranker,
    NoopReranker,
)

from cn.mcheat.rag.tests._helpers import make_fused, make_parent, make_scored


class TestRerankers(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            make_fused("c1", "完全不相关的天气描述", 0.030),
            make_fused("c2", "父子切分 Small-to-Big 用小片段检索返回大段落", 0.025),
            make_fused("c3", "又一段无关内容", 0.020),
        ]

    def test_lexical_reranker_puts_relevant_first_and_truncates(self):
        result = LexicalOverlapReranker().rerank(
            "父子切分 Small-to-Big", self.candidates, top_n=2
        )
        self.assertEqual(len(result), 2)  # 强制截断
        self.assertEqual(result[0].chunk.id, "c2")
        self.assertEqual([item.rank for item in result], [0, 1])
        self.assertIn("dense", result[0].recall_sources)

    def test_noop_reranker_keeps_fusion_order(self):
        result = NoopReranker().rerank("任意", self.candidates, top_n=3)
        self.assertEqual([item.chunk.id for item in result], ["c1", "c2", "c3"])

    def test_cross_encoder_with_injected_scorer_needs_no_dependency(self):
        class FakeScorer:
            def predict(self, pairs):
                return [1.0 if "Small-to-Big" in passage else 0.0
                        for _, passage in pairs]

        reranker = CrossEncoderReranker(scorer=FakeScorer(), batch_size=2)
        result = reranker.rerank("父子切分", self.candidates, top_n=2)
        self.assertEqual(result[0].chunk.id, "c2")
        self.assertEqual(len(result), 2)

    def test_empty_candidates(self):
        self.assertEqual(LexicalOverlapReranker().rerank("q", [], 5), [])


class TestSmallToBigExpander(unittest.TestCase):
    def test_dedupes_by_parent_and_attaches_full_text(self):
        store = InMemoryParentStore()
        parent = make_parent("p1", "完整的大段落上下文" * 10)
        store.put([parent])
        chunks = [
            make_scored(0, "c1", "片段甲"),
            make_scored(1, "c2", "片段乙"),
        ]
        # 手工把 parent_id 指到同一个父块
        for item in chunks:
            item.chunk.parent_id = "p1"
        expander = SmallToBigExpander(store)
        expanded = expander.expand(chunks)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0].parent.text, parent.text)
        self.assertEqual(expanded[0].chunk.id, "c1")  # 保留最相关的子块用于溯源

    def test_missing_parent_keep_child_and_drop(self):
        store = InMemoryParentStore()
        chunks = [make_scored(0, "c1", "片段甲")]
        kept = SmallToBigExpander(store, on_missing="keep_child").expand(chunks)
        self.assertEqual(len(kept), 1)
        self.assertIsNone(kept[0].parent)
        dropped = SmallToBigExpander(store, on_missing="drop").expand(chunks)
        self.assertEqual(dropped, [])

    def test_disabled_expander_returns_chunks_untouched(self):
        result = SmallToBigExpander(InMemoryParentStore(), enabled=False).expand(
            [make_scored(0, "c1", "甲")]
        )
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].parent)


class TestLostInTheMiddle(unittest.TestCase):
    def test_display_order_is_u_shaped(self):
        self.assertEqual(_display_order(5, "edges"), [0, 2, 4, 3, 1])
        self.assertEqual(_display_order(5, "sequential"), [0, 1, 2, 3, 4])
        self.assertEqual(_display_order(5, "tail"), [4, 3, 2, 1, 0])

    def test_high_relevance_chunks_sit_at_head_and_tail(self):
        chunks = [make_scored(rank, f"c{rank}", f"第{rank}段内容") for rank in range(5)]
        built = EdgePlacementContextBuilder(
            placement="edges", max_context_chars=100000
        ).build("q", chunks)
        ranks_by_position = {p.position: p.relevance_rank for p in built.placements}
        self.assertEqual(ranks_by_position[0], 0)                    # 最相关在开头
        self.assertEqual(ranks_by_position[len(chunks) - 1], 1)      # 次相关在结尾
        self.assertEqual(ranks_by_position[len(chunks) // 2], 4)     # 最不相关在中间
        self.assertEqual(len(built.chunks), 5)

    def test_sequential_baseline_differs(self):
        chunks = [make_scored(rank, f"c{rank}", f"第{rank}段内容") for rank in range(5)]
        built = SequentialContextBuilder(max_context_chars=100000).build("q", chunks)
        self.assertEqual(
            [p.relevance_rank for p in built.placements], [0, 1, 2, 3, 4]
        )

    def test_budget_drops_middle_first(self):
        chunks = [make_scored(rank, f"c{rank}", "甲" * 400) for rank in range(5)]
        built = EdgePlacementContextBuilder(
            placement="edges", max_context_chars=250
        ).build("q", chunks)
        self.assertGreater(len(built.truncated_ids), 0)
        self.assertEqual(built.placements[0].relevance_rank, 0)
        self.assertLessEqual(len(built.text), 700)  # 含单块截断后的上限

    def test_parent_text_is_used_when_available(self):
        parent = make_parent("p1", "父块完整内容父块完整内容")
        item = make_scored(0, "c1", "子片段", parent=parent)
        built = EdgePlacementContextBuilder(max_context_chars=5000).build("q", [item])
        self.assertIn("父块完整内容", built.text)
        self.assertNotIn("子片段", built.text)

    def test_empty_chunks_produce_empty_context(self):
        built = EdgePlacementContextBuilder().build("q", [])
        self.assertEqual(built.text, "")
        self.assertEqual(built.placements, [])


class TestPrompt(unittest.TestCase):
    def test_question_repeated_at_tail_for_recency(self):
        cfg = PromptConfig(repeat_question_at_tail=True)
        context = EdgePlacementContextBuilder().build("q", [])
        prompt = render_prompt("什么是 RRF？", context, cfg)
        self.assertEqual(prompt.count("问题：什么是 RRF？"), 2)
        self.assertIn("再次强调", prompt)

    def test_no_repeat_when_disabled(self):
        cfg = PromptConfig(repeat_question_at_tail=False)
        prompt = render_prompt(
            "问题甲", EdgePlacementContextBuilder().build("q", []), cfg
        )
        self.assertEqual(prompt.count("问题：问题甲"), 1)


if __name__ == "__main__":
    unittest.main()
