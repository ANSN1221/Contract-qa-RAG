"""检索与问答的单元测试：不联网。

FusionTest 直接调用生产环境的 fuse_rankings，不再自写公式——
测试要覆盖的是会被上线执行的代码。
"""

from __future__ import annotations

import unittest

from contract_qa.index import tokenize
from contract_qa.retrieval import RRF_K, fuse_rankings


class TokenizeTest(unittest.TestCase):
    def test_chinese_uses_bigrams(self):
        self.assertEqual(tokenize("付款方式"), ["付款", "款方", "方式"])

    def test_single_char_kept(self):
        self.assertEqual(tokenize("合同"), ["合同"])
        self.assertEqual(tokenize("款"), ["款"])

    def test_alnum_kept_whole(self):
        self.assertIn("doc003", tokenize("DOC003"))
        self.assertIn("2025", tokenize("2025年"))

    def test_punctuation_dropped(self):
        self.assertEqual(tokenize("。，、"), [])


class FusionTest(unittest.TestCase):
    """生产融合函数：双路都命中的条款必须排在单路命中的前面。"""

    def test_both_lists_hit_wins(self):
        result = fuse_rankings(["A", "B", "C"], ["B", "D"])
        self.assertEqual(result[0][0], "B")

    def test_rrf_score_matches_formula(self):
        result = dict(fuse_rankings(["A"], ["A"], dense_weight=1.0, bm25_weight=1.0))
        self.assertAlmostEqual(result["A"], 2.0 / (RRF_K + 1))

    def test_default_weights_match_settings(self):
        # 不传权重时用生产默认值（1:1，D5 三轮评测结论）；改默认值必须改此测试
        result = dict(fuse_rankings(["A"], ["A"]))
        self.assertAlmostEqual(result["A"], (1.0 + 1.0) / (RRF_K + 1))

    def test_weights_apply_per_channel(self):
        # dense 权重为 0 时，向量排名完全不影响结果
        result = fuse_rankings(["A", "B"], ["C"], dense_weight=0.0, bm25_weight=1.0)
        self.assertEqual([cid for cid, _ in result], ["C", "A", "B"])

    def test_dense_weight_higher_changes_order(self):
        # A 只在向量路第 1 名；C 在 BM25 路第 1 名。dense 权重大时应 A 优先
        light = fuse_rankings(["A"], ["C"], dense_weight=0.5, bm25_weight=1.0)
        heavy = fuse_rankings(["A"], ["C"], dense_weight=2.0, bm25_weight=1.0)
        self.assertEqual(light[0][0], "C")
        self.assertEqual(heavy[0][0], "A")


if __name__ == "__main__":
    unittest.main()
