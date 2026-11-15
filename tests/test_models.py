"""数据模型的判定规则测试。"""

from __future__ import annotations

import unittest

from contract_qa.models import PageStat, ParseResult


def make_result(page_chars: list[int]) -> ParseResult:
    stats = [PageStat(page_no=i + 1, chars=c, low_text=c < 20) for i, c in enumerate(page_chars)]
    return ParseResult(
        doc_id="DOC999",
        source_file="x.pdf",
        page_count=len(stats),
        page_stats=stats,
    )


class IsScannedTest(unittest.TestCase):
    def test_mostly_low_text_pages_is_scanned(self):
        # 10 页里 9 页低文本：页数再多、字符总和再多也该判扫描件
        self.assertTrue(make_result([5] * 9 + [500]).is_scanned)

    def test_normal_doc_not_scanned(self):
        self.assertFalse(make_result([500] * 10).is_scanned)

    def test_short_doc_absolute_floor(self):
        # 只有 2 页、每页 30 字：占比不触发，但总和低于下限
        self.assertTrue(make_result([30, 30]).is_scanned)

    def test_empty_document_is_scanned(self):
        self.assertTrue(make_result([]).is_scanned)

    def test_contract_with_blank_pages_not_scanned(self):
        # 常见情形：10 页里 1 页几乎空白（签字页），不应误判
        self.assertFalse(make_result([600] * 9 + [5]).is_scanned)


if __name__ == "__main__":
    unittest.main()
