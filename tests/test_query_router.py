import unittest
from unittest.mock import patch

import query_router


class QueryRouterGuardTests(unittest.TestCase):
    def test_limitation_question_cannot_silently_skip_retrieval(self):
        self.assertEqual(
            query_router.apply_retrieval_guard(
                "传统 self-attention 用于长序列预测时有哪些限制？", "B"
            ),
            "A2",
        )

    def test_paper_specific_identifier_is_a1_but_generic_definition_stays_b(self):
        self.assertEqual(
            query_router.apply_retrieval_guard("PatchTST 如何处理多变量时间序列？", "B"),
            "A1",
        )
        self.assertEqual(
            query_router.apply_retrieval_guard("什么是 FlashAttention？", "B"),
            "B",
        )
        self.assertEqual(
            query_router.apply_retrieval_guard("请解释 Transformer 在长序列上的限制", "B"),
            "B",
        )

    def test_experimental_design_question_is_c(self):
        self.assertEqual(
            query_router.apply_retrieval_guard(
                "如何比较不同 quantization 粒度对预测精度的影响？", "B"
            ),
            "C",
        )

    def test_non_b_prediction_is_not_overridden(self):
        self.assertEqual(query_router.apply_retrieval_guard("任意问题", "C"), "C")

    def test_route_exposes_raw_decision_and_guard(self):
        with patch.object(query_router, "classify_question", return_value="B"):
            route = query_router.build_route("该方法可能有哪些局限？")
        self.assertEqual(route["raw_category"], "B")
        self.assertEqual(route["category"], "A2")
        self.assertTrue(route["guard_applied"])
        self.assertTrue(route["needs_retrieval"])
        self.assertEqual(route["queries"][0], "该方法可能有哪些局限？")


if __name__ == "__main__":
    unittest.main()
