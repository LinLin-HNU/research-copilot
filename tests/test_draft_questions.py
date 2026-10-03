import unittest

from benchmark.draft_questions import make_cases


class DraftQuestionsTests(unittest.TestCase):
    def test_creates_balanced_review_required_dev_drafts(self):
        chunks = [
            {"id": "x_0", "section": "abstract", "page_start": 1, "page_end": 1, "content": "Chronos Forecasting System " * 20},
            {"id": "x_1", "section": "method", "page_start": 2, "page_end": 3, "content": "Chronos Tokenizer Architecture " * 20},
            {"id": "x_2", "section": "evaluation", "page_start": 4, "page_end": 5, "content": "Chronos Benchmark Dataset " * 20},
            {"id": "x_3", "section": "conclusion", "page_start": 6, "page_end": 6, "content": "Chronos Limitation FutureWork " * 20},
        ]
        cases, review = make_cases("thread", "chr", chunks)
        self.assertEqual(len(cases), 10)
        self.assertEqual([c["expected_category"] for c in cases].count("A1"), 5)
        self.assertEqual([c["expected_category"] for c in cases].count("A2"), 1)
        self.assertEqual([c["expected_category"] for c in cases].count("B"), 2)
        self.assertEqual([c["expected_category"] for c in cases].count("C"), 2)
        self.assertTrue(all(c["split"] == "dev" for c in cases))
        self.assertEqual(len(review), 10)
        self.assertTrue(all(row["review_status"] == "needs_human_verification" for row in review))
        self.assertEqual(cases[6]["expected_terms"], [])


if __name__ == "__main__":
    unittest.main()
