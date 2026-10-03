import unittest
from benchmark.experiments.common import windows, top_k, retrieval_metrics, proportion, agreement, pack_chunks


class ExperimentTests(unittest.TestCase):
    def test_window_order_overlap_and_metadata(self):
        chunks = [{"id": "b", "content": "EFGH", "section": "results", "page_start": 2, "page_end": 2},
                  {"id": "a", "content": "ABCD", "section": "method", "page_start": 1, "page_end": 1}]
        actual = windows(chunks, 6, 2)
        self.assertEqual(actual[0]["content"], "ABCD\nE")
        self.assertEqual(actual[1]["content"], "\nEFGH\n")
        self.assertEqual(actual[0]["source_ids"], ["a", "b"])
        self.assertEqual(actual[0]["page_end"], 2)

    def test_window_empty_and_invalid_overlap(self):
        self.assertEqual(windows([]), [])
        with self.assertRaises(ValueError):
            windows([], 10, 10)

    def test_topk_cosine_not_magnitude_or_diversity(self):
        self.assertEqual(top_k([1, 0], [[0, 99], [2, 0], [1, 0], [-1, 0]], 2), [1, 2])
        self.assertEqual(top_k([0, 0], [[1, 0]], 5), [0])
        with self.assertRaises(ValueError):
            top_k([1, 0], [[1]], 1)

    def test_wrong_b_route_does_not_disappear_from_denominator(self):
        case = {"expected_category": "A1", "expected_terms": ["Pile"]}
        self.assertFalse(retrieval_metrics(case, "", [])['retrieval_hit'])
        self.assertIsNone(retrieval_metrics({"expected_category": "B"}, "Pile", [])['retrieval_hit'])

    def test_all_terms_and_any_section(self):
        case = {"expected_category": "A1", "expected_terms": ["Pile", "SSD"], "expected_sections": ["method", "results"]}
        self.assertTrue(retrieval_metrics(case, "pile SSD", ["Results"])['retrieval_hit'])
        self.assertFalse(retrieval_metrics(case, "pile", ["Results"])['retrieval_hit'])

    def test_empty_gold_is_not_automatic_success(self):
        self.assertIsNone(retrieval_metrics({"expected_category": "A1"}, "text", [])['retrieval_hit'])

    def test_wilson_missing_and_perfect(self):
        self.assertEqual(proportion([None])["n"], 0)
        result = proportion([True]*30)
        self.assertAlmostEqual(result["ci95"][0], 0.8864866, places=6)
        self.assertEqual(result["successes"], 30)

    def test_agreement_handles_degenerate_case(self):
        self.assertEqual(agreement([(0,0), (1,1)])["kappa"], 1)
        self.assertIsNone(agreement([(1,1)])["kappa"])
        self.assertEqual(agreement([(0,1), (1,0)])["kappa"], -1)

    def test_pack_chunks_uses_original_ids_and_exact_shared_budget(self):
        chunks = [
            {"id": "c1", "content": "A" * 900, "section": "method", "page_start": 2, "page_end": 2},
            {"id": "c2", "content": "B" * 900, "section": "results", "page_start": 3, "page_end": 3},
        ]
        evidence, selected = pack_chunks(chunks, char_budget=800, max_chunk_chars=600, max_chunks=8)
        self.assertLessEqual(len(evidence), 800)
        self.assertEqual([row["id"] for row in selected], ["c1"])
        self.assertEqual(len(selected[0]["content_emitted"]), 600)
        self.assertIn("chunk_id: c1", evidence)

    def test_pack_chunks_rejects_invalid_budget(self):
        with self.assertRaises(ValueError):
            pack_chunks([], char_budget=0)


if __name__ == "__main__":
    unittest.main()
