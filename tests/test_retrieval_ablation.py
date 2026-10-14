import unittest

from benchmark.experiments.run_retrieval_ablation import (
    _diagnose_term_stage,
    _paired_changes,
    _summarize,
)


CASE = {
    "id": "x",
    "thread_id": "paper",
    "expected_category": "A1",
    "expected_sections": ["method"],
    "expected_terms": ["normalization"],
}


def chunk(chunk_id, text, section="method"):
    return {"id": chunk_id, "content": text, "section": section, "page_start": 1, "page_end": 1}


class RetrievalAblationTests(unittest.TestCase):
    def test_failure_stage_is_attributed_in_order(self):
        good = chunk("good", "normalization appears here")
        bad = chunk("bad", "unrelated text")
        self.assertEqual(
            _diagnose_term_stage(CASE, [bad], [bad], [bad], [bad]),
            "expected_terms_not_found_in_corpus",
        )
        self.assertEqual(
            _diagnose_term_stage(CASE, [good], [bad], [bad], [bad]),
            "candidate_pool_miss",
        )
        self.assertEqual(
            _diagnose_term_stage(CASE, [good], [good, bad], [bad], [bad]),
            "selection_miss",
        )
        self.assertEqual(
            _diagnose_term_stage(CASE, [good], [good], [good], [bad]),
            "packing_or_truncation_loss",
        )
        self.assertIsNone(_diagnose_term_stage(CASE, [good], [good], [good], [good]))

    def test_diagnosis_uses_exact_emitted_text_after_truncation(self):
        full = chunk("full", "prefix normalization")
        emitted = {**full, "content_emitted": "prefix"}
        self.assertEqual(
            _diagnose_term_stage(CASE, [full], [full], [full], [emitted]),
            "packing_or_truncation_loss",
        )

    def test_paired_changes_reports_both_improvement_and_regression(self):
        rows = [
            {"id": "improved", "variant": "R0_vector", "retrieval_hit": False, "section_hit": True, "term_hit": False},
            {"id": "improved", "variant": "R1_mmr", "retrieval_hit": True, "section_hit": True, "term_hit": True},
            {"id": "regressed", "variant": "R0_vector", "retrieval_hit": True, "section_hit": True, "term_hit": True},
            {"id": "regressed", "variant": "R1_mmr", "retrieval_hit": False, "section_hit": True, "term_hit": False},
        ]
        changes = _paired_changes(rows)
        self.assertEqual(changes["term_hit"]["R0_miss_to_R1_hit"], ["improved"])
        self.assertEqual(changes["term_hit"]["R0_hit_to_R1_miss"], ["regressed"])

    def test_summary_rejects_budget_drift(self):
        base = {
            "id": "x", "thread_id": "p", "expected_category": "A1",
            "retrieval_hit": True, "section_hit": True, "term_hit": True,
            "evidence_chars": 10, "selected_chunk_ids": ["a"],
            "term_failure_stage": None, "candidate_pool": 20, "max_chunks": 8,
            "max_chunk_chars": 600, "char_budget": 4000,
        }
        rows = [
            {**base, "variant": "R0_vector"},
            {**base, "variant": "R1_mmr", "char_budget": 5000},
        ]
        with self.assertRaises(ValueError):
            _summarize(rows)


if __name__ == "__main__":
    unittest.main()
