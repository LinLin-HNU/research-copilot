import json
from pathlib import Path
import tempfile
import unittest

from benchmark.experiments.ablation_protocol import (
    EvidenceBudget,
    assert_shared_evidence_budget,
    build_manifest,
    chunk_fingerprint,
    validate_cases,
)


class AblationProtocolTests(unittest.TestCase):
    def test_budget_rejects_impossible_or_inconsistent_values(self):
        with self.assertRaises(ValueError):
            EvidenceBudget(candidate_pool=4, max_chunks=8).validate()
        with self.assertRaises(ValueError):
            EvidenceBudget(max_chunk_chars=600, char_budget=500).validate()
        self.assertEqual(EvidenceBudget().to_dict()["char_budget"], 4000)

    def test_case_validation_filters_split_and_protects_ids(self):
        cases = [
            {"id": "dev-1", "split": "dev", "question": "q", "thread_id": "t", "expected_category": "A1"},
            {"id": "test-1", "split": "test", "question": "q", "thread_id": "t", "expected_category": "C"},
        ]
        self.assertEqual([row["id"] for row in validate_cases(cases)], ["dev-1"])
        with self.assertRaises(ValueError):
            validate_cases([cases[0], cases[0]])

    def test_chunk_fingerprint_is_order_independent_but_content_sensitive(self):
        first = {"id": "a", "content": "alpha", "section": "method", "page_start": 1, "page_end": 1}
        second = {"id": "b", "content": "beta", "section": "result", "page_start": 2, "page_end": 2}
        self.assertEqual(chunk_fingerprint([first, second]), chunk_fingerprint([second, first]))
        changed = {**second, "content": "changed"}
        self.assertNotEqual(chunk_fingerprint([first, second]), chunk_fingerprint([first, changed]))

    def test_shared_budget_rejects_missing_variant_and_budget_drift(self):
        base = {"id": "x", "candidate_pool": 20, "max_chunks": 8, "max_chunk_chars": 600, "char_budget": 4000}
        with self.assertRaises(ValueError):
            assert_shared_evidence_budget([{**base, "variant": "R0_vector"}], ("R0_vector", "R1_mmr"))
        with self.assertRaises(ValueError):
            assert_shared_evidence_budget([
                {**base, "variant": "R0_vector"},
                {**base, "variant": "R1_mmr", "char_budget": 5000},
            ], ("R0_vector", "R1_mmr"))

    def test_manifest_records_hash_budget_and_no_generation_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            cases_path = Path(temp) / "cases.json"
            cases = [{"id": "x", "split": "dev", "question": "q", "thread_id": "t", "expected_category": "A1"}]
            cases_path.write_text(json.dumps(cases), encoding="utf-8")
            manifest = build_manifest(
                cases_path=cases_path,
                cases=cases,
                variants=("R0_vector", "R1_mmr"),
                budget=EvidenceBudget(),
            )
            self.assertEqual(len(manifest["cases_sha256"]), 64)
            self.assertFalse(manifest["answer_generation_called"])
            self.assertEqual(manifest["shared_budget"]["max_chunk_chars"], 600)


if __name__ == "__main__":
    unittest.main()
