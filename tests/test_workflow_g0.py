import unittest

from benchmark.experiments.validate_workflow_g0 import validate


def fixture():
    source = [{"id": f"p-a1-{i:02d}", "split": "dev"} for i in range(10)]
    cases = []
    for index in range(10):
        cases.append({
            "id": f"wf-{index:02d}",
            "split": "dev" if index < 8 else "reserve",
            "mode": "paper_deep",
            "paper_alias": "p",
            "question": "A genuine research question?",
            "subquestions": ["What is the evidence?"],
            "source_case_ids": [f"p-a1-{index:02d}"],
        })
    for mode in ("paper_deep_no_paper", "general"):
        cases.append({
            "id": f"wf-{mode}", "split": "dev", "mode": mode,
            "paper_alias": None, "question": "A control question?",
            "subquestions": ["Is the mode safe?"], "source_case_ids": [],
        })
    return cases, source


class WorkflowG0Tests(unittest.TestCase):
    def test_valid_structure_is_offline(self):
        cases, source = fixture()
        report = validate(cases, source)
        self.assertEqual((report["cases"], report["paper_dev"], report["paper_reserve"]), (12, 8, 2))
        self.assertEqual(report["network_calls"], 0)

    def test_rejects_gold_and_thread_leakage(self):
        for field in ("thread_id", "gold", "expected_terms"):
            with self.subTest(field=field):
                cases, source = fixture()
                cases[0][field] = "private"
                with self.assertRaises(ValueError):
                    validate(cases, source)

    def test_rejects_cross_paper_and_oversized_plan(self):
        cases, source = fixture()
        cases[0]["source_case_ids"] = ["other-a1-01"]
        with self.assertRaises(ValueError):
            validate(cases, source)
        cases, source = fixture()
        cases[0]["subquestions"] = ["a", "b", "c"]
        with self.assertRaises(ValueError):
            validate(cases, source)

    def test_survey_requires_separate_pdf_provenance(self):
        cases, source = fixture()
        cases[9]["paper_alias"] = "survey"
        cases[9]["source_case_ids"] = []
        self.assertEqual(validate(cases, source)["paper_reserve"], 2)
        cases[9]["source_case_ids"] = ["p-a1-09"]
        with self.assertRaises(ValueError):
            validate(cases, source)


if __name__ == "__main__":
    unittest.main()
