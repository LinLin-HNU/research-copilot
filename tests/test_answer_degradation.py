"""Local final-record recovery: no network, rewriting, or truth certification."""
import unittest
from deep_research import _prepare_final_contract, _render_final, WorkflowStop


class AnswerDegradationTests(unittest.TestCase):
    def setUp(self):
        self.seen = {"c1": {"section": "method", "page_start": 1}}
        self.trace = {"events": []}

    def item(self, text="A source-linked statement.", cid="c1"):
        return {"kind": "paper_fact", "text": text, "evidence_ids": [cid]}

    def prepare(self, data):
        return _prepare_final_contract(data, [{"id": "q1", "text": "Explain"}],
                                       self.seen, True, self.trace)

    def test_1732_character_record_is_preserved_without_rewrite(self):
        original = "实验结果" * 433
        result = self.prepare({"items": [self.item(original)]})
        self.assertEqual("".join(x["text"] for x in result[0]["items"]), original)
        self.assertEqual(len(result[0]["items"]), 2)
        self.assertEqual(self.trace["answer_degradation"]["extra_model_calls"], 0)
        self.assertFalse(self.trace["answer_degradation"]["semantic_accuracy_proven"])

    def test_invalid_citation_does_not_discard_other_valid_records(self):
        result = self.prepare({"items": [self.item(), self.item(cid="foreign")]})
        self.assertEqual(result[0]["items"], [self.item()])
        self.assertNotIn("foreign", str(result[0]))

    def test_bad_coverage_preserves_answer_but_not_supported_status(self):
        result = self.prepare({"items": [self.item()], "coverage": [{"status": "answered"}]})
        self.assertFalse(result[3])
        self.assertEqual(result[1], [])
        self.assertTrue(self.trace["answer_degradation"]["decisions"][0]["metadata_discarded"])

    def test_bad_footer_and_heading_do_not_destroy_valid_neighbours(self):
        result = self.prepare({"items": [self.item(), dict(self.item(), heading="x" * 81)],
                               "unresolved": "not a list", "caveats": ["x" * 501]})
        answer, claims, _ = _render_final(result[0], self.seen, True)
        self.assertEqual(len(claims), 1)
        self.assertIn("source-linked", answer)

    def test_no_valid_records_does_not_invent_paper_facts(self):
        with self.assertRaises(WorkflowStop):
            self.prepare({"items": [self.item(cid="foreign")]})
        result = self.prepare({"items": [], "unresolved": ["Evidence is missing"]})
        self.assertEqual(result[0]["items"], [])

    def test_item_limit_preserves_bounded_prefix_and_records_omissions(self):
        result = self.prepare({"items": [self.item()] * 65})
        self.assertEqual(len(result[0]["items"]), 64)
        self.assertTrue(self.trace["answer_degradation"]["decisions"][-1]["discarded"])

    def test_unmodified_valid_answer_has_no_degradation(self):
        self.prepare({"items": [self.item()]})
        self.assertNotIn("answer_degradation", self.trace)

    def test_trace_disk_failure_does_not_erase_answer(self):
        from unittest.mock import patch
        from pathlib import Path
        from deep_research import _save_private_trace
        trace = {"request_id": "offline", "answer": "A retained answer"}
        with patch.object(Path, "mkdir", side_effect=PermissionError):
            _save_private_trace(trace, Path("offline-trace"))
        self.assertEqual(trace["answer"], "A retained answer")
        self.assertEqual(trace["trace_persistence"]["status"], "failed")

    def test_closed_long_record_is_retained_after_token_truncation(self):
        import json
        from partial_output import recover_final
        data, report = recover_final({"parse_error": "output_token_limit",
            "raw": '{"items":[' + json.dumps(self.item("x" * 1732)) + ',{"text":"cut'}, self.seen)
        self.assertEqual("".join(x["text"] for x in data["items"]), "x" * 1732)
        self.assertEqual(report["extra_model_calls"], 0)

    def test_full_and_rag_workflows_return_partial_without_extra_calls(self):
        from adaptive_research import run_adaptive_research
        from deep_research import run_deep_research
        from tests.test_deep_research import FakeModel, FakeRetriever
        from tests.test_paper_structure import ROWS

        class FullModel:
            def call(self, stage, payload, **limits):
                return {"data": {"items": [{"kind": "paper_fact", "text": "x" * 1732,
                                           "evidence_ids": ["c1"]}]}, "usage": {}}

        full = run_adaptive_research("Explain", "paper", ROWS, None, FullModel(),
                                    strategy="deep", answer_depth="advanced")
        self.assertEqual((full["status"], full["model_calls"]), ("partial", 1))
        self.assertEqual(len(full["claims"]), 2)
        model = FakeModel(final_data={"items": [{"kind": "paper_fact", "text": "x" * 1732,
                                                "evidence_ids": ["thread_2"]}]})
        rag = run_deep_research("Explain", "thread", FakeRetriever(), model)
        self.assertEqual((rag["status"], rag["model_calls"]), ("partial", 3))
        self.assertEqual(len(rag["claims"]), 2)

    def test_general_background_also_keeps_valid_records_without_paper_claims(self):
        from deep_research import run_deep_research
        class Model:
            def call(self, *args, **kwargs):
                return {"data": {"items": [{"kind": "background", "text": "x" * 1732,
                                            "evidence_ids": []},
                                           {"kind": "paper_fact", "text": "unverified"}]},
                        "usage": {}}
        result = run_deep_research("Explain", "", None, Model(), mode="general")
        self.assertEqual((result["status"], result["model_calls"]), ("partial", 1))
        self.assertNotIn("unverified", result["answer"])

    def test_late_full_response_is_validated_and_retained_without_review(self):
        from adaptive_research import run_adaptive_research
        from tests.test_paper_structure import ROWS
        current = [0.0]
        calls = []
        class Model:
            def call(self, stage, *args, **kwargs):
                calls.append(stage)
                current[0] = 121.0
                return {"data": {"items": [{"kind": "paper_fact", "text": "A statement",
                                            "evidence_ids": ["c1"]}]}, "usage": {}}
        result = run_adaptive_research("Explain", "paper", ROWS, None, Model(),
            strategy="deep", enable_quality_review=True, clock=lambda: current[0])
        self.assertEqual(result["stop_reason"], "deadline_answer_retained")
        self.assertIn("A statement", result["answer"])
        self.assertEqual(calls, ["full_context"])
