import copy
import json
import tempfile
import unittest
from pathlib import Path

from adaptive_research import run_adaptive_research
from paper_structure import validate_structure


def brief():
    return {"paper_type": "method", "method_components": [
        {"name": "Encoder", "role": "core", "stage": "backbone", "input": "unknown", "output": "unknown",
         "function": "Build a representation", "evidence_ids": ["c1"], "source_quote": "Encoder builds representations."},
        {"name": "Head", "role": "core", "stage": "output", "input": "representation", "output": "prediction",
         "function": "Predict", "evidence_ids": ["c1"], "source_quote": "Head produces predictions."}],
        "method_flow": [{"from": "Encoder", "to": "Head", "relationship": "Representation conditions the head",
                         "evidence_ids": ["c1"], "source_quote": "Representation conditions the head."}],
        "experiment_evidence": [], "author_limitations": [],
        "section_coverage": [{"name": "Appendix", "status": "unconfirmed", "evidence_ids": [], "source_quote": ""}]}


TEXT = "Encoder builds representations. Head produces predictions. Representation conditions the head."
ROWS = [{"id": "c1", "content": TEXT, "section": "method", "page_start": 1}]


class Model:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def call(self, stage, payload, **limits):
        self.calls.append((stage, payload, limits))
        if self.failure == stage:
            return {"data": None, "raw": "{bad", "parse_error": "invalid_model_json",
                    "usage": {"input_tokens": 10, "output_tokens": 5, "cached_tokens": 0}}
        data = brief() if stage == "source_structure" else {
            "items": [{"kind": "paper_fact", "text": "Encoder supplies the head.", "evidence_ids": ["c1"]}],
            "coverage": [{"subquestion_id": "q1", "status": "answered", "reason": "Answered", "evidence_ids": ["c1"]}], "caveats": []}
        return {"data": data, "raw": json.dumps(data), "usage": {"input_tokens": 10, "output_tokens": 5, "cached_tokens": 0}}


class PaperStructureTests(unittest.TestCase):
    def run_workflow(self, model, **kwargs):
        return run_adaptive_research("Explain the method", "paper", ROWS, None, model,
                                    strategy="deep", answer_depth="advanced", **kwargs)

    def test_two_calls_use_full_source_without_history_in_extraction_or_reviewer(self):
        model = Model()
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_workflow(model, explanation_pipeline="structured", enable_quality_review=True,
                                       conversation_history=[{"request_id": "old", "answer": "old fact"}], trace_dir=Path(temp))
            trace = json.loads((Path(temp) / (result["request_id"] + ".json")).read_text(encoding="utf-8"))
        self.assertEqual([call[0] for call in model.calls], ["source_structure", "structured_context"])
        self.assertNotIn("conversation", model.calls[0][1])
        self.assertIn(TEXT, model.calls[1][1]["paper_context"])
        self.assertEqual(model.calls[1][1]["reading_brief"]["method_components"][0]["input"], "unknown")
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["input_tokens"], 20)
        self.assertEqual([call[2]["max_output_tokens"] for call in model.calls], [2200, 6400])
        self.assertEqual(trace["quality_review"]["status"], "disabled")
        self.assertFalse(trace["structure_checks"]["semantic_accuracy_proven"])

    def test_bad_extraction_falls_back_once_but_bad_writer_never_retries(self):
        model = Model(failure="source_structure")
        result = self.run_workflow(model, explanation_pipeline="structured")
        self.assertEqual([call[0] for call in model.calls], ["source_structure", "full_context"])
        self.assertEqual(result["status"], "ok")
        failed = Model(failure="structured_context")
        result = self.run_workflow(failed, explanation_pipeline="structured")
        self.assertEqual(len(failed.calls), 2)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["stop_reason"], "invalid_model_json")

    def test_foreign_sources_and_fabricated_quotes_are_discarded_not_verified(self):
        data = brief()
        data["method_components"][0]["evidence_ids"] = ["foreign"]
        data["method_components"][1]["source_quote"] = "A made up quote"
        parsed, checks = validate_structure(data, [{"id": "c1", "text": TEXT}])
        self.assertIsNone(parsed)
        self.assertIn("evidence_out_of_scope", [row["reason"] for row in checks["decisions"]])
        self.assertIn("quote_mismatch", [row["reason"] for row in checks["decisions"]])
        self.assertFalse(checks["semantic_accuracy_proven"])

    def test_explicit_unknown_and_theory_structure_do_not_force_neural_flow(self):
        data = brief()
        data.update(paper_type="theory", method_components=[], method_flow=[],
                    author_limitations=[{"text": "Limited scope", "evidence_ids": ["c1"], "source_quote": "Encoder builds representations."}])
        parsed, _ = validate_structure(data, [{"id": "c1", "text": TEXT}])
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["method_flow"], [])

    def test_direct_is_default_and_fresh_does_not_delete_history(self):
        history = [{"request_id": "old", "question": "previous", "answer": "old fact"}]
        model = Model()
        result = self.run_workflow(model, history_mode="fresh", conversation_history=history)
        self.assertEqual([call[0] for call in model.calls], ["full_context"])
        self.assertEqual(model.calls[0][1]["conversation"], [])
        self.assertEqual(history[0]["answer"], "old fact")
        self.assertEqual(result["explanation_pipeline"], "direct")

    def test_structured_rejects_inapplicable_mode_before_model_calls(self):
        model = Model()
        with self.assertRaisesRegex(ValueError, "structured_requires"):
            run_adaptive_research("question", "paper", ROWS, None, model, strategy="fast",
                                 explanation_pipeline="structured", answer_depth="advanced")
        self.assertEqual(model.calls, [])

    def test_serialized_input_limit_stops_before_external_call(self):
        model = Model()
        result = self.run_workflow(model, explanation_pipeline="structured",
                                   conversation_history=[{"answer": "x" * 130000}])
        self.assertEqual([call[0] for call in model.calls], ["source_structure"])
        self.assertEqual(result["stop_reason"], "serialized_input_budget")
        self.assertEqual(result["model_calls"], 1)

    def test_citation_checks_reject_writer_foreign_id_without_retry(self):
        class ForeignWriter(Model):
            def call(self, stage, payload, **limits):
                response = super().call(stage, payload, **limits)
                if stage == "structured_context":
                    response["data"]["items"][0]["evidence_ids"] = ["another_paper"]
                return response
        model = ForeignWriter()
        result = self.run_workflow(model, explanation_pipeline="structured")
        self.assertEqual(result["stop_reason"], "invalid_citation")
        self.assertEqual(len(model.calls), 2)

    def test_deadline_exhaustion_prevents_second_call(self):
        now = [0]
        class SlowModel(Model):
            def call(self, stage, payload, **limits):
                response = super().call(stage, payload, **limits)
                now[0] = 121
                return response
        model = SlowModel()
        result = self.run_workflow(model, explanation_pipeline="structured", clock=lambda: now[0])
        self.assertLessEqual(len(model.calls), 1)
        self.assertEqual(result["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
