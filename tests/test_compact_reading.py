import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_research import run_adaptive_research, build_full_paper_context
from deep_research_runtime import AliyunResearchModel
from paper_structure import compact_source_context, validate_structure
from tests.test_paper_structure import Model, ROWS


def compact_brief():
    return {"schema": "brief_v2", "paper_type": "method",
            "evidence": [{"key": "e1", "ref": "s1", "quote": "Encoder builds representations."},
                         {"key": "e2", "ref": "s1", "quote": "Head produces predictions."},
                         {"key": "e3", "ref": "s1", "quote": "Representation conditions the head."}],
            "components": [{"name": "Encoder", "stage": "backbone", "io": "unknown -> representation", "basis": ["e1"]},
                           {"name": "Head", "stage": "output", "io": "representation -> prediction", "basis": ["e2"]}],
            "relations": [{"from": "Encoder", "to": "Head", "basis": ["e3"]}],
            "experiments": [], "limitations": [], "unconfirmed": ["Hidden dimension is not yet verified"]}


class CompactReadingTests(unittest.TestCase):
    def test_compact_source_mapping_resolves_and_writer_still_gets_full_source(self):
        class CompactModel(Model):
            def call(self, stage, payload, **limits):
                response = super().call(stage, payload, **limits)
                if stage == "source_structure":
                    response["data"] = compact_brief()
                return response
        model = CompactModel()
        with patch("deep_research._save_private_trace") as save:
            result = run_adaptive_research("Explain", "paper", ROWS, None, model,
                        strategy="deep", answer_depth="advanced", explanation_pipeline="structured")
        self.assertEqual([c[0] for c in model.calls], ["source_structure", "structured_context"])
        self.assertTrue(result["reading_brief_used"])
        self.assertEqual([c[2]["max_output_tokens"] for c in model.calls], [2200, 6400])
        self.assertIn("[s1 |", model.calls[0][1]["paper_context"])
        self.assertIn("[c1 |", model.calls[1][1]["paper_context"])
        self.assertEqual(model.calls[1][1]["reading_brief"]["evidence"]["e1"]["id"], "c1")
        self.assertEqual(save.call_args.args[0]["structure_source_aliases"], {"s1": "c1"})

    def test_bad_source_and_omission_spliced_quotes_are_not_repaired(self):
        _, evidence = build_full_paper_context(ROWS, max_chars=10000)
        _, aliases = compact_source_context(evidence)
        data = compact_brief()
        data["evidence"][0]["quote"] = "Encoder ... representations."
        data["evidence"][1]["ref"] = "other_paper"
        parsed, report = validate_structure(data, evidence, aliases)
        self.assertIsNone(parsed)
        self.assertIn("quote_mismatch", [r["reason"] for r in report["decisions"]])
        self.assertIn("evidence_out_of_scope", [r["reason"] for r in report["decisions"]])

    def test_duplicate_keys_and_non_neural_papers(self):
        _, evidence = build_full_paper_context(ROWS, max_chars=10000)
        _, aliases = compact_source_context(evidence)
        data = compact_brief()
        data["evidence"].append(copy.deepcopy(data["evidence"][0]))
        self.assertEqual(validate_structure(data, evidence, aliases)[1]["reason"], "duplicate_evidence_key")
        data = compact_brief()
        data.update(paper_type="review", components=[], relations=[],
                    experiments=[{"text": "A review observation", "basis": ["e1"]}])
        parsed, report = validate_structure(data, evidence, aliases)
        self.assertEqual(parsed["components"], [])
        self.assertFalse(report["semantic_accuracy_proven"])

    def test_provider_length_is_distinct_from_missing_finish_reason(self):
        truncated = '{"schema":"brief_v2","evidence":['
        response = AliyunResearchModel._package_response(truncated, {"output_tokens": 2200},
                                                        finish_reason="length", max_output_tokens=2200)
        self.assertEqual(response["parse_error"], "output_token_limit")
        self.assertEqual(response["finish_reason"], "length")
        missing = AliyunResearchModel._package_response(truncated, {"output_tokens": 2200}, max_output_tokens=2200)
        self.assertIsNone(missing["finish_reason"])
        self.assertEqual(missing["parse_error"], "invalid_model_json")
        self.assertTrue(missing["output_limit_reached"])

    def test_transport_preserves_provider_finish_reason_without_network(self):
        from langchain_core.messages import AIMessage
        model = AliyunResearchModel.__new__(AliyunResearchModel)
        model._key, model._base = "fake", "https://example.invalid"
        from research_model_settings import ResearchModelSettings
        model._settings = ResearchModelSettings("test-model", model._base, model._key)
        with patch("langchain_openai.ChatOpenAI") as client:
            client.return_value.invoke.return_value = AIMessage(
                content='{"broken":', response_metadata={"finish_reason": "length"})
            result = model.call("source_structure", {"paper_context": "test"}, max_output_tokens=2200, timeout_seconds=1)
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["parse_error"], "output_token_limit")

    def test_quota_error_codes_do_not_leak_exception_message(self):
        exc = RuntimeError("secret API key must not be saved")
        exc.status_code, exc.code = 429, "insufficient_quota"
        self.assertEqual(AliyunResearchModel.failure_metadata(exc),
                         {"http_status": 429, "provider_error_code": "insufficient_quota"})

    def test_quota_rejection_stops_before_fallback_writer(self):
        class EmptyQuota(Model):
            def call(self, stage, payload, **limits):
                self.calls.append((stage, payload, limits))
                exc = RuntimeError("secret message")
                exc.status_code, exc.code = 429, "insufficient_quota"
                raise exc
        model = EmptyQuota()
        with patch("deep_research._save_private_trace") as save:
            result = run_adaptive_research("Explain", "paper", ROWS, None, model,
                        strategy="deep", answer_depth="advanced", explanation_pipeline="structured")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result["stop_reason"], "provider_quota_exhausted")
        self.assertFalse(result["reading_brief_used"])
        trace = save.call_args.args[0]
        self.assertNotIn("secret message", json.dumps(trace))
        event = next(e for e in trace["events"] if e["step"] == "model")
        self.assertEqual(event["http_status"], 429)

    def test_real_saved_truncation_replay_when_private_fixture_available(self):
        fixture = Path(__file__).resolve().parents[1] / "benchmark/experiments/private/reading_upgrade_20261009/pilot/sundial_retry_1/sundial_call_2_response.json"
        if not fixture.exists():
            self.skipTest("Private real response is intentionally not committed")
        original = json.loads(fixture.read_text(encoding="utf-8"))
        repackaged = AliyunResearchModel._package_response(original["raw"], original["usage"], max_output_tokens=2200)
        self.assertEqual(repackaged["parse_error"], "invalid_model_json")
        self.assertTrue(repackaged["output_limit_reached"])
        self.assertIsNone(repackaged["finish_reason"])
        class ReplayModel(Model):
            def call(self, stage, payload, **limits):
                response = super().call(stage, payload, **limits)
                return repackaged if stage == "source_structure" else response
        model = ReplayModel()
        with patch("deep_research._save_private_trace") as save:
            result = run_adaptive_research("Explain", "paper", ROWS, None, model,
                        strategy="deep", answer_depth="advanced", explanation_pipeline="structured")
        self.assertEqual([c[0] for c in model.calls], ["source_structure", "full_context"])
        self.assertFalse(result["reading_brief_used"])
        self.assertEqual(result["structure_fallback_reason"], "invalid_model_json")
        self.assertEqual(result["output_tokens"], 2205)
        event = next(e for e in save.call_args.args[0]["events"] if e["step"] == "model")
        self.assertTrue(event["output_limit_reached"])


if __name__ == "__main__":
    unittest.main()
