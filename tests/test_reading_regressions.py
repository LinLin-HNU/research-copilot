"""Budget, rendering and partial-brief regression tests; no network calls."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_research import build_full_paper_context, run_adaptive_research
from deep_research import _render_final, WorkflowStop
from deep_research_runtime import AliyunResearchModel
from paper_structure import compact_source_context, validate_structure
from research_model_settings import build_chat_model, resolve_research_model
from tests.test_compact_reading import compact_brief
from tests.test_paper_structure import Model, ROWS


class ReadingRegressionTests(unittest.TestCase):
    def test_uninitialized_client_cannot_fall_back_to_real_environment(self):
        model = AliyunResearchModel.__new__(AliyunResearchModel)
        with patch("deep_research_runtime.build_chat_model") as factory:
            with self.assertRaisesRegex(RuntimeError, "configuration missing"):
                model.call("general", {"question": "test"}, max_output_tokens=50, timeout_seconds=1)
        factory.assert_not_called()

    def test_renderer_preserves_late_claims_and_checks_their_citations(self):
        items = [{"kind": "paper_fact", "text": f"statement {i}", "evidence_ids": ["c1"]}
                 for i in range(17)]
        seen = {"c1": {"section": "method", "page_start": 1, "page_end": 1}}
        answer, claims, _ = _render_final({"items": items}, seen, True)
        self.assertEqual(len(claims), 17)
        self.assertIn("statement 16", answer)
        items[-1]["evidence_ids"] = ["foreign"]
        with self.assertRaisesRegex(WorkflowStop, "invalid_citation"):
            _render_final({"items": items}, seen, True)

    def test_oversized_answer_is_explicitly_rejected_not_trimmed(self):
        with self.assertRaisesRegex(WorkflowStop, "final_item_budget"):
            _render_final({"items": [{}] * 65}, {}, False)

    def test_deepseek_serialized_limit_and_router_override(self):
        from langchain_core.messages import HumanMessage
        settings = resolve_research_model({"OPENAI_BASE_URL": "https://api.deepseek.com",
                            "MODEL_NAME": "test-model", "DEEPSEEK_API_KEY": "offline-test-key"})
        model = build_chat_model(settings=settings, max_output_tokens=3600, timeout_seconds=60)
        for limit in (3600, 50):
            payload = model._get_request_payload([HumanMessage(content="offline")], max_tokens=limit)
            self.assertEqual(payload["max_tokens"], limit)
            self.assertNotIn("max_completion_tokens", payload)
            self.assertEqual(payload["extra_body"], {"thinking": {"type": "disabled"}})

    def test_other_provider_keeps_sdk_default_serialization(self):
        from langchain_core.messages import HumanMessage
        settings = resolve_research_model({"OPENAI_BASE_URL": "https://example.invalid/v1",
                            "MODEL_NAME": "test-model", "OPENAI_API_KEY": "offline-test-key"})
        model = build_chat_model(settings=settings, max_output_tokens=3600, timeout_seconds=60)
        payload = model._get_request_payload([HumanMessage(content="offline")])
        self.assertEqual(payload["max_completion_tokens"], 3600)
        self.assertNotIn("max_tokens", payload)

    def test_provider_overrun_keeps_usage_and_raw_but_stops(self):
        raw = '{"items":[],"unresolved":[]}'
        response = AliyunResearchModel._package_response(raw, {"output_tokens": 3601},
                                        finish_reason="stop", max_output_tokens=3600)
        self.assertEqual(response["parse_error"], "provider_output_budget_exceeded")
        self.assertEqual(response["raw"], raw)
        self.assertEqual(response["usage"]["output_tokens"], 3601)
        at_cap = AliyunResearchModel._package_response(raw, {"output_tokens": 3600},
                                        finish_reason="stop", max_output_tokens=3600)
        self.assertIsNone(at_cap["parse_error"])

    def test_engineering_only_brief_is_partial_and_not_used_as_core_chain(self):
        data = compact_brief()
        for row in data["components"]:
            row["stage"] = "engineering"
        _, evidence = build_full_paper_context(ROWS, max_chars=10000)
        _, aliases = compact_source_context(evidence)
        candidate, report = validate_structure(data, evidence, aliases)
        self.assertEqual(report["completeness"], "partial")
        self.assertFalse(report["usable_for_structured_writer"])
        self.assertFalse(candidate["validation_status"]["semantic_accuracy_proven"])
        class PartialModel(Model):
            def call(self, stage, payload, **limits):
                response = super().call(stage, payload, **limits)
                if stage == "source_structure":
                    response["data"] = copy.deepcopy(data)
                return response
        model = PartialModel()
        with patch("deep_research._save_private_trace") as save:
            result = run_adaptive_research("Explain", "paper", ROWS, None, model,
                           strategy="deep", answer_depth="advanced", explanation_pipeline="structured")
        self.assertEqual([c[0] for c in model.calls], ["source_structure", "full_context"])
        self.assertFalse(result["reading_brief_used"])
        self.assertEqual(result["structure_fallback_reason"], "incomplete_core_structure")
        self.assertIsNotNone(save.call_args.args[0]["reading_brief_candidate"])

    def test_complete_structure_is_not_semantic_proof(self):
        _, evidence = build_full_paper_context(ROWS, max_chars=10000)
        _, aliases = compact_source_context(evidence)
        _, report = validate_structure(compact_brief(), evidence, aliases)
        self.assertEqual(report["completeness"], "complete")
        self.assertTrue(report["usable_for_structured_writer"])
        self.assertFalse(report["semantic_accuracy_proven"])

    def test_extraction_budget_breach_does_not_spend_on_fallback_writer(self):
        class OverrunModel(Model):
            def call(self, stage, payload, **limits):
                self.calls.append((stage, payload, limits))
                return AliyunResearchModel._package_response('{"schema":"brief_v2"}',
                    {"input_tokens": 10, "output_tokens": 2201}, finish_reason="stop", max_output_tokens=2200)
        model = OverrunModel()
        result = run_adaptive_research("Explain", "paper", ROWS, None, model,
                         strategy="deep", answer_depth="advanced", explanation_pipeline="structured")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result["stop_reason"], "provider_output_budget_exceeded")
        self.assertEqual(result["output_tokens"], 2201)


    def test_saved_autoformer_responses_replay_without_model_calls(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "benchmark/experiments/private/reading_upgrade_20261012/pilot"
        if not (directory / "manifest.json").exists():
            self.skipTest("Private Autoformer fixture is not distributed")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        case = next(row for row in manifest["cases"] if row["id"] == "autoformer")
        _, evidence = build_full_paper_context(case["rows"], max_chars=100000)
        seen = {row["id"]: row for row in evidence}
        for number, count in ((1, 17), (3, 13)):
            response = json.loads((directory / f"autoformer_call_{number}_response.json").read_text(encoding="utf-8"))
            _, claims, _ = _render_final(response["data"], seen, True)
            self.assertEqual(len(claims), count)
        trace = json.loads((directory / "traces/e2d67775-3380-4062-bc31-57bf70c93c47.json").read_text(encoding="utf-8"))
        response = json.loads((directory / "autoformer_call_2_response.json").read_text(encoding="utf-8"))
        _, report = validate_structure(response["data"], evidence, trace["structure_source_aliases"])
        self.assertEqual(report["counts"]["components"]["accepted"], 2)
        self.assertFalse(report["usable_for_structured_writer"])
