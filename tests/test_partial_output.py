"""Partial final-answer recovery never calls a real model."""
import json
import unittest
from unittest.mock import patch

from adaptive_research import run_adaptive_research, public_research_result
from deep_research import run_deep_research
from deep_research_runtime import AliyunResearchModel
from partial_output import complete_items, recover_final
from tests.test_paper_structure import Model, ROWS
from tests.test_deep_research import FakeModel, FakeRetriever


def item(cid="c1", text="A complete supported statement."):
    return {"kind": "paper_fact", "text": text, "evidence_ids": [cid]}


def cut_output(cid="c1"):
    return '{"items":[' + json.dumps(item(cid)) + ',{"kind":"paper_fact","text":"unfinished'


def response(raw, limit=6400):
    return AliyunResearchModel._package_response(raw, {"input_tokens": 10, "output_tokens": limit},
                                                finish_reason="length", max_output_tokens=limit)


class PartialOutputTests(unittest.TestCase):
    def test_only_closed_items_are_recovered(self):
        seen = {"c1": {"section": "method", "page_start": 1}}
        data, report = recover_final(response(cut_output()), seen)
        self.assertEqual(data["items"], [item()])
        self.assertEqual(report["retained_items"], 1)
        self.assertFalse(report["semantic_accuracy_proven"])
        self.assertNotIn("unfinished", json.dumps(data))

    def test_escaped_strings_and_unrelated_nested_items_do_not_confuse_parser(self):
        nested = '{"metadata":{"items":[{"text":"not an answer"}]},"items":['
        raw = nested + json.dumps(item(text='Quote " and braces } ] are data.')) + ',{"text":"cut'
        self.assertEqual(complete_items(raw), [item(text='Quote " and braces } ] are data.')])
        self.assertEqual(complete_items('{"metadata":{"items":['), [])
        self.assertEqual(complete_items('{"items":[' + json.dumps(item()) + ' junk'), [])

    def test_wrong_citation_and_missing_citation_are_discarded(self):
        raw = '{"items":' + json.dumps([item(), item("foreign"), {"kind": "paper_fact", "text": "unsupported"}])
        data, report = recover_final(response(raw), {"c1": {"section": "method", "page_start": 1}})
        self.assertEqual(len(data["items"]), 1)
        self.assertEqual(len(report["rejected_items"]), 2)

    def test_no_complete_or_valid_item_means_no_fabricated_answer(self):
        self.assertIsNone(recover_final(response('{"items":[{"text":"cut'), {})[0])
        self.assertIsNone(recover_final(response('{"items":' + json.dumps([item("foreign")])), {})[0])
        self.assertIsNone(recover_final({"raw": cut_output(), "parse_error": "invalid_model_json"}, {})[0])

    def test_complete_array_with_truncated_footer_preserves_every_item(self):
        raw = '{"items":' + json.dumps([item(text=f"claim {i}") for i in range(17)]) + ',"coverage":['
        data, report = recover_final(response(raw), {"c1": {"section": "method", "page_start": 1}})
        self.assertEqual(report["retained_items"], 17)
        self.assertEqual(len(data["items"]), 17)
        self.assertEqual(complete_items('{"items":' + json.dumps([item()] * 65)), [])

    def test_full_workflow_returns_partial_without_review_or_extra_calls(self):
        class CutModel(Model):
            def call(self, stage, payload, **limits):
                self.calls.append((stage, payload, limits))
                return response(cut_output())
        model = CutModel()
        with patch("deep_research._save_private_trace") as save:
            result = run_adaptive_research("Explain", "paper", ROWS, None, model,
                         strategy="deep", answer_depth="advanced", enable_quality_review=True)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual((result["status"], result["stop_reason"]), ("partial", "output_token_limit_partial"))
        self.assertIn("complete supported", result["answer"])
        self.assertIn("本次输出未完成", result["answer"])
        self.assertTrue(public_research_result(result)["suggestion"])
        self.assertEqual(save.call_args.args[0]["output_recovery"]["extra_model_calls"], 0)

    def test_rag_partial_keeps_current_request_sources_and_three_call_budget(self):
        class CutModel(FakeModel):
            def call(self, stage, payload, **limits):
                if stage != "final":
                    return super().call(stage, payload, **limits)
                self.calls.append((stage, payload, limits["max_output_tokens"], limits["timeout_seconds"]))
                return response(cut_output("thread_2"))
        model = CutModel()
        result = run_deep_research("Explain", "thread", FakeRetriever(), model,
                         answer_depth="advanced", enable_quality_review=True)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(result["stop_reason"], "output_token_limit_partial")
        self.assertEqual(result["claims"][0]["evidence_ids"], ["thread_2"])

    def test_structure_extraction_is_not_recovered_as_a_final_answer(self):
        class CutModel(Model):
            def call(self, stage, payload, **limits):
                if stage == "source_structure":
                    self.calls.append((stage, payload, limits))
                    return response('{"schema":"brief_v2","evidence":[', limit=2200)
                return super().call(stage, payload, **limits)
        model = CutModel()
        result = run_adaptive_research("Explain", "paper", ROWS, None, model,
                  strategy="deep", answer_depth="advanced", explanation_pipeline="structured")
        self.assertFalse(result["reading_brief_used"])
        self.assertEqual([row[0] for row in model.calls], ["source_structure", "full_context"])
