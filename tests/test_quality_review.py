import json
import tempfile
import unittest
from pathlib import Path

from adaptive_research import run_adaptive_research, public_research_result
from quality_review import add_usage, cache_total, review_once, locate_quote, validate_findings


class Reviewer:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.response = response
        self.error = error

    def call(self, stage, payload, **limits):
        self.calls.append((stage, payload, limits))
        if self.error:
            raise self.error
        return self.response


def trace():
    return dict(model_calls=1, external_payload_chars=0, events=[],
                input_tokens=100, output_tokens=20, cached_tokens=0, usage_missing_calls=0)


class QualityReviewTests(unittest.TestCase):
    def test_whitespace_matching_keeps_original_offsets_and_does_not_change_numbers(self):
        text = "Before 0.290\n MSE; after"
        match = locate_quote(text, "0.290 MSE")
        self.assertEqual(match["normalization"], "whitespace_only")
        self.assertEqual(text[match["start"]:match["end"]], "0.290\n MSE")
        self.assertIsNone(locate_quote(text, "0.296 MSE"))
        self.assertIsNone(locate_quote(text, "-0.290 MSE"))

    def test_claim_ids_and_rejection_reasons_are_auditable(self):
        issue = dict(category="number", claim_id="claim_1", reason="Check value", evidence_ids=["c1"], source_quote="0.290 MSE")
        selected = [{"id": "c1", "text": "0.290\n MSE"}]
        claims = [{"claim_id": "claim_1", "text": "A claim"}]
        valid, decisions = validate_findings([issue, {**issue, "claim_id": "missing"},
                                             {**issue, "evidence_ids": ["foreign"]},
                                             {**issue, "source_quote": "0.296 MSE"}], "A claim", selected, claims)
        self.assertEqual(len(valid), 1)
        self.assertEqual([row["reason"] for row in decisions], ["source_text_matched", "claim_not_found", "evidence_out_of_scope", "quote_mismatch"])

    def test_advisory_findings_keep_valid_quotes_and_reject_invented_sources(self):
        issue = dict(category="number", claim_id="claim_1", reason="这是训练时间点数量",
                     evidence_ids=["c1"], source_quote="94B time points")
        model = Reviewer(dict(data={"issues": [issue, {**issue, "evidence_ids": ["foreign"]}]},
                              usage={"input_tokens": 50, "output_tokens": 10, "cached_tokens": 0}))
        state = trace()
        result = review_once(model, "规模？", "规模是94B参数", [{"id": "c1", "text": "94B time points"}],
                             state, enabled=True)
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["rejected_findings"], 1)
        self.assertEqual(result["rewrite_calls"], 0)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual((state["input_tokens"], state["output_tokens"]), (150, 30))

    def test_failure_is_bounded_and_missing_usage_remains_unknown(self):
        model = Reviewer(error=TimeoutError())
        state = trace()
        result = review_once(model, "问题", "已有回答", [{"id": "c1", "text": "原文"}], state, enabled=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(model.calls), 1)
        self.assertIsNone(cache_total(state))
        self.assertEqual(state["usage_missing_calls"], 1)

    def test_no_scores_and_no_evidence_expansion(self):
        model = Reviewer(dict(data={"issues": []}, usage={"input_tokens": 20, "output_tokens": 10}))
        evidence = [{"id": f"c{i}", "text": "method words " * 1000} for i in range(30)]
        state = trace()
        report = review_once(model, "method", "method answer", evidence, state, enabled=True)
        payload = model.calls[0][1]
        self.assertLessEqual(sum(len(row["text"]) for row in payload["evidence"]), 12000)
        self.assertLessEqual(len(payload["evidence"]), 12)
        self.assertEqual(report["status"], "reviewed")
        self.assertNotIn("score", report)
        self.assertEqual(model.calls[0][2], {"max_output_tokens": 1200, "timeout_seconds": 30.0})

    def test_cache_aggregation_does_not_turn_partial_observation_into_zero(self):
        state = trace()
        add_usage(state, {"input_tokens": 5, "output_tokens": 1, "cached_tokens": 80})
        add_usage(state, {"input_tokens": 5, "output_tokens": 1})
        self.assertIsNone(cache_total(state))

    def test_full_answer_survives_failed_review_and_public_metrics_stay_private(self):
        class Model:
            def __init__(self):
                self.calls = []

            def call(self, stage, payload, **limits):
                self.calls.append(stage)
                if stage == "quality_review":
                    raise TimeoutError()
                self.depth = payload["answer_depth"]
                data = {"items": [{"kind": "paper_fact", "text": "已有可靠初稿", "evidence_ids": ["c1"]}],
                        "coverage": [{"subquestion_id": "q1", "status": "answered", "reason": "已回答", "evidence_ids": ["c1"]}],
                        "caveats": []}
                return {"data": data, "raw": json.dumps(data), "usage": {"input_tokens": 100, "output_tokens": 20, "cached_tokens": 0}}

        model = Model()
        with tempfile.TemporaryDirectory() as temp:
            result = run_adaptive_research("讲解论文", "paper", [{"id": "c1", "content": "原文", "section": "method", "page_start": 1}],
                                          None, model, strategy="deep", answer_depth="advanced", enable_quality_review=True, trace_dir=Path(temp))
            saved = json.loads((Path(temp) / (result["request_id"] + ".json")).read_text(encoding="utf-8"))
        self.assertEqual(model.depth, "advanced")
        self.assertEqual(model.calls, ["full_context", "quality_review"])
        self.assertEqual(result["status"], "ok")
        self.assertIn("已有可靠初稿", result["answer"])
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(saved["quality_review"]["status"], "failed")
        self.assertNotIn("quality_review", public_research_result(result))
        self.assertIsNone(result["cached_tokens"])

    def test_rag_review_adds_one_call_without_rerunning_retrieval_or_generation(self):
        from deep_research import run_deep_research
        from tests.test_deep_research import FakeModel, FakeRetriever

        class Model(FakeModel):
            def call(self, stage, payload, **limits):
                if stage == "quality_review":
                    self.calls.append((stage, payload, limits["max_output_tokens"], limits["timeout_seconds"]))
                    return {"data": {"issues": []}, "usage": {"input_tokens": 7, "output_tokens": 3, "cached_tokens": 0}}
                return super().call(stage, payload, **limits)

        model = Model()
        result = run_deep_research("实验？", "thread", FakeRetriever(), model, enable_quality_review=True)
        stages = [call[0] for call in model.calls]
        self.assertEqual(stages, ["plan", "assess", "final", "quality_review"])
        self.assertEqual(result["model_calls"], 4)
        self.assertEqual(result["input_tokens"], 37)
        self.assertIn("论文报告了实验结果", result["answer"])
        self.assertIsNone(result["cached_tokens"])


if __name__ == "__main__":
    unittest.main()
