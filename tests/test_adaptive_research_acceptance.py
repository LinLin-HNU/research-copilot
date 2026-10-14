import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_research import (
    AdaptivePolicy,
    build_full_paper_context,
    public_research_result,
    run_adaptive_research,
    select_research_mode,
)


def chunk(chunk_id: str, content: str, *, page: int, section: str = "method") -> dict:
    return {
        "id": chunk_id,
        "content": content,
        "section": section,
        "page_start": page,
        "page_end": page,
        "_order": page,
    }


class AdaptiveResearchAcceptanceTests(unittest.TestCase):
    def test_auto_routes_global_question_to_full_and_local_question_to_rag(self):
        rows = [chunk("paper_0", "A" * 20_000, page=1)]
        global_decision = select_research_mode(
            [rows], question="请总结这篇论文的整体架构和主要贡献", strategy="auto"
        )
        local_decision = select_research_mode(
            [rows], question="ProbSparse 的稀疏度量公式是什么？", strategy="auto"
        )
        self.assertEqual((global_decision.mode, global_decision.question_scope),
                         ("paper_full", "global"))
        self.assertEqual((local_decision.mode, local_decision.question_scope),
                         ("paper_deep", "local"))
        self.assertGreater(global_decision.estimated_full_input_tokens, 0)
        self.assertGreater(global_decision.estimated_cost_ratio, 0)

    def test_full_paper_explanation_is_global_but_formula_question_is_local(self):
        rows = [chunk("sundial_0", "A" * 78_206, page=1)]
        broad = select_research_mode(
            [rows], question="请全面讲解一下Sundial这篇论文，帮我和之前读过的论文连贯起来",
            strategy="auto",
        )
        focused = select_research_mode(
            [rows], question="TimeFlow Loss 的公式如何推导？", strategy="auto"
        )
        self.assertEqual((broad.mode, broad.question_scope), ("paper_full", "global"))
        self.assertEqual((focused.mode, focused.question_scope), ("paper_deep", "local"))

    def test_fast_and_deep_are_explicit_user_overrides(self):
        rows = [chunk("paper_0", "A" * 20_000, page=1)]
        fast = select_research_mode(
            [rows], question="总结全文", strategy="fast"
        )
        deep = select_research_mode(
            [rows], question="某个公式？", strategy="deep"
        )
        self.assertEqual((fast.mode, fast.reason), ("paper_deep", "user_selected_fast"))
        self.assertEqual((deep.mode, deep.reason), ("paper_full", "user_selected_deep"))

    def test_default_policy_matches_current_corpus_split_boundary(self):
        short_paper = [chunk("short_0", "A" * 88_091, page=1)]
        long_paper = [chunk("long_0", "A" * 142_019, page=1)]
        self.assertEqual(select_research_mode([short_paper]).mode, "paper_full")
        self.assertEqual(select_research_mode([long_paper]).mode, "paper_deep")

    def test_no_selected_paper_uses_general_mode(self):
        decision = select_research_mode([], policy=AdaptivePolicy(full_context_max_chars=1_000))
        self.assertEqual(decision.mode, "general")
        self.assertEqual(decision.reason, "no_paper_selected")

    def test_stopped_public_answer_is_actionable_without_admin_details(self):
        public = public_research_result({
            "request_id": "req-1", "mode": "paper_deep", "status": "stopped",
            "stop_reason": "invalid_model_json",
            "answer": "请查看停止原因", "sources": [], "unresolved": ["invalid_model_json"],
        })
        self.assertIn("分析未能完成", public["answer"])
        self.assertIn("请求编号", public["suggestion"])
        self.assertNotIn("stop_reason", public)

    def test_short_single_paper_uses_full_context_without_silent_truncation(self):
        rows = [
            chunk("paper_0", "摘要内容", page=1, section="abstract"),
            chunk("paper_1", "核心方法", page=3),
        ]
        decision = select_research_mode([rows], policy=AdaptivePolicy(full_context_max_chars=1_000))
        context, evidence = build_full_paper_context(rows, max_chars=1_000)

        self.assertEqual(decision.mode, "paper_full")
        self.assertIn("摘要内容", context)
        self.assertIn("核心方法", context)
        self.assertEqual([item["id"] for item in evidence], ["paper_0", "paper_1"])
        self.assertEqual(sum(len(row["content"]) for row in rows), decision.paper_chars)

    def test_full_context_rejects_over_budget_instead_of_silently_becoming_rag(self):
        rows = [chunk("paper_0", "A" * 1_001, page=1)]
        decision = select_research_mode([rows], policy=AdaptivePolicy(full_context_max_chars=1_000))
        self.assertEqual((decision.mode, decision.reason), ("paper_deep", "paper_exceeds_full_context_budget"))
        with self.assertRaisesRegex(ValueError, "full_context_budget_exceeded"):
            build_full_paper_context(rows, max_chars=1_000)

    def test_multiple_papers_route_to_rag_even_when_each_is_short(self):
        papers = [
            [chunk("paper_a_0", "A" * 10, page=1)],
            [chunk("paper_b_0", "B" * 10, page=1)],
        ]
        decision = select_research_mode(papers, policy=AdaptivePolicy(full_context_max_chars=1_000))
        self.assertEqual((decision.mode, decision.reason), ("paper_deep", "multiple_papers_selected"))

    def test_public_result_hides_admin_metrics_and_internal_trace(self):
        internal = {
            "request_id": "req-1",
            "mode": "paper_full",
            "status": "ok",
            "stop_reason": "full_context_answer",
            "answer": "结论",
            "claims": [{"kind": "paper_fact", "text": "结论", "evidence_ids": ["paper_0"]}],
            "unresolved": [],
            "model_calls": 1,
            "retrieval_calls": 0,
            "input_tokens": 123,
            "output_tokens": 45,
            "elapsed_ms": 678.9,
            "steps": [{"step": "mode_selection"}],
            "local_limits": {"charged_tokens_today": 168},
            "external_processing": {"payload_chars": 2_000},
        }

        public = public_research_result(internal)

        self.assertEqual(
            set(public),
            {"request_id", "mode", "status", "answer", "sources", "unresolved", "caveats", "suggestion"},
        )
        self.assertNotIn("claims", public)
        serialized = json.dumps(public, ensure_ascii=False)
        for private_name in (
            "model_calls", "retrieval_calls", "input_tokens", "output_tokens",
            "elapsed_ms", "steps", "local_limits", "external_processing", "stop_reason",
        ):
            self.assertNotIn(private_name, serialized)

    def test_full_context_preserves_page_section_and_source_ids_for_citations(self):
        rows = [
            chunk("paper_1", "第一段", page=2, section="introduction"),
            chunk("paper_2", "第二段", page=4, section="method"),
        ]
        context, evidence = build_full_paper_context(rows, max_chars=1_000)
        self.assertIn("[paper_1 | introduction | p.2-2]", context)
        self.assertIn("[paper_2 | method | p.4-4]", context)
        self.assertTrue(all(item["sha256"] for item in evidence))

    def test_short_paper_runtime_uses_one_model_call_and_zero_searches(self):
        class FullContextModel:
            def __init__(self):
                self.calls = []

            def call(self, stage, payload, *, max_output_tokens, timeout_seconds):
                self.calls.append((stage, payload))
                data = {
                    "items": [{
                        "kind": "paper_fact",
                        "text": "论文采用核心方法。",
                        "evidence_ids": ["paper_1"],
                    }],
                    "unresolved": [],
                }
                return {
                    "data": data,
                    "raw": json.dumps(data, ensure_ascii=False),
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "parse_error": None,
                }

        class SearchMustNotRun:
            def search(self, *args, **kwargs):
                raise AssertionError("short-paper full-context mode must not search")

        rows = [chunk("paper_1", "核心方法的完整说明", page=3)]
        model = FullContextModel()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_adaptive_research(
                "采用了什么方法？",
                "paper",
                rows,
                SearchMustNotRun(),
                model,
                policy=AdaptivePolicy(full_context_max_chars=1_000),
                strategy="deep",
                conversation_history=[{
                    "question": "上一轮问题",
                    "answer": "上一轮回答",
                    "mode": "paper_full",
                }],
                trace_dir=Path(tmp),
            )
            trace = json.loads(
                (Path(tmp) / f"{result['request_id']}.json").read_text(encoding="utf-8")
            )

        self.assertEqual((result["mode"], result["status"]), ("paper_full", "ok"))
        self.assertEqual((result["model_calls"], result["retrieval_calls"]), (1, 0))
        self.assertEqual([call[0] for call in model.calls], ["full_context"])
        payload = model.calls[0][1]
        self.assertEqual(
            list(payload),
            ["paper_context", "conversation", "question", "answer_depth"],
        )
        self.assertEqual(payload["answer_depth"], "beginner")
        self.assertEqual(payload["conversation"][0]["question"], "上一轮问题")
        self.assertEqual(trace["events"][0]["step"], "mode_selection")
        self.assertEqual(trace["events"][1]["step"], "full_context")
        self.assertIn("Method，第3页", result["answer"])
        self.assertNotIn("paper_1", result["answer"])

    def test_full_context_invalid_citation_fails_closed(self):
        class BadCitationModel:
            def call(self, stage, payload, *, max_output_tokens, timeout_seconds):
                data = {
                    "items": [{
                        "kind": "paper_fact",
                        "text": "无法核验的结论",
                        "evidence_ids": ["other_paper_99"],
                    }],
                    "unresolved": [],
                }
                return {"data": data, "raw": json.dumps(data),
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "parse_error": None}

        result = run_adaptive_research(
            "结论？", "paper", [chunk("paper_1", "真实证据", page=1)], None,
            BadCitationModel(), strategy="deep",
            policy=AdaptivePolicy(full_context_max_chars=1_000),
        )
        self.assertEqual((result["status"], result["stop_reason"]),
                         ("stopped", "invalid_citation"))
        self.assertNotIn("无法核验的结论", result["answer"])

    def test_full_context_structured_answer_uses_readable_sources_and_soft_caveats(self):
        class FullContextModel:
            def call(self, stage, payload, *, max_output_tokens, timeout_seconds):
                self.output_limit = max_output_tokens
                data = {
                    "items": [{
                        "heading": "核心方法", "kind": "paper_fact",
                        "text": "论文采用渐进式分解。", "evidence_ids": ["paper_1"],
                    }],
                    "coverage": [{
                        "subquestion_id": "q1", "status": "answered",
                        "reason": "已经回答核心问题", "evidence_ids": ["paper_1"],
                    }],
                    "caveats": ["论文没有逐层报告额外内存数值"],
                }
                return {"data": data, "raw": json.dumps(data, ensure_ascii=False),
                        "usage": {"input_tokens": 100, "output_tokens": 30},
                        "parse_error": None}

        model = FullContextModel()
        result = run_adaptive_research(
            "请讲解核心方法", "paper",
            [chunk("paper_1", "渐进式分解的完整说明", page=4, section="method")],
            None, model, strategy="deep",
            policy=AdaptivePolicy(full_context_max_chars=1_000),
        )

        self.assertEqual((result["status"], result["stop_reason"]),
                         ("ok_with_limits", "answer_with_caveats"))
        self.assertEqual(result["unresolved"], [])
        self.assertEqual(model.output_limit, 2600)
        self.assertIn("### 核心方法", result["answer"])
        self.assertIn("Method，第4页", result["answer"])
        self.assertNotIn("paper_1", result["answer"])
        self.assertEqual(result["sources"], [{
            "number": 1, "section": "method", "page_start": 4, "page_end": 4,
        }])

    def test_auto_rag_partial_falls_back_to_full_once_and_aggregates_usage(self):
        class FullContextModel:
            def call(self, stage, payload, *, max_output_tokens, timeout_seconds):
                self.assert_stage = stage
                data = {"items": [{"kind": "paper_fact", "text": "全文补齐结论",
                                    "evidence_ids": ["paper_1"]}], "unresolved": []}
                return {"data": data, "raw": json.dumps(data),
                        "usage": {"input_tokens": 100, "output_tokens": 20,
                                  "cached_tokens": 80}, "parse_error": None}

        rag_result = {
            "request_id": "rag-stage-1", "status": "partial",
            "stop_reason": "followup_incomplete", "answer": "证据不足",
            "claims": [], "unresolved": ["机制"], "model_calls": 3,
            "actions": 2, "retrieval_calls": 2, "evidence_chars": 4000,
            "unique_evidence_ids": 2, "evidence_ids": ["paper_1", "paper_2"],
            "input_tokens": 4000,
            "output_tokens": 800, "cached_tokens": 0, "usage_missing_calls": 0,
            "external_payload_chars": 9000, "elapsed_ms": 1000, "steps": [],
        }
        rows = [chunk("paper_1", "完整论文证据", page=1)]
        with tempfile.TemporaryDirectory() as tmp, \
                patch("deep_research.run_deep_research", return_value=rag_result):
            result = run_adaptive_research(
                "具体机制是什么？", "paper", rows, object(), FullContextModel(),
                strategy="auto", policy=AdaptivePolicy(full_context_max_chars=1_000),
                allow_full_fallback=True,
                trace_dir=Path(tmp),
            )

        self.assertEqual((result["mode"], result["status"]), ("paper_full", "ok"))
        self.assertTrue(result["fallback_used"])
        self.assertEqual((result["model_calls"], result["retrieval_calls"]), (4, 2))
        self.assertEqual((result["input_tokens"], result["cached_tokens"]), (4100, 80))
        self.assertEqual(result["strategy"], "auto")
        self.assertEqual(result["mode_reason"], "auto_rag_partial_fallback_to_full")
        self.assertEqual(result["unique_evidence_ids"], 2)
        self.assertEqual(result["evidence_ids"], ["paper_1", "paper_2"])

    def test_auto_partial_only_recommends_deep_by_default(self):
        rag_result = {
            "request_id": "rag-stage-1", "status": "partial",
            "stop_reason": "followup_incomplete", "answer": "已有较完整回答",
            "claims": [], "unresolved": ["一个可选细节"], "model_calls": 3,
            "actions": 2, "retrieval_calls": 2, "evidence_chars": 4000,
            "unique_evidence_ids": 2, "evidence_ids": ["paper_1", "paper_2"],
            "input_tokens": 4000, "output_tokens": 800, "cached_tokens": 0,
            "usage_missing_calls": 0, "external_payload_chars": 9000,
            "elapsed_ms": 1000, "steps": [],
        }
        rows = [chunk("paper_1", "完整论文证据", page=1)]
        with patch("deep_research.run_deep_research", return_value=rag_result) as rag:
            result = run_adaptive_research(
                "具体机制是什么？", "paper", rows, object(), object(),
                strategy="auto", policy=AdaptivePolicy(full_context_max_chars=1_000),
            )

        rag.assert_called_once()
        self.assertEqual(result["mode"], "paper_deep")
        self.assertFalse(result["fallback_used"])
        self.assertTrue(result["deep_retry_recommended"])
        self.assertIn("手动选择 deep", result["suggestion"])


if __name__ == "__main__":
    unittest.main()
