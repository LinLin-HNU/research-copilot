import json
import os
import tempfile
import unittest
from pathlib import Path

from deep_research import (
    ChromaPaperRetriever,
    ResearchBudget,
    WorkflowStop,
    _fuse_subquery_results,
    _pack,
    run_deep_research,
)


TEST_TMP = os.environ.get("RESEARCH_COPILOT_TEST_TMP")


def chunk(chunk_id: str, content: str = "论文证据" * 40, page: int = 3) -> dict:
    return {"id": chunk_id, "content": content, "section": "method",
            "page_start": page, "page_end": page}


def assessment(action: dict, state: str = "missing_specific_fact", ids=None) -> dict:
    return {
        "subquestions": [{"subquestion_id": "q1", "assessment": state,
                          "evidence_ids": ["thread_1"] if ids is None else ids,
                          "reason": "首轮证据的范围需要复核",
                          "missing_information": "" if state == "supported" else "实验结果"}],
        "action": action,
    }


class FakeModel:
    def __init__(self, assessment_data=None, final_data=None):
        self.assessment_data = assessment_data or assessment({"type": "search", "query": "实验结果"})
        self.final_data = final_data or {
            "items": [{"kind": "paper_fact", "text": "论文报告了实验结果。",
                       "evidence_ids": ["thread_2"]}], "unresolved": []}
        self.calls = []

    def call(self, stage, payload, *, max_output_tokens, timeout_seconds):
        self.calls.append((stage, payload, max_output_tokens, timeout_seconds))
        data = {
            "plan": {"subquestions": [{"id": "q1", "text": "论文实验结果是什么？",
                                           "query": "方法与实验"}]},
            "assess": self.assessment_data,
            "final": self.final_data,
            "general": {"items": [{"kind": "background", "text": "这属于通用背景。", "evidence_ids": []}],
                        "unresolved": []},
        }[stage]
        return {"data": data, "raw": json.dumps(data, ensure_ascii=False),
                "usage": {"input_tokens": 10, "output_tokens": 5}}


class FakeRetriever:
    def __init__(self, *, has_paper=True):
        self.available = has_paper
        self.calls = []
        self.rows = [chunk("thread_1", "首轮方法证据" * 50),
                     chunk("thread_2", "后续实验结果证据" * 50, 7)]

    def has_paper(self, thread_id):
        self.calls.append(("has_paper", thread_id))
        return self.available

    def search(self, thread_id, query, *, k, exclude_ids, timeout_seconds):
        self.calls.append(("search", thread_id, query, k, exclude_ids))
        return [row for row in self.rows if row["id"] not in exclude_ids][:1]

    def expand(self, thread_id, target_id, direction, *, seen_segments, timeout_seconds):
        self.calls.append(("expand", thread_id, target_id, direction))
        return [self.rows[0]] if target_id == "thread_1" else []


class WorkflowTests(unittest.TestCase):
    def test_rag_planner_and_final_receive_recent_conversation(self):
        history = [{"question": "它是什么？", "answer": "它是 Informer。",
                    "mode": "paper_deep"}]
        model = FakeModel()
        run_deep_research(
            "那它的复杂度呢？", "thread", FakeRetriever(), model,
            conversation_history=history,
        )
        plan_payload = next(call[1] for call in model.calls if call[0] == "plan")
        final_payload = next(call[1] for call in model.calls if call[0] == "final")
        self.assertEqual(plan_payload["conversation"], history)
        self.assertEqual(final_payload["conversation"], history)

    def test_invalid_final_json_preserves_raw_and_usage_in_trace(self):
        class BrokenFinalModel(FakeModel):
            def call(self, stage, payload, *, max_output_tokens, timeout_seconds):
                if stage != "final":
                    return super().call(stage, payload, max_output_tokens=max_output_tokens,
                                        timeout_seconds=timeout_seconds)
                self.calls.append((stage, payload, max_output_tokens, timeout_seconds))
                return {
                    "data": None,
                    "raw": "{broken-final-json",
                    "usage": {"input_tokens": 40, "output_tokens": 12},
                    "parse_error": "invalid_model_json",
                }

        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            result = run_deep_research(
                "方法？", "thread", FakeRetriever(), BrokenFinalModel(), trace_dir=Path(tmp)
            )
            saved = json.loads((Path(tmp) / f"{result['request_id']}.json").read_text(encoding="utf-8"))
        final_event = [event for event in saved["events"] if event.get("stage") == "final"][0]
        self.assertEqual((result["status"], result["stop_reason"]), ("stopped", "invalid_model_json"))
        self.assertEqual((result["input_tokens"], result["output_tokens"]), (60, 22))
        self.assertEqual(final_event["raw"], "{broken-final-json")
        self.assertEqual(final_event["parse_error"], "invalid_model_json")

    def test_fusion_reserves_multiple_candidates_per_subquestion_before_shared_rows(self):
        common_a = chunk("thread_common_a")
        common_b = chunk("thread_common_b")
        q1_rows = [common_a, chunk("thread_q1_definition"), chunk("thread_q1_detail"), common_b]
        q2_rows = [common_a, chunk("thread_q2_definition"), chunk("thread_q2_detail"), common_b]
        fused = _fuse_subquery_results([("q1", q1_rows), ("q2", q2_rows)], k=8)
        first_six = [row["id"] for row in fused[:6]]
        self.assertIn("thread_q1_definition", first_six)
        self.assertIn("thread_q1_detail", first_six)
        self.assertIn("thread_q2_definition", first_six)
        self.assertIn("thread_q2_detail", first_six)

    def test_valid_followup_has_three_model_calls_two_actions_and_full_trace(self):
        retriever, model = FakeRetriever(), FakeModel()
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            result = run_deep_research("方法和实验是什么？", "thread", retriever, model,
                                       trace_dir=Path(tmp))
            saved = json.loads((Path(tmp) / f"{result['request_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual((result["status"], result["actions"], result["model_calls"]), ("ok", 2, 3))
        self.assertEqual(result["retrieval_calls"], 2)
        self.assertEqual([call[0] for call in model.calls], ["plan", "assess", "final"])
        self.assertEqual(next(call[2] for call in model.calls if call[0] == "assess"), 800)
        self.assertIn("Method，第7页", result["answer"])
        self.assertNotIn("thread_2", result["answer"])
        self.assertLessEqual(result["evidence_chars"], 6000)
        self.assertLessEqual(result["unique_evidence_ids"], 8)
        self.assertEqual(saved["events"][2]["step"], "retrieval")
        self.assertIn("首轮方法证据", saved["events"][2]["provided_segments"][0]["text"])
        self.assertTrue(any(event["step"] == "gate" and event["allowed"] for event in saved["events"]))

    def test_two_subquestions_are_searched_separately_and_fused_with_coverage(self):
        class TwoQuestionModel(FakeModel):
            def call(self, stage, payload, *, max_output_tokens, timeout_seconds):
                if stage == "plan":
                    data = {"subquestions": [
                        {"id": "q1", "text": "分类框架是什么？", "query": "forecasting taxonomy classification"},
                        {"id": "q2", "text": "如何评估？", "query": "probabilistic forecast evaluation score"},
                    ]}
                    self.calls.append((stage, payload, max_output_tokens, timeout_seconds))
                    return {"data": data, "raw": json.dumps(data),
                            "usage": {"input_tokens": 10, "output_tokens": 5}}
                if stage == "assess":
                    data = {"subquestions": [
                        {"subquestion_id": "q1", "assessment": "supported",
                         "evidence_ids": ["thread_taxonomy"], "reason": "已找到分类", "missing_information": ""},
                        {"subquestion_id": "q2", "assessment": "supported",
                         "evidence_ids": ["thread_metric"], "reason": "已找到评分", "missing_information": ""},
                    ], "action": {"type": "stop"}}
                    self.calls.append((stage, payload, max_output_tokens, timeout_seconds))
                    return {"data": data, "raw": json.dumps(data),
                            "usage": {"input_tokens": 10, "output_tokens": 5}}
                if stage == "final":
                    data = {"items": [
                        {"kind": "paper_fact", "text": "论文给出分类框架。", "evidence_ids": ["thread_taxonomy"]},
                        {"kind": "paper_fact", "text": "论文给出概率评分。", "evidence_ids": ["thread_metric"]},
                    ], "unresolved": []}
                    self.calls.append((stage, payload, max_output_tokens, timeout_seconds))
                    return {"data": data, "raw": json.dumps(data),
                            "usage": {"input_tokens": 10, "output_tokens": 5}}
                return super().call(stage, payload, max_output_tokens=max_output_tokens,
                                    timeout_seconds=timeout_seconds)

        class TwoQueryRetriever(FakeRetriever):
            def search(self, thread_id, query, *, k, exclude_ids, timeout_seconds):
                self.calls.append(("search", thread_id, query, k, exclude_ids))
                common = chunk("thread_common", "通用背景" * 50)
                if "taxonomy" in query:
                    return [common, chunk("thread_taxonomy", "分类框架" * 50)]
                return [common, chunk("thread_metric", "概率评分" * 50)]

        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            retriever = TwoQueryRetriever()
            result = run_deep_research("综述如何分类并评估预测方法？", "thread", retriever,
                                       TwoQuestionModel(), trace_dir=Path(tmp))
            saved = json.loads((Path(tmp) / f"{result['request_id']}.json").read_text(encoding="utf-8"))
        initial_searches = [call for call in retriever.calls if call[0] == "search"]
        retrieval = next(event for event in saved["events"] if event["step"] == "retrieval")
        self.assertEqual((result["status"], result["actions"], len(initial_searches)), ("ok", 1, 2))
        self.assertEqual(result["retrieval_calls"], 2)
        self.assertEqual(retrieval["retrieval_calls"], 2)
        self.assertEqual({query["subquestion_id"] for query in retrieval["queries"]}, {"q1", "q2"})
        self.assertIn("thread_taxonomy", retrieval["candidate_ids"])
        self.assertIn("thread_metric", retrieval["candidate_ids"])
        self.assertEqual(retrieval["candidate_query_ids"]["thread_common"], ["q1", "q2"])

    def test_semantically_unnecessary_search_is_not_marked_correct_by_gate(self):
        retriever = FakeRetriever()
        model = FakeModel(assessment_data=assessment({"type": "search", "query": "再查"},
                                                     ids=["thread_1"]))
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            result = run_deep_research("首轮已足够的问题", "thread", retriever, model, trace_dir=Path(tmp))
            saved = json.loads((Path(tmp) / f"{result['request_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(result["actions"], 2)
        gate = next(event for event in saved["events"] if event["step"] == "gate")
        self.assertEqual((gate["allowed"], gate["reason"]), (True, "within_budget"))
        self.assertNotIn("semantic_correct", gate)

    def test_final_revealed_gap_after_supported_stop_is_reported_partial(self):
        retriever = FakeRetriever()
        model = FakeModel(assessment_data=assessment({"type": "stop"}, state="supported"),
                          final_data={"items": [], "unresolved": ["仍需人工核对实验结果"]})
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            result = run_deep_research("论文实验结果？", "thread", retriever, model, trace_dir=Path(tmp))
            saved = json.loads((Path(tmp) / f"{result['request_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual((result["status"], result["actions"], result["stop_reason"]),
                         ("partial", 1, "final_revealed_gap_after_stop"))
        self.assertEqual(len([call for call in retriever.calls if call[0] == "search"]), 1)
        self.assertEqual(next(e for e in saved["events"] if e["step"] == "assessment")
                         ["subquestions"][0]["assessment"], "supported")
        postcheck = next(e for e in saved["events"] if e["step"] == "postcheck")
        self.assertEqual((postcheck["allowed"], postcheck["reason"]),
                         (False, "final_unresolved_after_supported_stop"))

    def test_supported_with_explicit_gap_cannot_stop_and_uses_guard_fallback(self):
        contradictory = assessment({"type": "stop"}, state="supported")
        contradictory["subquestions"][0]["missing_information"] = "移动平均分解公式"
        retriever = FakeRetriever()
        model = FakeModel(assessment_data=contradictory)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            result = run_deep_research("分解操作？", "thread", retriever, model, trace_dir=Path(tmp))
            saved = json.loads((Path(tmp) / f"{result['request_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual((result["actions"], result["stop_reason"]), (2, "followup_executed"))
        assessment_event = next(e for e in saved["events"] if e["step"] == "assessment")
        self.assertEqual(assessment_event["subquestions"][0]["normalized_from"], "supported")
        reasons = [e.get("reason") for e in saved["events"] if e["step"] == "gate"]
        self.assertEqual(reasons, ["stop_with_explicit_gap", "guard_fallback_search"])

    def test_low_yield_continuation_is_replaced_by_gap_search(self):
        retriever = FakeRetriever()
        retriever.rows[0] = chunk("thread_1", "A" * 650)
        model = FakeModel(assessment_data=assessment(
            {"type": "expand", "target_id": "thread_1", "direction": "continue"},
            state="missing_specific_fact",
        ))
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            result = run_deep_research("具体机制？", "thread", retriever, model, trace_dir=Path(tmp))
            saved = json.loads((Path(tmp) / f"{result['request_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(retriever.calls[-1][0], "search")
        reasons = [event.get("reason") for event in saved["events"] if event["step"] == "gate"]
        self.assertEqual(reasons, ["low_yield_expand", "guard_fallback_search"])

    def test_unresolved_after_followup_is_partial_not_ok(self):
        model = FakeModel(
            assessment_data=assessment({"type": "search", "query": "补查机制"}),
            final_data={"items": [], "unresolved": ["机制证据仍不足"]},
        )
        result = run_deep_research("机制？", "thread", FakeRetriever(), model)
        self.assertEqual((result["status"], result["stop_reason"]),
                         ("partial", "followup_incomplete"))

    def test_structured_coverage_keeps_optional_depth_as_ok_with_limits(self):
        model = FakeModel(final_data={
            "items": [{
                "heading": "方法与实现", "kind": "paper_fact",
                "text": "论文给出核心机制。", "evidence_ids": ["thread_2"],
            }],
            "coverage": [{
                "subquestion_id": "q1", "status": "answered",
                "reason": "核心问题已由实验结果支持", "evidence_ids": ["thread_2"],
            }],
            "caveats": ["论文没有报告逐层内存数值"],
        })
        result = run_deep_research("论文实验结果是什么？", "thread", FakeRetriever(), model)
        self.assertEqual((result["status"], result["stop_reason"]),
                         ("ok_with_limits", "answer_with_caveats"))
        self.assertEqual(result["unresolved"], [])
        self.assertIn("### 方法与实现", result["answer"])
        self.assertIn("### 论文未进一步说明", result["answer"])
        self.assertNotIn("thread_2", result["answer"])
        self.assertEqual(result["sources"][0]["page_start"], 7)

    def test_structured_coverage_only_marks_planned_core_gap_insufficient(self):
        model = FakeModel(final_data={
            "items": [],
            "coverage": [{
                "subquestion_id": "q1", "status": "insufficient",
                "reason": "实验结果没有出现在当前证据中", "evidence_ids": [],
            }],
            "caveats": [],
        })
        result = run_deep_research("论文实验结果是什么？", "thread", FakeRetriever(), model)
        self.assertEqual(result["status"], "insufficient")
        self.assertIn("论文实验结果是什么", result["unresolved"][0])

    def test_structured_coverage_cannot_invent_new_subquestion(self):
        model = FakeModel(final_data={
            "items": [],
            "coverage": [{
                "subquestion_id": "q99", "status": "insufficient",
                "reason": "额外问题", "evidence_ids": [],
            }],
            "caveats": [],
        })
        result = run_deep_research("论文实验结果是什么？", "thread", FakeRetriever(), model)
        self.assertEqual((result["status"], result["stop_reason"]),
                         ("stopped", "invalid_final_coverage"))

    def test_cross_thread_or_unseen_evidence_blocks_followup(self):
        model = FakeModel(assessment_data=assessment({"type": "expand", "target_id": "other_1",
                                                     "direction": "continue"}, ids=["other_1"]))
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            result = run_deep_research("论文证据？", "thread", FakeRetriever(), model,
                                       trace_dir=Path(tmp))
            saved = json.loads((Path(tmp) / f"{result['request_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual((result["status"], result["stop_reason"]), ("stopped", "foreign_or_unseen_evidence"))
        self.assertEqual([call[0] for call in model.calls], ["plan", "assess"])
        self.assertTrue(any(e["step"] == "gate" and not e["allowed"] for e in saved["events"]))

    def test_expand_is_targeted_and_respects_character_budget(self):
        retriever = FakeRetriever()
        retriever.rows[0] = chunk("thread_1", "A" * 1500)
        model = FakeModel(assessment_data=assessment({"type": "expand", "target_id": "thread_1",
                                                     "direction": "continue"}),
                          final_data={"items": [{"kind": "paper_fact", "text": "方法细节",
                                                 "evidence_ids": ["thread_1"]}], "unresolved": []})
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            result = run_deep_research("详细方法？", "thread", retriever, model,
                                       trace_dir=Path(tmp))
            saved = json.loads((Path(tmp) / f"{result['request_id']}.json").read_text(encoding="utf-8"))
        followup = next(event for event in saved["events"] if event["step"] == "followup")
        self.assertEqual((result["actions"], result["unique_evidence_ids"]), (2, 1))
        self.assertEqual(followup["provided_segments"][0]["start"], 600)
        self.assertTrue(saved["events"][2]["provided_segments"][0]["has_more_after"])
        self.assertTrue(followup["provided_segments"][0]["has_more_before"])
        self.assertLessEqual(result["evidence_chars"], 6000)

    def test_unknown_source_and_general_mode_never_search(self):
        no_source, model = FakeRetriever(has_paper=False), FakeModel()
        result = run_deep_research("论文内容？", "thread", no_source, model)
        self.assertEqual((result["stop_reason"], result["model_calls"]), ("no_paper", 0))
        general = run_deep_research("什么是时间序列？", "", None, FakeModel(), mode="general")
        self.assertEqual((general["status"], general["actions"], general["model_calls"]), ("ok", 0, 1))

    def test_unknown_citation_and_action_budget_fail_closed(self):
        bad_final = FakeModel(final_data={"items": [{"kind": "paper_fact", "text": "虚构论断",
                                                    "evidence_ids": ["thread_999"]}], "unresolved": []})
        result = run_deep_research("方法？", "thread", FakeRetriever(), bad_final)
        self.assertEqual((result["status"], result["stop_reason"]), ("stopped", "invalid_citation"))
        self.assertNotIn("虚构论断", result["answer"])
        stop = run_deep_research("方法？", "thread", FakeRetriever(),
                                 FakeModel(final_data={"items": [{"kind": "paper_fact", "text": "首轮方法细节",
                                         "evidence_ids": ["thread_1"]}], "unresolved": []}),
                                 budget=ResearchBudget(max_actions=1))
        self.assertEqual((stop["actions"], stop["stop_reason"]), (1, "action_budget_exhausted"))

    def test_chroma_adapter_expands_only_a_seen_chunk_in_bound_thread(self):
        class Rag:
            def list_chunks(self, thread_id):
                self.requested = thread_id
                return [chunk("thread_1"), chunk("thread_2")]

            def search(self, thread_id, query, *, k, exclude_ids, pool, timeout_seconds=None):
                self.search_args = (thread_id, query, k, exclude_ids, pool)
                return [chunk("thread_1")]

        rag = Rag()
        adapter = ChromaPaperRetriever(rag)
        self.assertTrue(adapter.has_paper("thread"))
        self.assertEqual(adapter.search("thread", "q", k=8, exclude_ids=set(), timeout_seconds=5)[0]["id"], "thread_1")
        with self.assertRaises(WorkflowStop):
            adapter.expand("thread", "other_1", "next", seen_segments=[], timeout_seconds=5)
        neighbor = adapter.expand("thread", "thread_1", "next",
                                  seen_segments=[{"id": "thread_1"}], timeout_seconds=5)
        self.assertEqual((rag.requested, neighbor[0]["id"]), ("thread", "thread_2"))

    def test_chroma_adapter_rejects_foreign_collection_rows(self):
        class CorruptRag:
            def list_chunks(self, thread_id):
                return [chunk("another_thread_1")]

        with self.assertRaisesRegex(WorkflowStop, "foreign_source_evidence"):
            ChromaPaperRetriever(CorruptRag()).has_paper("thread")

    def test_hybrid_search_promotes_named_mechanisms_and_centers_excerpt(self):
        prefix = "generic background " * 60

        class Rag:
            def list_chunks(self, thread_id):
                return [
                    chunk("thread_irrelevant", "unrelated experiment results " * 50),
                    chunk("thread_prob", prefix + "ProbSparse self-attention selects top queries by sparsity."),
                    chunk("thread_distill", prefix + "Self-attention distilling uses Conv1d and max pooling."),
                ]

            def search(self, thread_id, query, *, k, exclude_ids, pool, timeout_seconds=None):
                return [chunk("thread_irrelevant", "unrelated experiment results " * 50)]

        adapter = ChromaPaperRetriever(Rag())
        rows = adapter.search(
            "thread",
            "ProbSparse self-attention mechanism and self-attention distilling",
            k=3,
            exclude_ids=set(),
            timeout_seconds=5,
        )
        self.assertCountEqual([row["id"] for row in rows[:2]], ["thread_distill", "thread_prob"])
        self.assertTrue(all(row["_retrieval_source"] == "lexical" for row in rows[:2]))
        self.assertTrue(all(row["_start"] > 0 for row in rows[:2]))
        segments = _pack(rows, [], limit=4000, budget=ResearchBudget())
        evidence = " ".join(segment["text"] for segment in segments)
        self.assertIn("ProbSparse", evidence)
        self.assertIn("distilling", evidence)

    def test_hybrid_search_promotes_exact_component_definition_heading(self):
        class Rag:
            def list_chunks(self, thread_id):
                return [
                    chunk("thread_later", "In a later experiment and broad architecture discussion, the model "
                          "mentions series decomposition and Auto Correlation among many components, "
                          "without defining either operation in this paragraph."),
                    chunk("thread_decomp", "3.1 Series Decomposition Block\nMoving average extracts the trend."),
                    chunk("thread_auto", "3.2 Auto Correlation Mechanism\nIt discovers periodic dependencies."),
                ]

            def search(self, thread_id, query, *, k, exclude_ids, pool, timeout_seconds=None):
                return [chunk("thread_later", "A later semantic mention of both components.")]

        rows = ChromaPaperRetriever(Rag()).search(
            "thread",
            "Autoformer series decomposition Auto Correlation mechanism",
            k=3,
            exclude_ids=set(),
            timeout_seconds=5,
        )
        self.assertEqual([row["id"] for row in rows[:2]], ["thread_auto", "thread_decomp"])

    def test_repeated_page_title_is_not_treated_as_definition_heading(self):
        repeated_title = "Time Series Forecasting Methods A Systematic Review"

        class Rag:
            def list_chunks(self, thread_id):
                repeated = [
                    chunk(f"thread_header_{index}", repeated_title + "\nGeneric discussion of forecasting.")
                    for index in range(4)
                ]
                return repeated + [chunk(
                    "thread_taxonomy",
                    "2 Forecasting Method Taxonomy\nWe classify methods by model family and forecast setting.",
                )]

            def search(self, thread_id, query, *, k, exclude_ids, pool, timeout_seconds=None):
                return [chunk("thread_header_0", repeated_title)]

        rows = ChromaPaperRetriever(Rag()).search(
            "thread",
            "time series forecasting method taxonomy classification",
            k=3,
            exclude_ids=set(),
            timeout_seconds=5,
        )
        self.assertEqual(rows[0]["id"], "thread_taxonomy")

    def test_evaluation_query_prefers_evaluation_section_over_review_mentions(self):
        class Rag:
            def list_chunks(self, thread_id):
                related = chunk(
                    "thread_review",
                    "Prior reviews discuss forecasting evaluation metrics but provide limited detail.",
                )
                related["section"] = "related work"
                evaluation = chunk(
                    "thread_scores",
                    "Evaluation Scores\nPoint forecasting uses MSE and MAE. "
                    "Probabilistic forecasting uses NLL, CRPS, and quantile loss.",
                )
                evaluation["section"] = "evaluation"
                return [related, evaluation]

            def search(self, thread_id, query, *, k, exclude_ids, pool, timeout_seconds=None):
                return [self.list_chunks(thread_id)[0]]

        rows = ChromaPaperRetriever(Rag()).search(
            "thread",
            "evaluation metrics for point vs probabilistic forecasting",
            k=2,
            exclude_ids=set(),
            timeout_seconds=5,
        )
        self.assertEqual(rows[0]["id"], "thread_scores")

    def test_rejects_search_without_named_gap_and_deduplicates_same_excerpt(self):
        no_gap = assessment({"type": "search", "query": "再查"}, state="supported")
        model = FakeModel(assessment_data=no_gap)
        result = run_deep_research("方法？", "thread", FakeRetriever(), model)
        self.assertEqual((result["status"], result["stop_reason"]),
                         ("stopped", "missing_information_required"))

        retriever = FakeRetriever()
        retriever.rows = [chunk("thread_1", "同一片段" * 50),
                          chunk("thread_2", "同一片段" * 50)]
        retriever.search = lambda thread_id, query, *, k, exclude_ids, timeout_seconds: retriever.rows
        stop_model = FakeModel(assessment_data=assessment({"type": "stop"}, state="supported"),
                               final_data={"items": [], "unresolved": []})
        duplicate = run_deep_research("方法？", "thread", retriever, stop_model)
        self.assertEqual(duplicate["unique_evidence_ids"], 1)


if __name__ == "__main__":
    unittest.main()
