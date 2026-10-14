import unittest
from deep_research import _isolated_assessment, WorkflowStop, run_deep_research
from tests.test_deep_research import FakeModel, FakeRetriever, assessment


class AssessmentDegradationTests(unittest.TestCase):
    questions = [{"id": "q1", "text": "Question one"}, {"id": "q2", "text": "Question two"}]

    def row(self, qid, state="supported", missing=""):
        return {"subquestion_id": qid, "assessment": state, "evidence_ids": ["thread_1"],
                "reason": "Grounded explanation", "missing_information": missing}

    def test_refuted_premise_is_not_internal_conflict(self):
        data = {"subquestions": [self.row("q1", "refuted_premise")], "action": {"type": "stop"}}
        rows, action, kept, blocked, error = _isolated_assessment(data, self.questions[:1], {"thread_1"})
        self.assertEqual(rows[0]["assessment"], "refuted_premise")
        self.assertEqual((blocked, error, action["type"]), ([], None, "stop"))

    def test_invalid_row_does_not_discard_supported_neighbour(self):
        for state in ("unknown", "truncated_context", "conflicting_evidence", ["bad type"]):
            with self.subTest(state=state):
                data = {"subquestions": [self.row("q1", state), self.row("q2")],
                        "action": {"type": "search", "query": "Do not execute"}}
                _, action, kept, blocked, _ = _isolated_assessment(data, self.questions, {"thread_1"})
                self.assertEqual([q["id"] for q in kept], ["q2"])
                self.assertEqual(blocked[0]["subquestion_id"], "q1")
                self.assertEqual(action["type"], "stop")

    def test_explicit_gap_is_retained_without_inventing_search(self):
        data = {"subquestions": [self.row("q1", missing="Formula")], "action": {"type": "stop"}}
        rows, action, *_ = _isolated_assessment(data, self.questions[:1], {"thread_1"})
        self.assertEqual(rows[0]["assessment"], "missing_specific_fact")
        self.assertEqual(action, {"type": "stop"})

    def test_invalid_action_is_blocked_but_valid_assessment_survives(self):
        data = {"subquestions": [self.row("q1")], "action": {"type": "shell"}}
        rows, action, _, _, error = _isolated_assessment(data, self.questions[:1], {"thread_1"})
        self.assertEqual(rows[0]["assessment"], "supported")
        self.assertEqual((action["type"], error), ("stop", "invalid_action"))

    def test_foreign_evidence_is_still_a_hard_failure(self):
        data = {"subquestions": [dict(self.row("q1"), evidence_ids=["foreign"])], "action": {"type": "stop"}}
        with self.assertRaises(WorkflowStop):
            _isolated_assessment(data, self.questions[:1], {"thread_1"})

    def test_all_bad_assessments_allow_limited_final_without_tools(self):
        data = assessment({"type": "stop"}, state="conflicting_evidence")
        data["subquestions"][0]["missing_information"] = ""
        model = FakeModel(assessment_data=data, final_data={"items": [
            {"kind": "paper_fact", "text": "Scoped source fact", "evidence_ids": ["thread_1"]}]})
        retriever = FakeRetriever()
        result = run_deep_research("Explain", "thread", retriever, model)
        self.assertEqual((result["status"], result["stop_reason"]), ("partial", "assessment_validation_partial"))
        self.assertIn("Scoped source fact", result["answer"])
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(len([c for c in retriever.calls if c[0] == "search"]), 1)
        self.assertTrue(model.calls[-1][1]["assessment_unavailable"])

    def test_top_level_protocol_error_stops_tools_not_final_writing(self):
        rows, action, _, blocked, error = _isolated_assessment(
            {"subquestions": "invalid"}, self.questions, {"thread_1"})
        self.assertEqual((action["type"], error), ("stop", "assessment_unavailable"))
        self.assertEqual(len(blocked), 2)
        self.assertTrue(all(r["assessment"] == "unknown" for r in rows))

    def test_valid_subquestion_is_written_and_bad_one_is_named(self):
        questions = self.questions
        rows = [self.row("q1", "unknown"), self.row("q2")]
        class Model(FakeModel):
            def call(self, stage, payload, **limits):
                if stage == "plan":
                    self.calls.append((stage, payload, limits))
                    return {"data": {"subquestions": [dict(q, query=q["text"]) for q in questions]}, "usage": {}}
                return super().call(stage, payload, **limits)
        model = Model(assessment_data={"subquestions": rows, "action": {"type": "stop"}},
            final_data={"items": [{"kind": "paper_fact", "text": "Question two answer", "evidence_ids": ["thread_1"]}],
                        "coverage": [{"subquestion_id": "q2", "status": "answered", "reason": "Evidence", "evidence_ids": ["thread_1"]}]})
        result = run_deep_research("Explain both", "thread", FakeRetriever(), model)
        self.assertIn("Question two answer", result["answer"])
        self.assertIn("q1：Question one", result["answer"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual([q["id"] for q in model.calls[-1][1]["subquestions"]], ["q2"])
