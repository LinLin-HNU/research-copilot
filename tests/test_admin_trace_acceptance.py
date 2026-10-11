import json
import tempfile
import unittest
from pathlib import Path

from trace_admin import TraceRepository, summarize_trace


class AdminTraceAcceptanceTests(unittest.TestCase):
    def test_trace_summary_is_readable_and_keeps_admin_metrics(self):
        trace = {
            "request_id": "req-1",
            "thread_id": "paper-1",
            "mode": "paper_deep",
            "question": "方法是什么？",
            "status": "partial",
            "deep_retry_recommended": True,
            "stop_reason": "followup_incomplete",
            "model_calls": 3,
            "retrieval_calls": 2,
            "input_tokens": 4000,
            "output_tokens": 900,
            "elapsed_ms": 12500,
            "evidence_chars": 4100,
            "unique_evidence_ids": 6,
            "events": [
                {"step": "plan", "subquestions": [{"id": "q1", "text": "机制？"}]},
                {"step": "retrieval", "provided_segments": [{"id": "paper-1_2"}]},
                {"step": "assessment", "action": {"type": "search", "query": "补查机制"}},
                {"step": "gate", "allowed": True, "reason": "within_budget"},
                {"step": "finish", "stop_reason": "followup_incomplete"},
            ],
        }

        summary = summarize_trace(trace)

        self.assertEqual(summary["request_id"], "req-1")
        self.assertEqual(summary["usage"]["total_tokens"], 4900)
        self.assertEqual(summary["timing"]["elapsed_ms"], 12500)
        self.assertTrue(summary["decision"]["deep_retry_recommended"])
        self.assertEqual(
            [item["label"] for item in summary["timeline"]],
            ["规划", "首次检索", "证据判断", "动作门禁", "完成"],
        )
        self.assertEqual(summary["timeline"][3]["result"], "放行：within_budget")

    def test_repository_lists_newest_first_and_blocks_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "older.json").write_text(json.dumps({
                "request_id": "older", "events": [], "elapsed_ms": 1,
            }), encoding="utf-8")
            (root / "newer.json").write_text(json.dumps({
                "request_id": "newer", "events": [], "elapsed_ms": 2,
            }), encoding="utf-8")
            (root / "older.json").touch()
            repository = TraceRepository(root)

            self.assertEqual({item["request_id"] for item in repository.list(limit=10)}, {"older", "newer"})
            self.assertEqual(repository.get("newer")["request_id"], "newer")
            with self.assertRaisesRegex(ValueError, "invalid_request_id"):
                repository.get("../secrets")

    def test_corrupt_trace_is_skipped_in_list_and_reported_on_direct_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "broken.json").write_text("{broken", encoding="utf-8")
            repository = TraceRepository(root)
            self.assertEqual(repository.list(), [])
            with self.assertRaisesRegex(ValueError, "invalid_trace"):
                repository.get("broken")


if __name__ == "__main__":
    unittest.main()
