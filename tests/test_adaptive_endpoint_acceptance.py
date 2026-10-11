import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
os.environ.setdefault("DASHSCOPE_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from fastapi import HTTPException

import app as app_module
from deep_research_runtime import ResearchAccessGuard
from schemas import AdaptiveResearchRequest


TOKEN = "adaptive-endpoint-token-0123456789abcdef"


def internal_result() -> dict:
    return {
        "request_id": "req-adaptive-1",
        "mode": "general",
        "status": "ok",
        "stop_reason": "general_answer",
        "answer": "通用回答",
        "claims": [],
        "unresolved": [],
        "model_calls": 1,
        "retrieval_calls": 0,
        "input_tokens": 20,
        "output_tokens": 10,
        "elapsed_ms": 12.5,
        "steps": [{"step": "mode_selection"}],
    }


class AdaptiveEndpointAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_skips_reading_history_without_deleting_it(self):
        payload = AdaptiveResearchRequest(question="讲解论文", thread_id="paper", history_mode="fresh",
                                          enable_quality_review=False, confirm_external_processing=True)
        fake_rag = Mock()
        fake_rag.list_chunks.return_value = []
        with patch.object(app_module, "session_exists", return_value=True), \
                patch.object(app_module, "get_rag", return_value=fake_rag), \
                patch.object(app_module, "AliyunResearchModel", return_value=object()), \
                patch.object(app_module, "list_adaptive_turns") as history, \
                patch.object(app_module, "record_adaptive_turn"), \
                patch.object(app_module, "run_adaptive_research", return_value=internal_result()) as workflow:
            await app_module.adaptive_research_endpoint(payload, x_research_token=TOKEN)
        history.assert_not_called()
        self.assertEqual(workflow.call_args.kwargs["conversation_history"], [])
        self.assertEqual(workflow.call_args.kwargs["history_mode"], "fresh")
        self.assertFalse(workflow.call_args.kwargs["enable_quality_review"])

    def setUp(self):
        app_module._research_guard = ResearchAccessGuard(TOKEN)

    def tearDown(self):
        app_module._research_guard = None

    async def test_user_endpoint_returns_no_admin_metrics(self):
        payload = AdaptiveResearchRequest(
            question="什么是时间序列预测？",
            confirm_external_processing=True,
        )
        with patch.object(app_module, "AliyunResearchModel", return_value=object()), \
                patch.object(app_module, "run_adaptive_research", return_value=internal_result()):
            result = await app_module.adaptive_research_endpoint(
                payload, x_research_token=TOKEN
            )

        self.assertEqual(result["answer"], "通用回答")
        self.assertEqual(result["mode"], "general")
        for hidden in (
            "model_calls", "retrieval_calls", "input_tokens", "output_tokens",
            "elapsed_ms", "steps", "stop_reason",
        ):
            self.assertNotIn(hidden, result)

    async def test_paper_request_reads_and_records_thread_memory(self):
        payload = AdaptiveResearchRequest(
            question="继续解释复杂度",
            thread_id="paper-thread",
            strategy="auto",
            confirm_external_processing=True,
        )
        fake_rag = Mock()
        fake_rag.list_chunks.return_value = [{
            "id": "paper-thread_0", "content": "论文", "section": "method",
            "page_start": 1, "page_end": 1, "_order": 0,
        }]
        history = [{"question": "上一问", "answer": "上一答", "mode": "paper_deep"}]
        result = internal_result()
        result.update({"mode": "paper_deep", "answer": "连续回答"})
        with patch.object(app_module, "session_exists", return_value=True), \
                patch.object(app_module, "get_rag", return_value=fake_rag), \
                patch.object(app_module, "AliyunResearchModel", return_value=object()), \
                patch.object(app_module, "list_adaptive_turns", return_value=history), \
                patch.object(app_module, "record_adaptive_turn") as record, \
                patch.object(app_module, "run_adaptive_research", return_value=result) as workflow:
            response = await app_module.adaptive_research_endpoint(
                payload, x_research_token=TOKEN
            )

        self.assertEqual(workflow.call_args.kwargs["conversation_history"], history)
        self.assertEqual(workflow.call_args.kwargs["strategy"], "auto")
        self.assertEqual(workflow.call_args.kwargs["answer_depth"], "beginner")
        self.assertEqual(record.call_args.args[0]["answer"], "连续回答")
        self.assertEqual(response["mode"], "paper_deep")

    async def test_memory_write_failure_does_not_discard_paid_answer(self):
        payload = AdaptiveResearchRequest(
            question="继续解释复杂度",
            thread_id="paper-thread",
            confirm_external_processing=True,
        )
        fake_rag = Mock()
        fake_rag.list_chunks.return_value = [{
            "id": "paper-thread_0", "content": "论文", "section": "method",
            "page_start": 1, "page_end": 1, "_order": 0,
        }]
        result = internal_result()
        result.update({"mode": "paper_deep", "answer": "已经生成的付费回答"})
        with patch.object(app_module, "session_exists", return_value=True), \
                patch.object(app_module, "get_rag", return_value=fake_rag), \
                patch.object(app_module, "AliyunResearchModel", return_value=object()), \
                patch.object(app_module, "list_adaptive_turns", return_value=[]), \
                patch.object(app_module, "record_adaptive_turn", side_effect=OSError("disk full")), \
                patch.object(app_module.logger, "exception") as logged, \
                patch.object(app_module, "run_adaptive_research", return_value=result):
            response = await app_module.adaptive_research_endpoint(
                payload, x_research_token=TOKEN
            )

        self.assertEqual(response["answer"], "已经生成的付费回答")
        logged.assert_called_once()

    async def test_admin_trace_api_requires_token(self):
        with self.assertRaises(HTTPException) as captured:
            await app_module.list_research_traces(x_research_token=None)
        self.assertEqual(captured.exception.status_code, 401)

    async def test_admin_can_list_and_open_readable_trace(self):
        trace = {
            "request_id": "req-1",
            "mode": "paper_full",
            "status": "ok",
            "question": "方法？",
            "input_tokens": 100,
            "output_tokens": 20,
            "elapsed_ms": 1000,
            "events": [{"step": "mode_selection", "reason": "single_paper"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "req-1.json").write_text(
                json.dumps(trace, ensure_ascii=False), encoding="utf-8"
            )
            with patch.object(app_module, "ADAPTIVE_TRACE_DIR", root):
                listing = await app_module.list_research_traces(
                    x_research_token=TOKEN
                )
                detail = await app_module.get_research_trace(
                    "req-1", x_research_token=TOKEN
                )

        self.assertEqual(listing["items"][0]["usage"]["total_tokens"], 120)
        self.assertEqual(detail["timeline"][0]["label"], "模式选择")
        self.assertIn("detail", detail["timeline"][0])

    async def test_admin_page_file_exists(self):
        response = await app_module.research_trace_page()
        self.assertTrue(Path(response.path).is_file())

    async def test_adaptive_research_page_file_exists(self):
        response = await app_module.adaptive_research_page()
        self.assertTrue(Path(response.path).is_file())
        html = Path(response.path).read_text(encoding="utf-8")
        self.assertIn("请选择一篇论文", html)
        self.assertIn("入门详解（推荐）", html)
        self.assertIn("!paper.value", html)

    async def test_local_pdf_import_indexes_without_generating_answer(self):
        class PdfRequest:
            headers = {"content-type": "application/pdf", "content-length": "9"}

            async def stream(self):
                yield b"%PDF-test"

        fake_rag = Mock()
        with patch.object(app_module, "extract_text_from_bytes", return_value=("Sundial paper", [1])), \
                patch.object(app_module, "detect_sections", return_value=({"abstract@@1": "text"}, {"abstract@@1": [1]})), \
                patch.object(app_module, "chunk_sections", return_value=[{"content": "text", "section": "abstract"}]), \
                patch.object(app_module, "guess_title", return_value="Sundial"), \
                patch.object(app_module, "get_rag", return_value=fake_rag), \
                patch.object(app_module, "create_session") as create_session, \
                patch.object(app_module, "AliyunResearchModel") as generation_model:
            result = await app_module.import_research_pdf(
                PdfRequest(), x_research_token=TOKEN, x_confirm_embedding="true",
                x_paper_filename="2025_Sundial_A_Family.pdf",
            )

        self.assertEqual((result["title"], result["chunk_count"]), ("Sundial A Family", 1))
        self.assertEqual(result["page_count"], 1)
        fake_rag.store.assert_called_once()
        create_session.assert_called_once()
        generation_model.assert_not_called()

    async def test_pdf_import_requires_explicit_embedding_confirmation(self):
        class PdfRequest:
            headers = {"content-type": "application/pdf"}

            async def stream(self):
                raise AssertionError("unconfirmed PDF must not be read")
                yield b""

        with self.assertRaises(HTTPException) as captured:
            await app_module.import_research_pdf(
                PdfRequest(), x_research_token=TOKEN, x_confirm_embedding=None
            )
        self.assertEqual(captured.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
