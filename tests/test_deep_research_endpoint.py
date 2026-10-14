import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
os.environ.setdefault("DASHSCOPE_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from fastapi.testclient import TestClient

import app as app_module
from deep_research_runtime import ResearchAccessGuard


TOKEN = "endpoint-test-token-0123456789abcdef"


def fake_result():
    return {
        "request_id": "request-1",
        "status": "ok",
        "stop_reason": "general_answer",
        "answer": "受限测试回答",
        "claims": [],
        "unresolved": [],
        "model_calls": 1,
        "actions": 0,
        "evidence_chars": 0,
        "unique_evidence_ids": 0,
        "input_tokens": 10,
        "output_tokens": 5,
        "usage_missing_calls": 0,
        "external_payload_chars": 80,
        "elapsed_ms": 1.0,
        "steps": [],
    }


class DeepResearchEndpointTests(unittest.TestCase):
    def setUp(self):
        app_module._research_guard = ResearchAccessGuard(TOKEN)
        self.client = TestClient(app_module.app)

    def tearDown(self):
        app_module._research_guard = None

    def test_requires_header_token(self):
        response = self.client.post("/research/deep", json={
            "question": "什么是时间序列预测？",
            "mode": "general",
            "confirm_external_processing": True,
        })
        self.assertEqual(response.status_code, 401)

    def test_requires_explicit_external_processing_confirmation(self):
        response = self.client.post(
            "/research/deep",
            headers={"X-Research-Token": TOKEN},
            json={"question": "什么是时间序列预测？", "mode": "general"},
        )
        self.assertEqual(response.status_code, 400)

    def test_approved_general_request_returns_limits_and_usage(self):
        with patch.object(app_module, "AliyunResearchModel", return_value=object()), \
                patch.object(app_module, "run_deep_research", return_value=fake_result()) as workflow:
            response = self.client.post(
                "/research/deep",
                headers={"X-Research-Token": TOKEN},
                json={
                    "question": "什么是时间序列预测？",
                    "mode": "general",
                    "confirm_external_processing": True,
                },
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["local_limits"]["charged_tokens_today"], 15)
        self.assertEqual(body["external_processing"]["payload_chars"], 80)
        self.assertTrue(body["external_processing"]["confirmed"])
        self.assertEqual(workflow.call_args.kwargs["mode"], "general")


if __name__ == "__main__":
    unittest.main()
