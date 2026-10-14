import unittest

from deep_research_runtime import AliyunResearchModel, ResearchAccessError, ResearchAccessGuard


TOKEN = "local-research-token-0123456789abcdef"


class MutableClock:
    def __init__(self, value=1_700_000_000.0):
        self.value = value

    def __call__(self):
        return self.value


class ResearchAccessGuardTests(unittest.TestCase):
    def test_full_context_places_stable_paper_before_history_and_question(self):
        content = AliyunResearchModel._build_user_content("full_context", {
            "paper_context": "PAPER-CONTENT",
            "conversation": [{"question": "OLD-QUESTION", "answer": "OLD-ANSWER"}],
            "question": "NEW-QUESTION",
            "answer_depth": "advanced",
        })
        self.assertLess(content.index("PAPER-CONTENT"), content.index("OLD-QUESTION"))
        self.assertLess(content.index("OLD-QUESTION"), content.index("NEW-QUESTION"))
        self.assertIn("【讲解深度 answer_depth】\nadvanced", content)

    def test_beginner_contract_requires_explanation_not_term_listing(self):
        for stage in ("final", "full_context"):
            instruction = AliyunResearchModel._instruction(stage)
            self.assertIn("从未读过论文", instruction)
            self.assertIn("它是什么、如何工作、解决什么问题", instruction)
            self.assertIn("不得只罗列术语", instruction)

    def test_planner_does_not_invent_lineage_from_reading_history(self):
        instruction = AliyunResearchModel._instruction("plan")
        self.assertIn("不得预设本论文统一、继承或优于它们", instruction)

    def test_usage_extraction_keeps_cached_tokens_from_openai_metadata(self):
        usage = AliyunResearchModel._extract_usage(
            {"input_tokens": 100, "output_tokens": 20,
             "input_token_details": {"cache_read": 70}},
            {"token_usage": {"prompt_tokens": 100, "completion_tokens": 20,
                             "prompt_tokens_details": {"cached_tokens": 65}}},
        )
        self.assertEqual((usage["input_tokens"], usage["output_tokens"], usage["cached_tokens"]),
                         (100, 20, 70))

    def test_usage_extraction_falls_back_to_provider_token_details(self):
        usage = AliyunResearchModel._extract_usage(
            {}, {"token_usage": {"prompt_tokens": 90, "completion_tokens": 10,
                                  "prompt_tokens_details": {"cached_tokens": 55}}}
        )
        self.assertEqual(usage["cached_tokens"], 55)
        self.assertEqual(usage["provider_usage"]["prompt_tokens_details"]["cached_tokens"], 55)

    def test_missing_cache_is_not_reported_as_a_known_zero(self):
        missing = AliyunResearchModel._extract_usage({"input_tokens": 10, "output_tokens": 2}, {})
        self.assertIsNone(missing["cached_tokens"])
        known_zero = AliyunResearchModel._extract_usage(
            {"input_token_details": {"cache_read": 0}}, {})
        self.assertEqual(known_zero["cached_tokens"], 0)

    def test_invalid_json_package_keeps_raw_output_and_usage(self):
        usage = {"input_tokens": 120, "output_tokens": 35}
        response = AliyunResearchModel._package_response("{not-json", usage)
        self.assertIsNone(response["data"])
        self.assertEqual(response["raw"], "{not-json")
        self.assertEqual(response["usage"], usage)
        self.assertEqual(response["parse_error"], "invalid_model_json")

    def test_rejects_bad_token_and_missing_external_confirmation(self):
        guard = ResearchAccessGuard(TOKEN)
        with self.assertRaises(ResearchAccessError) as bad_token:
            guard.begin("wrong-token", True)
        self.assertEqual(bad_token.exception.status_code, 401)
        with self.assertRaises(ResearchAccessError) as no_confirm:
            guard.begin(TOKEN, False)
        self.assertEqual(no_confirm.exception.status_code, 400)
        self.assertEqual(guard.snapshot()["active_requests"], 0)

    def test_concurrency_limit_and_cancel_release_reservation(self):
        guard = ResearchAccessGuard(TOKEN, max_concurrent=1)
        first = guard.begin(TOKEN, True)
        with self.assertRaises(ResearchAccessError) as concurrent:
            guard.begin(TOKEN, True)
        self.assertEqual(concurrent.exception.status_code, 429)
        guard.cancel(first)
        self.assertEqual(guard.snapshot()["reserved_tokens"], 0)

    def test_hourly_limit_counts_completed_requests(self):
        clock = MutableClock()
        guard = ResearchAccessGuard(TOKEN, max_requests_per_hour=1, clock=clock)
        request_id = guard.begin(TOKEN, True)
        guard.finish(request_id, 100)
        with self.assertRaises(ResearchAccessError) as limited:
            guard.begin(TOKEN, True)
        self.assertIn("每小时", limited.exception.detail)
        clock.value += 3601
        second = guard.begin(TOKEN, True)
        guard.finish(second, 50)

    def test_daily_budget_reserves_before_call_and_charges_actual_usage(self):
        guard = ResearchAccessGuard(
            TOKEN,
            max_tokens_per_day=100,
            token_reservation=60,
        )
        request_id = guard.begin(TOKEN, True)
        guard.finish(request_id, 80)
        self.assertEqual(guard.snapshot()["charged_tokens_today"], 80)
        with self.assertRaises(ResearchAccessError) as limited:
            guard.begin(TOKEN, True)
        self.assertIn("每日", limited.exception.detail)

    def test_missing_usage_charges_full_reservation(self):
        guard = ResearchAccessGuard(
            TOKEN,
            max_tokens_per_day=100,
            token_reservation=60,
        )
        request_id = guard.begin(TOKEN, True)
        guard.finish(request_id, None)
        snapshot = guard.snapshot()
        self.assertEqual(snapshot["charged_tokens_today"], 60)
        self.assertEqual(snapshot["active_requests"], 0)


if __name__ == "__main__":
    unittest.main()
