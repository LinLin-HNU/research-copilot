import unittest
from unittest.mock import patch

from deep_research_runtime import AliyunResearchModel
from research_model_settings import resolve_research_model, build_chat_model


class ResearchModelSettingsTests(unittest.TestCase):
    def test_shared_factory_keeps_limits_and_generic_provider_clean(self):
        settings = resolve_research_model({"OPENAI_BASE_URL": "https://example.invalid/v1",
                         "MODEL_NAME": "test-model", "OPENAI_API_KEY": "test-key"})
        with patch("langchain_openai.ChatOpenAI") as client:
            build_chat_model(settings=settings, max_output_tokens=8192, timeout_seconds=60)
        self.assertEqual(client.call_args.kwargs["max_retries"], 0)
        self.assertEqual(client.call_args.kwargs["temperature"], 0)
        self.assertEqual(client.call_args.kwargs["max_tokens"], 8192)
        self.assertEqual(client.call_args.kwargs["extra_body"], {})

    def test_legacy_config_uses_shared_factory_and_does_not_send_request(self):
        import runpy
        from pathlib import Path
        with patch("research_model_settings.build_chat_model") as factory, patch("dotenv.load_dotenv"):
            namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "config.py"))
        factory.assert_called_once_with(max_output_tokens=8192, timeout_seconds=60)
        self.assertIs(namespace["model"], factory.return_value)

    def test_router_does_not_override_provider_thinking_options(self):
        import query_router
        from types import SimpleNamespace
        with patch.object(query_router, "model") as model:
            model.bind.return_value.invoke.return_value = SimpleNamespace(content='{"category":"A1"}')
            self.assertEqual(query_router.classify_question("paper method"), "A1")
        self.assertNotIn("extra_body", model.bind.call_args.kwargs)

    def test_deepseek_switch_uses_its_key_not_old_alibaba_connection(self):
        env = {"OPENAI_BASE_URL": "https://api.deepseek.com", "MODEL_NAME": "deepseek-flash",
               "DEEPSEEK_API_KEY": "new-provider-key", "DASHSCOPE_API_KEY": "old-provider-key",
               "DASHSCOPE_BASE_URL": "https://old.aliyuncs.com/v1"}
        settings = resolve_research_model(env)
        self.assertEqual(settings.api_key, "new-provider-key")
        self.assertEqual(settings.name, "deepseek-flash")
        self.assertEqual(settings.extra_body, {"thinking": {"type": "disabled"}})
        self.assertNotIn("new-provider-key", repr(settings))
        with patch.dict("os.environ", env, clear=True), patch("deep_research_runtime.load_dotenv"):
            model = AliyunResearchModel()
            self.assertEqual(model._base, "https://api.deepseek.com")
            metadata = model.describe_request("full_context", {"question": "test"},
                        max_output_tokens=50, timeout_seconds=1, settings=model._settings)
            self.assertEqual(metadata["provider_host"], "api.deepseek.com")
            self.assertEqual(metadata["model"], "deepseek-flash")

    def test_missing_new_key_does_not_use_old_provider_key(self):
        with self.assertRaisesRegex(RuntimeError, "Missing API key"):
            resolve_research_model({"OPENAI_BASE_URL": "https://api.deepseek.com",
                                    "DASHSCOPE_API_KEY": "old"})

    def test_legacy_alibaba_and_explicit_research_override(self):
        old = resolve_research_model({"DASHSCOPE_BASE_URL": "https://old.aliyuncs.com/v1", "DASHSCOPE_API_KEY": "old"})
        self.assertEqual(old.name, "deepseek-v4-flash-0731")
        self.assertEqual(old.extra_body, {"enable_thinking": False})
        override = resolve_research_model({"OPENAI_BASE_URL": "https://api.deepseek.com",
                        "MODEL_NAME": "deepseek-flash", "RESEARCH_BASE_URL": "https://example.invalid/v1",
                        "RESEARCH_MODEL_NAME": "test-model", "RESEARCH_API_KEY": "dedicated"})
        self.assertEqual((override.name, override.host, override.api_key), ("test-model", "example.invalid", "dedicated"))

    def test_actual_client_receives_selected_model_and_provider_parameters(self):
        from langchain_core.messages import AIMessage
        model = AliyunResearchModel.__new__(AliyunResearchModel)
        model._settings = resolve_research_model({"OPENAI_BASE_URL": "https://api.deepseek.com",
                            "MODEL_NAME": "deepseek-flash", "DEEPSEEK_API_KEY": "test-key"})
        model._key, model._base = model._settings.api_key, model._settings.base_url
        with patch("langchain_openai.ChatOpenAI") as client:
            client.return_value.invoke.return_value = AIMessage(content='{"items":[],"unresolved":[]}',
                                                                response_metadata={"finish_reason": "stop"})
            response = model.call("general", {"question": "test"}, max_output_tokens=50, timeout_seconds=1)
        kwargs = client.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-flash")
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(response["request_meta"]["model"], "deepseek-flash")
        self.assertNotIn("test-key", str(response))
