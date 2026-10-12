"""Resolve the research endpoint's connection without crossing provider keys."""
import os
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass(frozen=True)
class ResearchModelSettings:
    name: str
    base_url: str
    api_key: str = field(repr=False)
    extra_body: dict = field(default_factory=dict)

    @property
    def host(self):
        return urlparse(self.base_url).hostname


def resolve_research_model(env=None, *, require_key=True):
    values = os.environ if env is None else env
    base = (values.get("RESEARCH_BASE_URL") or values.get("OPENAI_BASE_URL")
            or values.get("DASHSCOPE_BASE_URL") or "").strip().rstrip("/")
    host = urlparse(base).hostname or ""
    if not base and require_key:
        raise RuntimeError("Missing research model base URL")
    name = (values.get("RESEARCH_MODEL_NAME") or values.get("MODEL_NAME")
            or values.get("DASHSCOPE_MODEL_NAME") or "deepseek-v4-flash-0731").strip()
    key = values.get("RESEARCH_API_KEY")
    if host == "api.deepseek.com":
        key = key or values.get("DEEPSEEK_API_KEY") or values.get("OPENAI_API_KEY")
        extra = {"thinking": {"type": "disabled"}}
    elif host.endswith("aliyuncs.com"):
        key = key or values.get("DASHSCOPE_API_KEY")
        extra = {"enable_thinking": False}
    else:
        key = key or values.get("OPENAI_API_KEY")
        extra = {}
    if require_key and not key:
        raise RuntimeError("Missing API key for the selected research provider")
    return ResearchModelSettings(name=name, base_url=base, api_key=key or "", extra_body=extra)


def build_chat_model(*, max_output_tokens, timeout_seconds, settings=None):
    """Shared server-side connection policy with caller-specific output caps."""
    from langchain_openai import ChatOpenAI

    selected = settings or resolve_research_model()
    client_type = ChatOpenAI
    if selected.host == "api.deepseek.com":
        from provider_chat_model import DeepSeekChatOpenAI
        client_type = DeepSeekChatOpenAI
    return client_type(
        model=selected.name, api_key=selected.api_key,
        base_url=selected.base_url, temperature=0,
        timeout=timeout_seconds, max_tokens=max_output_tokens,
        max_retries=0, extra_body=selected.extra_body,
    )
