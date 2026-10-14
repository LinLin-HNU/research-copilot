"""Small Chat Completions adapter; no calls or retries happen on construction."""
from langchain_openai import ChatOpenAI


class DeepSeekChatOpenAI(ChatOpenAI):
    """Keep DeepSeek's max_tokens parameter on the serialized request.

    The installed SDK rewrites it to max_completion_tokens for OpenAI. Only
    the explicitly selected DeepSeek host uses this adapter. Actual provider
    enforcement still needs a separately approved live check.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload
