"""Tests for LLM provider response handling."""
import pytest

from podcast_scout.providers.base import LLMMessage
from podcast_scout.providers.llm import GeminiProvider, _extract_gemini_text

# ---------------------------------------------------------------------------
# Gemini response extraction.
#
# gemini-2.5-flash defaults to dynamic thinking and thinking tokens count
# against maxOutputTokens. When they exhaust it the API returns
# finishReason=MAX_TOKENS with NO parts. The old code did
# candidates[0].content.parts[0].text and raised a bare IndexError, which the
# batch ranker swallowed into metadata-only scoring -- 11 of 14 episodes in one
# production run lost their summary and key ideas with no usable log line.
# ---------------------------------------------------------------------------

def test_empty_parts_raises_with_diagnostic_detail():
    data = {
        "candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}],
        "usageMetadata": {"thoughtsTokenCount": 15000, "promptTokenCount": 4200},
    }
    with pytest.raises(RuntimeError) as exc:
        _extract_gemini_text(data, max_tokens=15000)
    msg = str(exc.value)
    assert "MAX_TOKENS" in msg
    assert "thoughtsTokens=15000" in msg
    assert "thinkingConfig" in msg


def test_missing_content_key_does_not_raise_keyerror():
    data = {"candidates": [{"finishReason": "MAX_TOKENS"}], "usageMetadata": {}}
    with pytest.raises(RuntimeError):
        _extract_gemini_text(data, max_tokens=100)


def test_no_candidates_surfaces_prompt_feedback():
    data = {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
    with pytest.raises(RuntimeError) as exc:
        _extract_gemini_text(data, max_tokens=100)
    assert "SAFETY" in str(exc.value)


def test_healthy_response_returns_text():
    data = {
        "candidates": [{"finishReason": "STOP",
                        "content": {"parts": [{"text": '[{"a":1}]'}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }
    assert _extract_gemini_text(data, max_tokens=100) == '[{"a":1}]'


def test_multiple_parts_are_joined():
    """Reading only parts[0] silently truncated the JSON array."""
    data = {
        "candidates": [{"finishReason": "STOP",
                        "content": {"parts": [{"text": '[{"a":'}, {"text": '1}]'}]}}],
        "usageMetadata": {},
    }
    assert _extract_gemini_text(data, max_tokens=100) == '[{"a":1}]'


def test_truncated_but_nonempty_still_returns(caplog):
    """MAX_TOKENS with partial text: return it so json_repair can salvage."""
    data = {
        "candidates": [{"finishReason": "MAX_TOKENS",
                        "content": {"parts": [{"text": '[{"a":1},{"b":'}]}}],
        "usageMetadata": {"thoughtsTokenCount": 9000},
    }
    out = _extract_gemini_text(data, max_tokens=10000)
    assert out.startswith("[{")


# ---------------------------------------------------------------------------
# thinkingConfig is model-generation specific and must be omittable.
# ---------------------------------------------------------------------------

def _payload(provider):
    """Build the request payload the provider would POST."""
    import asyncio
    from unittest.mock import patch

    captured = {}

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "ok"}]},
                                    "finishReason": "STOP"}],
                    "usageMetadata": {}}

    class _Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None, headers=None):
            captured.update(json)
            return _Resp()

    with patch("podcast_scout.providers.llm.httpx.AsyncClient", lambda **kw: _Client()):
        asyncio.run(provider.complete([LLMMessage(role="user", content="hi")]))
    return captured


def test_thinking_budget_sent_by_default():
    p = GeminiProvider("k", "gemini-3.6-flash")
    cfg = _payload(p)["generationConfig"]
    assert cfg["thinkingConfig"] == {"thinkingBudget": 0}


def test_thinking_config_omitted_when_none():
    """A model generation that rejects the field must still be usable."""
    p = GeminiProvider("k", "gemini-3.6-flash", thinking_budget=None)
    assert "thinkingConfig" not in _payload(p)["generationConfig"]


def test_custom_thinking_budget_is_passed_through():
    p = GeminiProvider("k", "gemini-3.6-flash", thinking_budget=512)
    assert _payload(p)["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 512}


# ---------------------------------------------------------------------------
# Error bodies must survive, and thinkingConfig must self-heal.
#
# A live run failed with 400 on every Stage 2 call. The log said only
# "Client error '400 Bad Request'" because raise_for_status() discards the
# body -- so the field Google was actually objecting to was invisible.
# ---------------------------------------------------------------------------

class _Seq:
    """Client returning a scripted sequence of (status, body) responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.payloads = []

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def post(self, url, json=None, headers=None):
        import copy
        self.payloads.append(copy.deepcopy(json))
        status, body = self.responses.pop(0)

        class _R:
            status_code = status
            text = body if isinstance(body, str) else __import__("json").dumps(body)
            def json(self_inner):
                return body if isinstance(body, dict) else {}
        return _R()


_OK = {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
       "usageMetadata": {}}


def _run(provider, client):
    import asyncio
    from unittest.mock import patch
    with patch("podcast_scout.providers.llm.httpx.AsyncClient", lambda **kw: client):
        return asyncio.run(provider.complete([LLMMessage(role="user", content="hi")]))


def test_error_body_is_surfaced_not_swallowed():
    body = '{"error":{"code":400,"message":"Unknown name \\"thinkingConfig\\""}}'
    client = _Seq((400, body), (400, body))
    with pytest.raises(RuntimeError) as exc:
        _run(GeminiProvider("k", "m", thinking_budget=None), client)
    assert "400" in str(exc.value)
    assert "Unknown name" in str(exc.value), "the API's own message must reach the log"


def test_thinking_config_rejection_retries_without_it():
    """A model that rejects the field must not fail the entire run."""
    body = '{"error":{"code":400,"message":"Unknown name \\"thinkingConfig\\""}}'
    client = _Seq((400, body), (200, _OK))
    resp = _run(GeminiProvider("k", "m", thinking_budget=0), client)
    assert resp.content == "ok"
    assert "thinkingConfig" in client.payloads[0]["generationConfig"]
    assert "thinkingConfig" not in client.payloads[1]["generationConfig"]


def test_generic_400_still_retries_without_thinking_config():
    """Google does not always name the field.

    A live gemini-3.6-flash run returned only "Request contains an invalid
    argument" with no mention of thinkingConfig, so a name-matched retry never
    fired and every Stage 2 call failed.
    """
    body = '{"error":{"code":400,"message":"Request contains an invalid argument."}}'
    client = _Seq((400, body), (200, _OK))
    resp = _run(GeminiProvider("k", "m", thinking_budget=0), client)
    assert resp.content == "ok"
    assert "thinkingConfig" not in client.payloads[1]["generationConfig"]


def test_400_without_thinking_config_is_not_retried():
    """Nothing left to drop, so a second identical request is pure waste."""
    body = '{"error":{"code":400,"message":"Request payload size exceeds the limit"}}'
    client = _Seq((400, body))
    with pytest.raises(RuntimeError) as exc:
        _run(GeminiProvider("k", "m", thinking_budget=None), client)
    assert "payload size" in str(exc.value)
    assert len(client.payloads) == 1


def test_api_key_is_not_in_the_url():
    """httpx puts the URL in every exception message."""
    client = _Seq((400, '{"error":"boom"}'))
    with pytest.raises(RuntimeError) as exc:
        _run(GeminiProvider("SECRET-KEY", "m", thinking_budget=None), client)
    assert "SECRET-KEY" not in str(exc.value)
