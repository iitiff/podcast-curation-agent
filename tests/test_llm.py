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


# ---------------------------------------------------------------------------
# Rate limiting. Free-tier Gemini allows 20 requests/minute, and a live run
# exhausted it halfway through because every call was costing two requests.
# ---------------------------------------------------------------------------

_429 = ('{"error":{"code":429,"message":"Quota exceeded for metric: '
        'generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash. '
        'Please retry in 49.83645608s.","status":"RESOURCE_EXHAUSTED"}}')


def test_retry_delay_is_parsed_from_the_body():
    from podcast_scout.providers.llm import _retry_delay_seconds
    assert _retry_delay_seconds(_429) == pytest.approx(49.83645608)
    assert _retry_delay_seconds('{"error":"no delay here"}') is None


def test_429_waits_the_supplied_delay_then_retries(monkeypatch):
    import podcast_scout.providers.llm as mod
    slept = []

    async def _fake_sleep(seconds): slept.append(seconds)
    monkeypatch.setattr(mod.asyncio, "sleep", _fake_sleep)

    client = _Seq((429, _429), (200, _OK))
    resp = _run(GeminiProvider("k", "m", thinking_budget=None), client)
    assert resp.content == "ok"
    assert slept == [pytest.approx(49.83645608)]


def test_excessive_retry_delay_is_not_waited_out(monkeypatch):
    """A daily job should fail fast, not sleep for minutes."""
    import podcast_scout.providers.llm as mod
    slept = []

    async def _fake_sleep(seconds): slept.append(seconds)
    monkeypatch.setattr(mod.asyncio, "sleep", _fake_sleep)

    body = _429.replace("49.83645608", "600")
    client = _Seq((429, body))
    with pytest.raises(RuntimeError):
        _run(GeminiProvider("k", "m", thinking_budget=None), client)
    assert slept == []


def test_thinking_config_disable_is_sticky_across_calls():
    """Otherwise every call costs two requests and halves the rate limit."""
    body = '{"error":{"code":400,"message":"Request contains an invalid argument."}}'
    provider = GeminiProvider("k", "m", thinking_budget=0)

    first = _Seq((400, body), (200, _OK))
    _run(provider, first)
    assert provider.thinking_budget is None

    second = _Seq((200, _OK))
    _run(provider, second)
    assert "thinkingConfig" not in second.payloads[0]["generationConfig"]
    assert len(second.payloads) == 1, "second call must not re-probe"


def test_waiting_stops_once_a_backoff_proves_futile(monkeypatch):
    """A daily cap cannot be waited out a minute at a time.

    A live run slept 60s seven times and was still 429 each time.
    """
    import podcast_scout.providers.llm as mod
    slept = []

    async def _fake_sleep(seconds): slept.append(seconds)
    monkeypatch.setattr(mod.asyncio, "sleep", _fake_sleep)

    provider = GeminiProvider("k", "m", thinking_budget=None)

    # First call: waits once, still 429 -> marks the run rate-limited.
    with pytest.raises(RuntimeError):
        _run(provider, _Seq((429, _429), (429, _429)))
    assert len(slept) == 1
    assert provider._rate_limited is True

    # Second call: fails immediately, no further sleeping.
    with pytest.raises(RuntimeError):
        _run(provider, _Seq((429, _429)))
    assert len(slept) == 1, "must not keep waiting once waiting is known futile"


# ---------------------------------------------------------------------------
# The OpenAI-compatible fallback swallowed its error bodies the same way
# Gemini did. A live run showed only "Client error '410 Gone'" when the
# default NVIDIA NIM endpoint turned out to be retired.
# ---------------------------------------------------------------------------

def test_fallback_error_body_is_surfaced():
    import asyncio
    from unittest.mock import patch

    from podcast_scout.providers.llm import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        api_key="k", base_url="https://integrate.api.nvidia.com/v1",
        model="m", provider_name="NVIDIA NIM",
    )
    client = _Seq((410, '{"error":"endpoint retired"}'))
    with patch("podcast_scout.providers.llm.httpx.AsyncClient", lambda **kw: client), \
         pytest.raises(RuntimeError) as exc:
        asyncio.run(provider.complete([LLMMessage(role="user", content="hi")]))
    msg = str(exc.value)
    assert "410" in msg and "endpoint retired" in msg
    # NVIDIA's 410 means a missing org permission, not a wrong URL, so the
    # message must not send the reader off to change the base URL.
    assert "Public API Endpoints" in msg
    assert "LLM_FALLBACK_BASE_URL will not help" in msg


# ---------------------------------------------------------------------------
# Model end-of-life. A pinned model id is correct for reproducibility right up
# until the provider retires it, at which point the pin is just an outage:
#   "meta/llama-3.3-70b-instruct has reached its end of life on 2026-08-26"
# ---------------------------------------------------------------------------

_EOL = ('{"status":410,"detail":"The model \'meta/llama-3.3-70b-instruct\' has '
        'reached its end of life on 2026-08-26T09:00:00Z and is no longer available."}')


class _SeqWithModels(_Seq):
    """_Seq plus a GET /models listing, as a real endpoint offers."""

    def __init__(self, *responses, models=()):
        super().__init__(*responses)
        self._models = list(models)

    async def get(self, url, headers=None):
        rows = [{"id": m} for m in self._models]
        class _R:
            status_code = 200
            text = ""
            def json(self_inner): return {"data": rows}
        return _R()


def _run_compat(provider, client):
    import asyncio
    from unittest.mock import patch
    with patch("podcast_scout.providers.llm.httpx.AsyncClient", lambda **kw: client):
        return asyncio.run(provider.complete([LLMMessage(role="user", content="hi")]))


def _compat(model="meta/llama-3.3-70b-instruct"):
    from podcast_scout.providers.llm import OpenAICompatibleProvider
    return OpenAICompatibleProvider(
        api_key="k", base_url="https://x.example/v1", model=model, provider_name="Test",
    )


def test_retired_model_is_replaced_from_the_live_listing():
    provider = _compat()
    client = _SeqWithModels(
        (410, _EOL), (200, {"choices": [{"message": {"content": "ok"}}], "usage": {}}),
        models=["nvidia/embed-qa-4", "meta/llama-4-maverick-instruct", "qwen/qwen3-32b"],
    )
    resp = _run_compat(provider, client)
    assert resp.content == "ok"
    # Same family as the model that died, not merely first alphabetically.
    assert provider.model == "meta/llama-4-maverick-instruct"
    assert client.payloads[1]["model"] == "meta/llama-4-maverick-instruct"


def test_embedding_and_rerank_models_are_never_selected():
    provider = _compat("some/dead-model")
    client = _SeqWithModels(
        (404, '{"error":"not found"}'),
        (200, {"choices": [{"message": {"content": "ok"}}], "usage": {}}),
        models=["nvidia/rerank-qa", "baai/bge-m3-embed", "mistral/mistral-small-instruct"],
    )
    _run_compat(provider, client)
    assert provider.model == "mistral/mistral-small-instruct"


def test_free_variants_are_preferred():
    provider = _compat("meta/llama-3.3-70b-instruct")
    client = _SeqWithModels(
        (410, _EOL), (200, {"choices": [{"message": {"content": "ok"}}], "usage": {}}),
        models=["meta-llama/llama-4-scout", "meta-llama/llama-4-scout:free"],
    )
    _run_compat(provider, client)
    assert provider.model.endswith(":free"), "a zero-cost fallback should stay zero-cost"


def test_resolution_is_attempted_only_once():
    """A second failure means the model id was never the problem."""
    provider = _compat()
    client = _SeqWithModels(
        (410, _EOL), (410, _EOL), models=["meta/llama-4-maverick-instruct"],
    )
    with pytest.raises(RuntimeError):
        _run_compat(provider, client)
    assert provider._model_resolved is True
    assert len(client.payloads) == 2, "must not loop re-resolving"


def test_code_models_are_never_chosen_as_a_chat_replacement():
    """A live run picked meta/codellama-70b to replace llama-3.3-70b-instruct
    purely on the shared 'meta' prefix, then 404'd on every call."""
    provider = _compat("meta/llama-3.3-70b-instruct")
    client = _SeqWithModels(
        (410, _EOL), (200, {"choices": [{"message": {"content": "ok"}}], "usage": {}}),
        models=["meta/codellama-70b", "meta/llama-4-maverick-instruct"],
    )
    _run_compat(provider, client)
    assert provider.model == "meta/llama-4-maverick-instruct"


def test_instruct_model_beats_a_closer_family_match():
    """Following a rubric prompt matters more than sharing a vendor prefix."""
    provider = _compat("meta/llama-3.3-70b-instruct")
    client = _SeqWithModels(
        (410, _EOL), (200, {"choices": [{"message": {"content": "ok"}}], "usage": {}}),
        models=["meta/llama-4-base", "qwen/qwen3-32b-instruct"],
    )
    _run_compat(provider, client)
    assert provider.model == "qwen/qwen3-32b-instruct"


# ---------------------------------------------------------------------------
# A model id is not portable between providers. An NVIDIA-style
# "meta/llama-3.3-70b-instruct" handed to OpenRouter returns
# 400 "... is not a valid model ID" -- OpenRouter spells it "meta-llama/...".
# ---------------------------------------------------------------------------

def test_model_error_detection():
    from podcast_scout.providers.llm import _is_model_error
    assert _is_model_error(404, "{}") is True
    assert _is_model_error(410, "end of life") is True
    assert _is_model_error(400, '"meta/llama-3.3-70b is not a valid model ID"') is True
    assert _is_model_error(400, '"unknown model"') is True
    # A genuine request problem must not trigger a model swap.
    assert _is_model_error(400, '"Request contains an invalid argument."') is False
    assert _is_model_error(429, "quota exceeded") is False
    assert _is_model_error(503, "high demand") is False


def test_cross_provider_model_id_is_re_resolved():
    provider = _compat("meta/llama-3.3-70b-instruct")
    client = _SeqWithModels(
        (400, '{"error":{"message":"meta/llama-3.3-70b-instruct is not a valid model ID"}}'),
        (200, {"choices": [{"message": {"content": "ok"}}], "usage": {}}),
        models=["meta-llama/llama-3.3-70b-instruct", "openai/gpt-4o-mini"],
    )
    resp = _run_compat(provider, client)
    assert resp.content == "ok"
    assert provider.model == "meta-llama/llama-3.3-70b-instruct"


def test_overloaded_model_is_retried(monkeypatch):
    """503 is the provider being busy, not a bad request."""
    import podcast_scout.providers.llm as mod
    slept = []

    async def _fake_sleep(seconds): slept.append(seconds)
    monkeypatch.setattr(mod.asyncio, "sleep", _fake_sleep)

    body = '{"error":{"code":503,"message":"This model is currently experiencing high demand."}}'
    client = _Seq((503, body), (200, _OK))
    resp = _run(GeminiProvider("k", "m", thinking_budget=None), client)
    assert resp.content == "ok"
    assert len(slept) == 1


def test_persistent_overload_gives_up_for_the_fallback(monkeypatch):
    """The second provider is standing by; don't burn the run on retries."""
    import podcast_scout.providers.llm as mod

    async def _fake_sleep(seconds): return None
    monkeypatch.setattr(mod.asyncio, "sleep", _fake_sleep)

    body = '{"error":{"code":503,"message":"high demand"}}'
    client = _Seq((503, body), (503, body), (503, body))
    with pytest.raises(RuntimeError) as exc:
        _run(GeminiProvider("k", "m", thinking_budget=None), client)
    assert "503" in str(exc.value)
    assert len(client.payloads) == 3, "initial attempt plus two retries"


# ---------------------------------------------------------------------------
# Daily quota is per MODEL, so rotate before giving up on Gemini
# ---------------------------------------------------------------------------

class _StubClient:
    """Stands in for httpx.AsyncClient's .get for the models listing."""

    def __init__(self, payload=None, status=200):
        self.payload = payload if payload is not None else {"models": []}
        self.status = status
        self.calls = 0

    async def get(self, url, params=None, headers=None):
        self.calls += 1

        class _R:
            status_code = self.status
            def json(_self):
                return self.payload

        return _R()


def _gemini(**kw):
    from podcast_scout.providers.llm import GeminiProvider

    return GeminiProvider("key", kw.pop("model", "model-a"), **kw)


def _listing(*names):
    return {"models": [
        {"name": f"models/{n}", "supportedGenerationMethods": ["generateContent"]}
        for n in names
    ]}


async def test_rotation_moves_to_the_next_model_and_persists():
    g = _gemini(model_fallbacks=["model-b", "model-c"])

    assert await g._rotate_model(_StubClient()) is True
    assert g.model == "model-b"
    # Spent pairs are (key index, model): the allowance is per pair.
    assert (0, "model-a") in g._exhausted


async def test_rotation_skips_models_already_known_spent():
    g = _gemini(model_fallbacks=["model-b", "model-c"])
    c = _StubClient()
    await g._rotate_model(c)
    await g._rotate_model(c)

    assert g.model == "model-c"
    assert g._exhausted == {(0, "model-a"), (0, "model-b")}


async def test_rotation_gives_up_once_every_model_is_spent():
    """Returning False is what lets the caller fall through to the secondary."""
    g = _gemini(model_fallbacks=["model-b"])
    c = _StubClient()
    await g._rotate_model(c)

    assert await g._rotate_model(c) is False


async def test_rotation_restores_the_per_minute_backoff():
    """A fresh model has its own allowance, so the wait is worth paying again."""
    g = _gemini(model_fallbacks=["model-b"])
    g._rate_limited = True

    await g._rotate_model(_StubClient())

    assert g._rate_limited is False


# -- discovery: the list maintains itself ------------------------------------

async def test_models_are_discovered_when_no_list_is_configured():
    g = _gemini(model_fallbacks=[])
    c = _StubClient(_listing("gemini-x-pro", "gemini-x-flash"))

    assert await g._rotate_model(c) is True
    # flash outranks pro: free-tier request limits are far more generous.
    assert g.model == "gemini-x-flash"


async def test_an_explicit_list_wins_over_discovery():
    """The operator may know what their billing covers; the listing does not."""
    g = _gemini(model_fallbacks=["chosen-model"])
    c = _StubClient(_listing("gemini-x-flash"))

    await g._rotate_model(c)

    assert g.model == "chosen-model"
    assert c.calls == 0


async def test_discovery_happens_once_per_run():
    g = _gemini(model_fallbacks=[])
    c = _StubClient(_listing("gemini-x-flash", "gemini-y-flash"))
    await g._rotate_model(c)
    await g._rotate_model(c)

    assert c.calls == 1


async def test_an_empty_listing_is_not_re_fetched():
    """[] must be cached; otherwise every later 429 pays for the same lookup."""
    g = _gemini(model_fallbacks=[])
    c = _StubClient({"models": []})

    assert await g._rotate_model(c) is False
    assert await g._rotate_model(c) is False
    assert c.calls == 1


async def test_models_that_cannot_generate_are_excluded():
    from podcast_scout.providers.llm import _discover_gemini_models

    c = _StubClient({"models": [
        {"name": "models/embed-thing", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/counter", "supportedGenerationMethods": ["countTokens"]},
        {"name": "models/gemini-x-flash", "supportedGenerationMethods": ["generateContent"]},
    ]})

    assert await _discover_gemini_models(c, "key", "https://example/models") == ["gemini-x-flash"]


async def test_discovery_prefers_lite_then_flash_then_pro_and_stable_over_preview():
    from podcast_scout.providers.llm import _discover_gemini_models

    c = _StubClient(_listing(
        "gemini-x-pro", "gemini-x-flash-preview", "gemini-x-flash", "gemini-x-flash-lite",
    ))

    assert await _discover_gemini_models(c, "k", "u") == [
        "gemini-x-flash-lite", "gemini-x-flash", "gemini-x-flash-preview", "gemini-x-pro",
    ]


async def test_a_failed_listing_degrades_quietly():
    from podcast_scout.providers.llm import _discover_gemini_models

    assert await _discover_gemini_models(_StubClient(status=500), "k", "u") == []


# -- thinking tokens are billed against maxOutputTokens ----------------------

def test_headroom_is_added_when_thinking_cannot_be_disabled():
    """thinking_budget None means thinkingConfig is not sent, so the model
    thinks by default and the answer needs room beyond the thoughts."""
    from podcast_scout.providers.llm import _THINKING_HEADROOM_TOKENS

    assert _THINKING_HEADROOM_TOKENS >= 2048


async def test_a_thinking_model_gets_more_than_the_caller_asked_for():
    from podcast_scout.providers.llm import _THINKING_HEADROOM_TOKENS, GeminiProvider

    disabled = GeminiProvider("k", "m", thinking_budget=0)
    unavailable = GeminiProvider("k", "m", thinking_budget=None)

    # Mirrors the calculation in complete(): only the model that will think
    # gets the extra room.
    def effective(g, asked):
        return asked if g.thinking_budget is not None else asked + _THINKING_HEADROOM_TOKENS

    assert effective(disabled, 1500) == 1500
    assert effective(unavailable, 1500) == 1500 + _THINKING_HEADROOM_TOKENS


# -- quota is per KEY and per MODEL, so the search space is keys x models -----

def _multikey(**kw):
    from podcast_scout.providers.llm import GeminiProvider

    return GeminiProvider("", kw.pop("model", "model-a"), **kw)


async def test_a_second_key_is_tried_before_a_lesser_model():
    """A new key restores the configured model; a new model is a downgrade."""
    g = _multikey(api_keys=["k1", "k2"], model_fallbacks=["model-b"])

    assert await g._rotate_model(_StubClient()) is True
    assert g.model == "model-a"      # model preserved
    assert g.api_key == "k2"         # key changed


async def test_the_model_drops_only_once_every_key_is_spent_on_it():
    g = _multikey(api_keys=["k1", "k2"], model_fallbacks=["model-b"])
    c = _StubClient()
    await g._rotate_model(c)         # k1/model-a -> k2/model-a

    assert await g._rotate_model(c) is True
    assert g.model == "model-b"
    # And back to the first key, whose allowance for model-b is untouched.
    assert g.api_key == "k1"


async def test_every_key_and_model_spent_falls_through():
    g = _multikey(api_keys=["k1", "k2"], model_fallbacks=["model-b"])
    c = _StubClient()
    for _ in range(3):
        await g._rotate_model(c)

    assert await g._rotate_model(c) is False


async def test_a_single_key_behaves_as_before():
    g = _multikey(api_keys=["only"], model_fallbacks=["model-b"])

    assert await g._rotate_model(_StubClient()) is True
    assert g.model == "model-b"
    assert g.api_key == "only"
# -- rotate across generations, where a separate quota bucket is likeliest ----

async def test_a_different_generation_is_preferred_over_a_sibling_variant():
    """The exhausted model's own family is where a shared bucket is likeliest."""
    from podcast_scout.providers.llm import _discover_gemini_models

    c = _StubClient(_listing(
        "gemini-3.6-flash-lite",   # same family as the dead model, cheapest tier
        "gemini-2.5-flash",        # different generation
    ))

    ranked = await _discover_gemini_models(c, "k", "u", current="gemini-3.6-flash")

    assert ranked[0] == "gemini-2.5-flash"


async def test_tier_still_decides_within_a_generation():
    from podcast_scout.providers.llm import _discover_gemini_models

    c = _StubClient(_listing("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"))

    ranked = await _discover_gemini_models(c, "k", "u", current="gemini-3.6-flash")

    assert ranked == ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]


async def test_same_family_is_still_offered_when_nothing_else_exists():
    """Last resort beats falling through to a provider with no credit."""
    from podcast_scout.providers.llm import _discover_gemini_models

    c = _StubClient(_listing("gemini-3.6-flash-lite"))

    ranked = await _discover_gemini_models(c, "k", "u", current="gemini-3.6-flash")

    assert ranked == ["gemini-3.6-flash-lite"]


def test_family_is_the_generation_not_the_tier():
    from podcast_scout.providers.llm import _model_family

    assert _model_family("gemini-2.5-flash-001") == "gemini-2.5"
    assert _model_family("gemini-2.5-pro") == "gemini-2.5"
    assert _model_family("gemini-3.6-flash") == "gemini-3.6"


# -- newest generation, not alphabetically first -----------------------------

async def test_the_newest_generation_wins_among_equal_tiers():
    """The live regression: every earlier component ties for a list of lite
    models, so the tiebreak decides -- and alphabetical picks the OLDEST."""
    from podcast_scout.providers.llm import _discover_gemini_models

    c = _StubClient(_listing(
        "gemini-2.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite",
    ))

    ranked = await _discover_gemini_models(c, "k", "u", current="gemini-3.6-flash")

    assert ranked[0] == "gemini-3.5-flash-lite"
    assert ranked[-1] == "gemini-2.5-flash-lite"


async def test_a_moving_alias_ranks_behind_every_explicit_version():
    """It can be repointed mid-run, and may resolve to the exhausted model."""
    from podcast_scout.providers.llm import _discover_gemini_models

    c = _StubClient(_listing("gemini-flash-lite-latest", "gemini-2.5-flash-lite"))

    ranked = await _discover_gemini_models(c, "k", "u", current="gemini-3.6-flash")

    assert ranked == ["gemini-2.5-flash-lite", "gemini-flash-lite-latest"]


def test_generation_is_parsed_from_the_id():
    from podcast_scout.providers.llm import _model_generation

    assert _model_generation("gemini-3.5-flash-lite") == 3.5
    assert _model_generation("gemini-2.5-flash") == 2.5
    assert _model_generation("gemini-flash-lite-latest") == -1.0


async def test_tier_still_outranks_generation():
    """A newer pro model is still worse than an older lite one under a spent
    quota: the lite allowance is far larger."""
    from podcast_scout.providers.llm import _discover_gemini_models

    c = _StubClient(_listing("gemini-3.5-pro", "gemini-2.5-flash-lite"))

    ranked = await _discover_gemini_models(c, "k", "u", current="gemini-3.6-flash")

    assert ranked[0] == "gemini-2.5-flash-lite"
