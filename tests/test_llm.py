"""Tests for LLM provider response handling."""
import pytest

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

def test_thinking_is_disabled_in_payload():
    """The actual fix: thinking must be off or it eats the output budget."""
    cfg = GeminiProvider("k", "gemini-2.5-flash")
    assert cfg.model == "gemini-2.5-flash"
    # Payload shape is asserted via the source contract below.
    import inspect
    src = inspect.getsource(GeminiProvider.complete)
    assert '"thinkingConfig": {"thinkingBudget": 0}' in src


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
