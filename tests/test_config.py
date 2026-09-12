"""Unit tests for Settings env-var handling."""
import pytest

from podcast_scout.config import Settings, _env

# ---------------------------------------------------------------------------
# _env() strips surrounding whitespace.
#
# GitHub Actions injects secrets verbatim. A PAGES_BASE_URL pasted with a
# leading space produced `<atom:link href=" https://...">` in every generated
# feed; some podcast clients use that self-link when refreshing and reject the
# malformed URL, presenting as "the feed never updates".
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    " https://x.dev/base",
    "https://x.dev/base ",
    "  https://x.dev/base  ",
    "https://x.dev/base\n",
    "\thttps://x.dev/base\t",
])
def test_env_strips_surrounding_whitespace(monkeypatch, raw):
    monkeypatch.setenv("T_URL", raw)
    assert _env("T_URL") == "https://x.dev/base"


@pytest.mark.parametrize("raw", ["", "   ", "\n", "\t "])
def test_env_treats_blank_as_absent(monkeypatch, raw):
    """Whitespace-only must fall back to the default, not yield ''."""
    monkeypatch.setenv("T_BLANK", raw)
    assert _env("T_BLANK", "fallback") == "fallback"


def test_env_unset_uses_default(monkeypatch):
    monkeypatch.delenv("T_MISSING", raising=False)
    assert _env("T_MISSING", "d") == "d"
    assert _env("T_MISSING") == ""


def test_env_preserves_internal_characters(monkeypatch):
    monkeypatch.setenv("T_KEY", "  AIzaSy-Abc_123.xyz  ")
    assert _env("T_KEY") == "AIzaSy-Abc_123.xyz"


def test_pages_base_url_strips_space_and_trailing_slash(monkeypatch):
    """Regression: the leading space observed in the live feed."""
    monkeypatch.setenv("PAGES_BASE_URL", " https://iitiff.github.io/podcast-curation-agent/ ")
    s = Settings()
    assert s.pages_base_url == "https://iitiff.github.io/podcast-curation-agent"
    assert not s.pages_base_url.startswith(" ")
    # The value is interpolated straight into atom:link href — must be clean.
    assert f'href="{s.pages_base_url}/ai-retail.xml"'.startswith('href="https://')


def test_numeric_settings_survive_blank_values(monkeypatch):
    """A declared-but-empty Actions secret must not crash int()/float()."""
    for name in ("LOOKBACK_DAYS", "MAX_LLM_TOKENS_PER_RUN", "MAX_COST_USD_PER_RUN"):
        monkeypatch.setenv(name, "   ")
    s = Settings()
    assert s.lookback_days == 3
    assert s.max_llm_tokens_per_run == 500_000
    assert s.max_cost_usd_per_run == 2.00


def test_numeric_settings_tolerate_padded_values(monkeypatch):
    monkeypatch.setenv("LOOKBACK_DAYS", " 8 ")
    assert Settings().lookback_days == 8


def test_api_keys_are_stripped(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", " key-with-space ")
    monkeypatch.setenv("NVIDIA_API_KEY", "\tnv-key\n")
    s = Settings()
    assert s.gemini_api_key == "key-with-space"
    assert s.fallback_api_key == "nv-key"


def test_fallback_prefers_generic_over_legacy_name(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_API_KEY", "generic")
    monkeypatch.setenv("NVIDIA_API_KEY", "legacy")
    assert Settings().fallback_api_key == "generic"


# ---------------------------------------------------------------------------
# Engine/instance split: output directories and packaged templates
# ---------------------------------------------------------------------------

def test_briefing_dir_defaults_to_public_dir(monkeypatch):
    """Unset BRIEFING_DIR must behave exactly as before the split."""
    from podcast_scout.config import Settings

    monkeypatch.delenv("BRIEFING_DIR", raising=False)
    monkeypatch.setenv("PUBLIC_DIR", "public")
    assert Settings().briefing_dir == Settings().public_dir


def test_briefing_dir_separates_from_public_dir(monkeypatch):
    """The briefing describes how the reader thinks and must stay unpublished."""
    from podcast_scout.config import Settings

    monkeypatch.setenv("PUBLIC_DIR", "feeds")
    monkeypatch.setenv("BRIEFING_DIR", "briefing")
    settings = Settings()
    assert settings.public_dir.name == "feeds"
    assert settings.briefing_dir.name == "briefing"
    assert settings.public_dir != settings.briefing_dir


def test_brain_dir_is_none_unless_set(monkeypatch):
    """An instance that has not opted into the brain must be unaffected."""
    from podcast_scout.config import Settings

    monkeypatch.delenv("BRAIN_DIR", raising=False)
    assert Settings().brain_dir is None
    monkeypatch.setenv("BRAIN_DIR", "brain")
    assert Settings().brain_dir is not None


def test_templates_resolve_to_an_absolute_packaged_path(monkeypatch):
    """Templates must resolve via the package, not relative to the cwd.

    A cwd-relative path silently fails once the engine is pip-installed into a
    separate instance repo: templates_dir.exists() is False and the briefing
    stops rendering with no error.
    """
    from podcast_scout.config import Settings

    monkeypatch.delenv("TEMPLATES_DIR", raising=False)
    templates = Settings().templates_dir
    assert templates.is_absolute()
    assert (templates / "index.html.j2").exists()


def test_provider_named_key_wins_over_the_generic_one(monkeypatch):
    """An OPENROUTER_API_KEY added to replace a broken setup must take effect.

    A live run kept using NVIDIA because a stale LLM_FALLBACK_API_KEY was still
    set, so the newly added provider key was silently ignored.
    """
    from podcast_scout.config import Settings

    monkeypatch.setenv("LLM_FALLBACK_API_KEY", "stale-nvidia")
    monkeypatch.setenv("OPENROUTER_API_KEY", "new-openrouter")
    monkeypatch.delenv("LLM_FALLBACK_BASE_URL", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    s = Settings()
    assert s.fallback_api_key == "new-openrouter"
    assert "openrouter" in s.fallback_base_url
    assert s.fallback_provider_name == "OpenRouter"


def test_explicit_base_url_still_overrides(monkeypatch):
    from podcast_scout.config import Settings

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "https://custom.example/v1")
    assert Settings().fallback_base_url == "https://custom.example/v1"


def test_openrouter_key_matched_by_name_shape(monkeypatch):
    """Enumerating spellings failed three times before the real secret name
    (OPEN_ROUTER_API) was known. Match the shape instead."""
    from podcast_scout.config import Settings

    for name in ("LLM_FALLBACK_API_KEY", "NVIDIA_API_KEY", "LLM_FALLBACK_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for variant in ("OPEN_ROUTER_API", "OPENROUTER_API_KEY", "OpenRouter_Key", "OPEN_ROUTER"):
        monkeypatch.setenv(variant, "k")
        s = Settings()
        assert s.fallback_api_key == "k", f"{variant} must be recognised"
        assert "openrouter" in s.fallback_base_url
        monkeypatch.delenv(variant, raising=False)


def test_unrelated_vars_are_not_mistaken_for_a_key(monkeypatch):
    from podcast_scout.config import Settings

    for name in ("LLM_FALLBACK_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROUTER_CONFIG", "x")
    monkeypatch.setenv("OPEN_FILES", "y")
    assert Settings().fallback_api_key == ""


# ---------------------------------------------------------------------------
# A Google app password is displayed spaced and must be sent unspaced
# ---------------------------------------------------------------------------

def test_a_spaced_app_password_is_accepted_as_displayed():
    """Google shows "abcd efgh ijkl mnop"; SMTP AUTH needs it without spaces."""
    from podcast_scout.cli import _smtp_password

    assert _smtp_password("abcd efgh ijkl mnop") == "abcdefghijklmnop"


def test_an_unspaced_app_password_is_untouched():
    from podcast_scout.cli import _smtp_password

    assert _smtp_password("abcdefghijklmnop") == "abcdefghijklmnop"


def test_a_space_in_any_other_password_is_left_alone():
    """Rewriting a real password would fail auth with no way to tell why."""
    from podcast_scout.cli import _smtp_password

    assert _smtp_password("correct horse battery staple") == "correct horse battery staple"
    assert _smtp_password("two words") == "two words"


def test_surrounding_whitespace_is_still_stripped():
    from podcast_scout.cli import _smtp_password

    assert _smtp_password("  abcd efgh ijkl mnop  ") == "abcdefghijklmnop"
    assert _smtp_password("  plainsecret  ") == "plainsecret"
