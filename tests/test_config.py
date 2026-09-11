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
