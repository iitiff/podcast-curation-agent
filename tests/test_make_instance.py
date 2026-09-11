"""Tests for the instance generator's workflow templates.

These templates are rendered with str.format, so every literal brace in them
must be doubled. Getting that wrong has produced both an outright
IndexError and, more dangerously, silently mangled ${{ secrets.* }}
expressions -- a workflow that parses fine and runs with no credentials.
Nothing else in the suite executes this file, so it is checked here.
"""
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

FMT = {
    "public_url": "https://github.com/example/engine",
    "public_repo": "example/engine",
    "ref": "0123456789abcdef0123456789abcdef01234567",
}


def _module():
    spec = importlib.util.spec_from_file_location(
        "make_instance", REPO_ROOT / "scripts" / "make_instance.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TEMPLATE_NAMES = ["DAILY_WORKFLOW", "LLM_DOCTOR_WORKFLOW", "MONTHLY_REVIEW_WORKFLOW"]


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_workflow_renders_to_valid_yaml_with_intact_expressions(name):
    text = getattr(_module(), name).format(**FMT)

    doc = yaml.safe_load(text)
    assert doc["jobs"], f"{name} declares no jobs"
    # YAML 1.1 parses the bare key `on` as the boolean True.
    assert doc.get("on", doc.get(True)), f"{name} declares no triggers"

    # Every GitHub expression must survive .format() as exactly one brace pair.
    # A single brace parses as valid YAML and runs with the secret unset, so
    # this is checked per occurrence rather than by "is ${{ present anywhere".
    single = re.findall(r"\$\{(?!\{)[^\n]*", text)
    assert not single, f"{name}: under-escaped expression(s): {single}"
    triple = re.findall(r"\$\{\{\{[^\n]*", text)
    assert not triple, f"{name}: over-escaped expression(s): {triple}"

    # Every secrets./vars./inputs. reference must sit inside an expression.
    for context in ("secrets.", "vars.", "inputs."):
        # A trailing \w keeps prose out: a comment ending "...its own inputs."
        # is not a context reference.
        used = len(re.findall(r"(?<![\w.])" + re.escape(context) + r"\w", text))
        wrapped = len(re.findall(r"\$\{\{\s*" + re.escape(context), text))
        assert used == wrapped, (
            f"{name}: {used - wrapped} bare {context} reference(s) outside ${{{{ }}}}"
        )

    # No placeholder left unsubstituted.
    assert "{public_url}" not in text and "{ref}" not in text


def test_daily_workflow_runs_the_weekly_synthesis_on_fridays():
    text = _module().DAILY_WORKFLOW.format(**FMT)

    # Anchored to the argument assignment, not to the bare flag: the word
    # "--synthesis" also appears in a comment and in the dispatch input's
    # description, so a looser check passes on prose after the flag is gone.
    assert 'ARGS="$ARGS --synthesis"' in text, "the weekly synthesis runs nowhere"
    # Derived from the date, not matched against the cron string, so editing
    # the schedule cannot silently drop the synthesis.
    assert 'date -u +%u' in text and '"5"' in text
    # The weekday cron must actually include Friday.
    schedule = yaml.safe_load(text).get("on", yaml.safe_load(text).get(True))["schedule"]
    assert any("1-5" in entry["cron"] or "* * 5" in entry["cron"] for entry in schedule)


def test_monthly_workflow_reviews_the_brain():
    text = _module().MONTHLY_REVIEW_WORKFLOW.format(**FMT)

    assert "podcast-scout brain review" in text
    assert "BRAIN_DIR: brain" in text
    doc = yaml.safe_load(text)
    schedule = doc.get("on", doc.get(True))["schedule"]
    assert schedule[0]["cron"].split()[2] == "1", "monthly review is not on the 1st"


def test_generator_emits_every_workflow_it_announces():
    """A template that exists but is never written is a silent regression --
    which is how the weekly synthesis came to run nowhere."""
    source = (REPO_ROOT / "scripts" / "make_instance.py").read_text()

    for name in TEMPLATE_NAMES:
        assert f"{name}.format(" in source, f"{name} is defined but never written"
