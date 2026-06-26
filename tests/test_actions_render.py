"""Tests for the action registry, suggested-actions menu, and renderers."""

from __future__ import annotations

from typing import Any

import pytest

from framesleuth.actions import (
    ACTIONS,
    DEFAULT_ACTION,
    auto_action_for,
    list_actions,
    resolve_action,
    resolve_action_task,
    suggest_actions,
)
from framesleuth.render import (
    RENDER_FORMATS,
    render,
    render_github_issue,
    render_markdown,
    render_test_plan,
)


def _report(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "job-1",
        "title": "Save button hangs",
        "classification": {"label": "bug"},
        "analysis_quality": {"level": "full", "warnings": []},
        "severity": "high",
        "suspected_component": "profile",
        "environment": {"browser": "Chrome"},
        "repro_steps": [{"n": 1, "t": 1.0, "action": "Click 'Save'"}],
        "expected_behavior": "saves",
        "actual_behavior": "spinner forever",
        "error_evidence": [{"t": 1.0, "source": "network", "text": "POST /save -> 500"}],
        "code_candidates": [{"file": "profile.py", "line": 9, "match_reason": "search"}],
        "source_video": "rec.webm",
    }
    base.update(extra)
    return base


# ===== Action registry =====


def test_resolve_action_precedence() -> None:
    """custom prompt > explicit action > auto-pick from classification."""
    # custom wins
    label, custom, task = resolve_action("fix", "Do exactly this", "bug")
    assert label == "custom" and custom == "Do exactly this" and task == "Do exactly this"

    # explicit known action
    label, custom, task = resolve_action("test", None, "bug")
    assert label == "test" and custom is None and task == ACTIONS["test"].task

    # auto-pick from classification
    assert resolve_action(None, None, "bug")[0] == "fix"
    assert resolve_action(None, None, "tutorial")[0] == "explain"
    assert resolve_action(None, None, "feedback")[0] == "report"


def test_unknown_action_falls_back_to_auto_pick() -> None:
    """An unknown action name never fails the run — it auto-picks instead."""
    label, _custom, _task = resolve_action("nonsense", None, "tutorial")
    assert label == "explain"  # auto-picked for tutorial, not an error


def test_auto_action_for_defaults() -> None:
    """Unknown/empty labels fall back to the default action."""
    assert auto_action_for(None) == DEFAULT_ACTION
    assert auto_action_for("weird-label") == DEFAULT_ACTION
    assert auto_action_for("DEMO") == "explain"  # case-insensitive


def test_resolve_action_task_shortcut() -> None:
    """resolve_action_task returns just the task string."""
    assert resolve_action_task("explain", None, "bug") == ACTIONS["explain"].task


def test_list_actions_catalog() -> None:
    """The catalog lists every built-in with a description."""
    names = {a["name"] for a in list_actions()}
    assert names == set(ACTIONS)
    assert all(a["description"] for a in list_actions())


# ===== Suggested actions =====


def test_suggest_actions_for_bug() -> None:
    """A bug surfaces fix/test/locate/issue affordances referencing real surfaces."""
    items = suggest_actions(_report())
    actions = {i["action"] for i in items}
    assert {"propose_fix", "write_test", "open_issue"} <= actions
    fix = next(i for i in items if i["action"] == "propose_fix")
    assert "fix-prompt" in fix["ref"]
    assert all(i["label"] and i["rationale"] for i in items)


def test_suggest_actions_degraded_recapture_first() -> None:
    """A degraded report leads with a re-capture suggestion."""
    items = suggest_actions(_report(analysis_quality={"level": "degraded", "warnings": ["thin"]}))
    assert items[0]["action"] == "recapture"


def test_suggest_actions_for_tutorial() -> None:
    """A tutorial suggests explain/docs rather than fix."""
    items = suggest_actions(
        _report(classification={"label": "tutorial"}, error_evidence=[], code_candidates=[])
    )
    actions = {i["action"] for i in items}
    assert "explain" in actions
    assert "propose_fix" not in actions


# ===== Renderers =====


def test_render_markdown_contains_core_sections() -> None:
    md = render_markdown(_report())
    assert md.startswith("# Save button hangs")
    assert "## Steps to reproduce" in md
    assert "## Error evidence" in md
    assert "profile.py:9" in md


def test_render_github_issue_payload() -> None:
    issue = render_github_issue(_report())
    assert issue["title"] == "Save button hangs"
    assert "bug" in issue["labels"]
    assert "severity:high" in issue["labels"]
    assert "evidence:full" in issue["labels"]
    # Body drops the H1 (the issue title carries it) and credits the source.
    assert not issue["body"].startswith("# ")
    assert "Framesleuth" in issue["body"]


def test_render_test_plan_has_arrange_act_assert() -> None:
    plan = render_test_plan(_report())
    assert "Arrange" in plan and "Act" in plan and "Assert" in plan
    assert "POST /save -> 500" in plan


def test_render_dispatch_and_unknown_format() -> None:
    for fmt in RENDER_FORMATS:
        assert render(_report(), fmt).strip()
    with pytest.raises(ValueError, match="unknown render format"):
        render(_report(), "pdf")


def test_renderers_tolerate_sparse_report() -> None:
    """Renderers never raise on a minimal/degraded report."""
    sparse = {"id": "j", "title": "x", "classification": {"label": "other"}}
    assert render_markdown(sparse)
    assert render_test_plan(sparse)
    assert render_github_issue(sparse)["title"] == "x"
