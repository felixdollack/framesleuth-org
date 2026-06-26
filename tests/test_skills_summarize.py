"""Tests for the skills registry and the summary synthesis pipeline."""

from typing import Any

import pytest

from framesleuth.pipeline.summarize import build_summary_input, summarize_recording
from framesleuth.schemas import SceneRecord, Transcript
from framesleuth.skills import DEFAULT_SKILL, SKILLS, list_skills, resolve_skill


def test_resolve_skill_defaults_to_summary() -> None:
    """No skill and no prompt resolves to the default skill."""
    label, prompt = resolve_skill(None, None)
    assert label == DEFAULT_SKILL
    assert prompt == SKILLS[DEFAULT_SKILL].system_prompt


def test_resolve_skill_named() -> None:
    """A known skill name resolves to its system prompt."""
    label, prompt = resolve_skill("bug_report", None)
    assert label == "bug_report"
    assert "bug report" in prompt.lower()


def test_resolve_skill_custom_prompt_wins() -> None:
    """A custom system_prompt overrides any skill and is labeled 'custom'."""
    label, prompt = resolve_skill("tutorial", "Only output JSON.")
    assert label == "custom"
    assert prompt == "Only output JSON."


def test_resolve_skill_unknown_falls_back() -> None:
    """An unknown skill name degrades to the default rather than failing."""
    label, prompt = resolve_skill("does-not-exist", None)
    assert label == DEFAULT_SKILL


def test_list_skills_includes_default() -> None:
    """The catalog lists the built-in skills with descriptions."""
    names = {s["name"] for s in list_skills()}
    assert DEFAULT_SKILL in names
    assert {"bug_report", "tutorial"} <= names


def _scene(t: float, caption: str, ocr: str = "") -> SceneRecord:
    return SceneRecord(
        t0=t, t1=t + 1, caption=caption, ocr_text=ocr, ui_action=None, is_error_state=False
    )


def test_build_summary_input_fuses_video_and_audio() -> None:
    """The model input contains the intent, scene captions, and transcript text."""
    scenes = [_scene(0.3, "A login page", ocr="Sign in")]
    transcript = Transcript(
        segments=[Transcript.Segment(t0=0.0, t1=1.0, text="here we log in", conf=0.9)], words=[]
    )
    content = build_summary_input(scenes, transcript, "Summarize the flow")
    assert "User request: Summarize the flow" in content
    assert "A login page" in content  # video evidence
    assert "here we log in" in content  # audio evidence


class _FakeClient:
    """Records the prompt and returns a canned completion."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def complete(
        self, system_prompt: str, user_content: str, *, max_tokens: int = 1024
    ) -> str:
        self.calls.append((system_prompt, user_content))
        return "SUMMARY OK"


class _DeadClient:
    async def complete(self, *_a: Any, **_k: Any) -> str:
        raise RuntimeError("model unavailable")


@pytest.mark.asyncio
async def test_summarize_recording_happy_path() -> None:
    """With evidence and a working model, the summary text is returned."""
    scenes = [_scene(0.3, "A dashboard")]
    transcript = Transcript(segments=[], words=[])
    client = _FakeClient()
    out = await summarize_recording(
        scenes, transcript, system_prompt="SP", user_intent=None, client=client  # type: ignore[arg-type]
    )
    assert out == "SUMMARY OK"
    assert client.calls and client.calls[0][0] == "SP"


@pytest.mark.asyncio
async def test_summarize_recording_no_evidence_returns_empty() -> None:
    """No scenes and no transcript -> empty summary, no model call."""
    client = _FakeClient()
    out = await summarize_recording(
        [], Transcript(segments=[], words=[]), system_prompt="SP", user_intent=None, client=client  # type: ignore[arg-type]
    )
    assert out == ""
    assert client.calls == []


@pytest.mark.asyncio
async def test_summarize_recording_degrades_on_model_failure() -> None:
    """A model failure degrades to an empty summary rather than raising."""
    scenes = [_scene(0.3, "A dashboard")]
    out = await summarize_recording(
        scenes,
        Transcript(segments=[], words=[]),
        system_prompt="SP",
        user_intent=None,
        client=_DeadClient(),  # type: ignore[arg-type]
    )
    assert out == ""
