"""Skills: named system prompts that shape the recording summary.

A *skill* is a reusable system prompt that tells the summarizer what kind of
output to produce (a narrative summary, a bug report, a tutorial, …). Callers
either pick a built-in skill by name or supply a fully custom ``system_prompt``
to override entirely. This keeps the perception pipeline (frames + audio) fixed
while letting each request choose how the extracted evidence is presented.
"""

from __future__ import annotations

from dataclasses import dataclass

from framesleuth.logging_config import get_logger

logger = get_logger("skills")

# Shared grounding clause prepended to every built-in skill so summaries stay
# faithful to the extracted evidence regardless of the chosen output shape.
_GROUNDING = (
    "You are analyzing a screen recording. You are given a time-ordered timeline "
    "of what appears on screen (scene captions and verbatim OCR text) and the "
    "spoken narration (audio transcript). Base everything ONLY on this evidence "
    "— never invent UI, speech, errors, or outcomes that are not present. If the "
    "evidence is thin, say so plainly."
)


@dataclass(frozen=True)
class Skill:
    """A named output style for the recording summary."""

    name: str
    description: str
    system_prompt: str


SKILLS: dict[str, Skill] = {
    "summary": Skill(
        name="summary",
        description="General narrative summary of the recording (default).",
        system_prompt=(
            f"{_GROUNDING} Produce three sections:\n"
            "1. **Summary** — a short paragraph describing what happens in the "
            "recording, weaving together what is shown on screen and what is said.\n"
            "2. **Steps** — the key actions/events in order, as a numbered list.\n"
            "3. **Issues** — call out any errors, bugs, or unexpected behavior; if "
            "none were observed, state that explicitly."
        ),
    ),
    "bug_report": Skill(
        name="bug_report",
        description="Structured QA bug report (title, repro, expected/actual, severity).",
        system_prompt=(
            f"{_GROUNDING} Write a concise QA bug report with these sections: "
            "Title, Environment, Steps to Reproduce (numbered), Expected Result, "
            "Actual Result, Severity (critical/high/medium/low), and Evidence "
            "(reference the timestamps you relied on). If no defect is evident, "
            "say so instead of inventing one."
        ),
    ),
    "tutorial": Skill(
        name="tutorial",
        description="Step-by-step how-to guide for what the recording demonstrates.",
        system_prompt=(
            f"{_GROUNDING} Write a clear step-by-step tutorial that teaches a "
            "reader to perform what the recording demonstrates. Start with any "
            "prerequisites shown, then give numbered steps with the exact UI "
            "labels/values visible on screen."
        ),
    ),
    "action_items": Skill(
        name="action_items",
        description="Decisions made and concrete action items / follow-ups.",
        system_prompt=(
            f"{_GROUNDING} Extract the decisions made and the concrete action "
            "items / follow-ups, as bullet points. Note an owner for each item "
            "when identifiable. If there are none, say so."
        ),
    ),
    "release_notes": Skill(
        name="release_notes",
        description="Short, user-facing notes describing what was shown or changed.",
        system_prompt=(
            f"{_GROUNDING} Write short, plain-language user-facing notes describing "
            "what was shown or changed in the recording, suitable for a changelog."
        ),
    ),
}

DEFAULT_SKILL = "summary"


def resolve_skill(skill: str | None, system_prompt: str | None) -> tuple[str, str]:
    """Resolve a request's skill into a ``(label, system_prompt)`` pair.

    Precedence: an explicit ``system_prompt`` wins (label ``"custom"``); then a
    known ``skill`` name; otherwise the default skill. An unknown skill name
    falls back to the default rather than failing the run.
    """
    if system_prompt and system_prompt.strip():
        return ("custom", system_prompt.strip())
    if skill:
        key = skill.strip().lower()
        if key in SKILLS:
            return (key, SKILLS[key].system_prompt)
        logger.warning("Unknown skill %r; falling back to %r", skill, DEFAULT_SKILL)
    return (DEFAULT_SKILL, SKILLS[DEFAULT_SKILL].system_prompt)


def list_skills() -> list[dict[str, str]]:
    """Return the catalog of built-in skills as ``{name, description}`` dicts."""
    return [{"name": s.name, "description": s.description} for s in SKILLS.values()]
