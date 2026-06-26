"""Summary synthesis: turn the fused video+audio timeline into prose.

Takes the visual scenes (captions + OCR) and the audio transcript, fuses them
into one time-ordered timeline, and asks a text LLM — steered by the caller's
skill/system prompt — to produce a summary. Degrades to an empty string when
there is no evidence or the model is unavailable, so it never aborts a run.
"""

from __future__ import annotations

from framesleuth.clients.coder import CoderClient
from framesleuth.logging_config import get_logger
from framesleuth.pipeline.fusion import build_timeline
from framesleuth.schemas import SceneRecord, Transcript

logger = get_logger("pipeline.summarize")

# Cap timeline events fed to the model so the prompt stays bounded on long clips.
_MAX_EVENTS = 80


def build_summary_input(
    scenes: list[SceneRecord],
    transcript: Transcript,
    user_intent: str | None,
) -> str:
    """Render the fused video+audio timeline into the model's user message."""
    lines: list[str] = []
    if user_intent and user_intent.strip():
        lines.append(f"User request: {user_intent.strip()}")
        lines.append("")
    lines.append("Recording timeline (t = seconds into the video):")
    count = 0
    for event in build_timeline(scenes, transcript):
        text = " ".join(event.text.split()).strip()
        if not text:
            continue
        lines.append(f"- t={event.t:.1f}s [{event.kind}] {text}")
        count += 1
        if count >= _MAX_EVENTS:
            lines.append("- … (timeline truncated)")
            break
    return "\n".join(lines)


async def summarize_recording(
    scenes: list[SceneRecord],
    transcript: Transcript,
    *,
    system_prompt: str,
    user_intent: str | None,
    client: CoderClient,
    max_tokens: int = 1024,
) -> str:
    """Generate a summary of the recording per ``system_prompt``.

    Returns ``""`` when there is nothing to summarize (no scenes and no
    transcript) or when the model call fails — the caller treats an empty
    summary as a degraded summarize stage.
    """
    if not scenes and not transcript.segments:
        return ""
    content = build_summary_input(scenes, transcript, user_intent)
    if not content.strip():
        return ""
    try:
        summary = await client.complete(system_prompt, content, max_tokens=max_tokens)
    except Exception as exc:  # model/network failure must not abort the run
        logger.warning("Summary generation degraded: %s", exc)
        return ""
    logger.info("Summary generated (%d chars)", len(summary))
    return summary
