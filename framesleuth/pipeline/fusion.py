"""Timeline fusion and citation-enforced summary assembly."""

from __future__ import annotations

from dataclasses import dataclass

from framesleuth.schemas import SceneRecord, Transcript


@dataclass(frozen=True)
class TimelineEvent:
    """Unified timeline event from visual or transcript evidence."""

    t: float
    kind: str
    text: str
    citation: str


def build_timeline(scenes: list[SceneRecord], transcript: Transcript) -> list[TimelineEvent]:
    """Merge scene and transcript events in timestamp order."""
    events: list[TimelineEvent] = []

    for index, scene in enumerate(scenes):
        events.append(
            TimelineEvent(
                t=scene.t0,
                kind="scene",
                text=scene.caption,
                citation=f"frame:{index}",
            )
        )
        if scene.ocr_text.strip():
            events.append(
                TimelineEvent(
                    t=scene.t0,
                    kind="ocr",
                    text=scene.ocr_text,
                    citation=f"frame:{index}",
                )
            )

    for index, segment in enumerate(transcript.segments):
        text = segment.text.strip()
        if not text:
            continue
        events.append(
            TimelineEvent(
                t=segment.t0,
                kind="transcript",
                text=text,
                citation=f"transcript:{index}",
            )
        )

    return sorted(events, key=lambda e: e.t)


def enforce_citations(claims: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop claims without citations to avoid uncited summaries."""
    return [claim for claim in claims if claim.get("citation", "").strip()]


def summarize_timeline(events: list[TimelineEvent]) -> list[dict[str, str]]:
    """Create concise summary claims preserving evidence citations."""
    claims: list[dict[str, str]] = []
    for event in events[:20]:
        claims.append({"claim": event.text, "citation": event.citation})
    return enforce_citations(claims)
