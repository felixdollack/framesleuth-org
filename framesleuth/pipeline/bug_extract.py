"""Bug context bundle extraction with evidence and anti-fabrication guards."""

from __future__ import annotations

from datetime import UTC, datetime

from framesleuth.schemas import (
    AnalysisQuality,
    BugContextBundle,
    Classification,
    ErrorEvidenceItem,
    KeyframeRef,
    Priority,
    Reproducibility,
    ReproStep,
    SceneRecord,
    Severity,
    Transcript,
)

_MAX_TITLE_LEN = 200

# Human-readable explanation for each pipeline stage that can degrade. Keeps the
# warning text consistent between the bundle and the downstream fix prompt.
_STAGE_WARNINGS = {
    "preprocess": "Video duration could not be determined; frame sampling was limited.",
    "understand": "Visual frame analysis was unavailable; no on-screen evidence was read.",
    "asr": "No usable audio transcript (silent video or speech model unavailable).",
}


def _derive_repro_steps(scenes: list[SceneRecord]) -> list[ReproStep]:
    """Create reproducible steps from scene actions while preserving citations."""
    steps: list[ReproStep] = []
    for idx, scene in enumerate(scenes, start=1):
        action = (scene.ui_action or scene.caption).strip()
        if not action:
            continue
        steps.append(
            ReproStep(
                n=idx,
                t=scene.t0,
                action=action,
                evidence=[f"frame:{idx - 1}"],
                confidence=0.8 if scene.ui_action else 0.65,
            )
        )
    return steps


def _derive_error_evidence(scenes: list[SceneRecord]) -> list[ErrorEvidenceItem]:
    """Extract error evidence only from observed OCR or flagged error states."""
    evidence: list[ErrorEvidenceItem] = []
    for scene in scenes:
        if scene.is_error_state and scene.ocr_text.strip():
            evidence.append(
                ErrorEvidenceItem(t=scene.t0, source="ocr", text=scene.ocr_text.strip())
            )
    return evidence


def _renumber(steps: list[ReproStep]) -> list[ReproStep]:
    """Sort steps by timestamp and assign sequential step numbers."""
    ordered = sorted(steps, key=lambda s: s.t)
    return [
        ReproStep(n=i, t=s.t, action=s.action, evidence=s.evidence, confidence=s.confidence)
        for i, s in enumerate(ordered, start=1)
    ]


_ERROR_MARKERS = ("error", "exception", "typeerror", "failed", "undefined", "null", " at ")


def _evidence_rank(item: ErrorEvidenceItem) -> tuple[int, float]:
    """Rank evidence so the most diagnostic item wins (higher score, earlier time).

    Redacted/secret lines carry no diagnostic value and are demoted; genuine
    errors (HTTP failures, exceptions, stack frames) are promoted.
    """
    text = item.text
    meaningful = text.replace("[REDACTED]", "").strip()
    score = 0
    if len(meaningful) < 6:
        score -= 5  # essentially a redacted/empty line
    if item.source == "network":
        score += 3
    lowered = text.lower()
    if any(marker in lowered for marker in _ERROR_MARKERS):
        score += 2
    return (score, -item.t)


def _primary_evidence(evidence: list[ErrorEvidenceItem]) -> ErrorEvidenceItem | None:
    """Select the single most diagnostic error for the title and summary."""
    if not evidence:
        return None
    return max(evidence, key=_evidence_rank)


def _synthesize_title(primary: ErrorEvidenceItem | None, scenes: list[SceneRecord]) -> str:
    """Derive a concise headline.

    Prefers the strongest error. With no error, describe what was actually
    observed (the first meaningful scene caption) so a non-bug recording gets an
    informative title instead of the generic placeholder. Falls back to the
    placeholder only when there is no visual evidence at all.
    """
    if primary is not None:
        first = primary.text.splitlines()[0].strip()
        return (first or "Error observed during recorded flow")[: _MAX_TITLE_LEN - 1]
    for scene in scenes:
        caption = (scene.caption or "").strip()
        if caption:
            return caption[: _MAX_TITLE_LEN - 1]
    return "Observed UI behavior during recorded flow"


def _assess_quality(
    *,
    degraded_stages: list[str],
    scenes: list[SceneRecord],
    evidence: list[ErrorEvidenceItem],
    cited_steps: list[ReproStep],
    transcript: Transcript,
    keyframes: list[KeyframeRef],
    has_real_steps: bool,
) -> AnalysisQuality:
    """Summarize how trustworthy the bundle is for a downstream agent.

    ``degraded`` means there is essentially nothing to act on (no visual scenes,
    no error evidence, and only the generic fallback repro step); ``partial``
    means some stages degraded but real evidence survived; ``full`` means the
    pipeline ran cleanly. The warnings explain *what* is missing so the consumer
    can gather more rather than guess.
    """
    warnings = [_STAGE_WARNINGS[stage] for stage in degraded_stages if stage in _STAGE_WARNINGS]

    has_evidence = bool(scenes or evidence)
    if not has_evidence and not has_real_steps:
        level: str = "degraded"
        warnings.append(
            "Insufficient evidence was extracted from the recording — treat findings "
            "as low confidence and gather more (re-record, attach console/network logs)."
        )
    elif degraded_stages:
        level = "partial"
    else:
        level = "full"

    return AnalysisQuality(
        level=level,  # type: ignore[arg-type]
        degraded_stages=list(degraded_stages),
        warnings=warnings,
        evidence_counts={
            "keyframes": len(keyframes),
            "error_evidence": len(evidence),
            "repro_steps": len(cited_steps),
            "scenes": len(scenes),
            "transcript_segments": len(transcript.segments),
        },
    )


def extract_bug_context_bundle(
    *,
    job_id: str,
    source_video: str,
    duration_s: float,
    classification: Classification,
    transcript: Transcript,
    scenes: list[SceneRecord],
    keyframes: list[KeyframeRef],
    environment: dict[str, str],
    sidecar_steps: list[ReproStep] | None = None,
    sidecar_evidence: list[ErrorEvidenceItem] | None = None,
    degraded_stages: list[str] | None = None,
) -> BugContextBundle:
    """Build canonical bundle, merging visual and sidecar evidence without fabrication."""
    scene_steps = _derive_repro_steps(scenes)
    all_steps = scene_steps + list(sidecar_steps or [])
    real_steps = _renumber([step for step in all_steps if step.evidence])
    has_real_steps = bool(real_steps)
    cited_steps = real_steps or [
        ReproStep(
            n=1,
            t=0.0,
            action="Open the page and reproduce the observed behavior",
            evidence=["sidecar:env"],
            confidence=0.5,
        )
    ]

    evidence = _derive_error_evidence(scenes) + list(sidecar_evidence or [])
    evidence.sort(key=lambda item: item.t)

    quality = _assess_quality(
        degraded_stages=list(degraded_stages or []),
        scenes=scenes,
        evidence=evidence,
        cited_steps=cited_steps,
        transcript=transcript,
        keyframes=keyframes,
        has_real_steps=has_real_steps,
    )

    primary = _primary_evidence(evidence)
    title = _synthesize_title(primary, scenes)
    severity = Severity.HIGH if evidence else Severity.MEDIUM
    priority = Priority.P1 if evidence else Priority.P2
    if primary is not None:
        actual_behavior = primary.text.strip()
    elif quality.level == "degraded":
        # Be honest: we could not extract enough to describe behavior. Do not imply
        # the flow succeeded — that would mislead a downstream agent into "no-op".
        actual_behavior = (
            "Analysis incomplete — not enough evidence was extracted to describe "
            "the observed behavior (see analysis_quality.warnings)."
        )
    else:
        actual_behavior = "Recorded flow completed; no explicit error surfaced."

    return BugContextBundle(
        schema_version="1.0",
        id=job_id,
        source_video=source_video,
        duration_s=duration_s,
        created_at=datetime.now(UTC),
        classification=classification,
        reproducibility=Reproducibility.SHOWN_ONCE,
        title=title,
        severity=severity,
        priority=priority,
        suspected_component=environment.get("component", "unknown"),
        environment=environment,
        preconditions="User is authenticated and page is loaded.",
        repro_steps=cited_steps,
        expected_behavior="Action completes successfully without errors.",
        actual_behavior=actual_behavior,
        error_evidence=evidence,
        keyframe_refs=keyframes,
        analysis_quality=quality,
        transcript_path="transcript.json" if transcript.segments else None,
        timeline_path="timeline.json",
        redactions=[],
        code_candidates=[],
    )
