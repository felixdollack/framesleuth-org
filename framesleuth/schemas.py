"""Data contracts and schemas for Framesleuth.

All data structures use Pydantic v2 for validation, serialization, and documentation.
Follows the interface segregation principle with focused, composable schemas.
"""

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ===== Enums =====


class ClassificationLabel(StrEnum):
    """Video classification labels."""

    BUG = "bug"
    TUTORIAL = "tutorial"
    DEMO = "demo"
    FEEDBACK = "feedback"
    OTHER = "other"


class Reproducibility(StrEnum):
    """Reproducibility of the reported issue."""

    SHOWN_ONCE = "shown_once"
    SHOWN_MULTIPLE = "shown_multiple"
    INTERMITTENT = "intermittent"
    CONSISTENT = "consistent"


class Severity(StrEnum):
    """Severity level of the bug."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Priority(StrEnum):
    """Priority level for fixing."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class JobState(StrEnum):
    """Job processing state (mirrors the orchestrator's stage transitions)."""

    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    UNDERSTANDING = "understanding"
    CLASSIFYING = "classifying"
    EXTRACTING = "extracting"
    GROUNDING = "grounding"
    DONE = "done"
    FAILED = "failed"


# ===== Input Contracts =====
#
# Raw browser sidecars arrive as loosely-typed JSON and are normalized by
# ``framesleuth.pipeline.sidecars`` (which tolerates both the flat event stream
# and the structured dict shape), so there is no rigid input model here.


class Transcript(BaseModel):
    """Timestamped transcript from audio."""

    class Segment(BaseModel):
        """Transcript segment."""

        t0: float = Field(..., description="Start time in seconds")
        t1: float = Field(..., description="End time in seconds")
        text: str = Field(..., description="Transcribed text")
        conf: float = Field(..., ge=0, le=1, description="Confidence 0-1")

    segments: list[Segment]
    words: list[dict[str, Any]] | None = Field(None, description="Word-level timing if available")


class SceneRecord(BaseModel):
    """Visual scene record from frame analysis."""

    t0: float = Field(..., description="Scene start time (seconds)")
    t1: float = Field(..., description="Scene end time (seconds)")
    caption: str = Field(..., description="What is visible in the scene")
    ocr_text: str = Field(..., description="All visible text in the scene")
    ui_action: str | None = Field(None, description="Apparent user action (click, type, etc.)")
    is_error_state: bool = Field(False, description="Whether scene shows an error or failure")
    reason: str | None = Field(None, description="Why this frame is marked as error state")


class PreprocessResult(BaseModel):
    """Result of video preprocessing."""

    video_path: Path
    duration_s: float
    fps: float
    width: int
    height: int
    has_audio: bool
    audio_path: Path | None
    frame_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


# ===== Output Contracts =====


class Classification(BaseModel):
    """Classification result with confidence and alternatives."""

    label: ClassificationLabel
    confidence: float = Field(..., ge=0, le=1)
    alt_labels: list[tuple[ClassificationLabel, float]] = Field(
        default_factory=list, description="Alternative labels and confidences"
    )


class ReproStep(BaseModel):
    """Numbered reproduction step with evidence and confidence."""

    n: int = Field(..., ge=1, description="Step number")
    t: float = Field(..., description="Timestamp in seconds")
    action: str = Field(..., description="What the user did")
    evidence: list[str] = Field(..., description="Citations like 'frame:5' or 'transcript:0:08'")
    confidence: float = Field(..., ge=0, le=1)


class ErrorEvidenceItem(BaseModel):
    """Error or failure indicator with source and timing."""

    t: float = Field(..., description="Timestamp in seconds")
    source: Literal["console", "ocr", "network", "ui"] = Field(
        ..., description="Where the error came from"
    )
    text: str = Field(..., description="Error message or observed behavior")


class KeyframeRef(BaseModel):
    """Reference to a keyframe image."""

    index: int = Field(..., description="Keyframe index")
    t: float = Field(..., description="Timestamp in seconds")
    shows: str = Field(..., description="What this keyframe shows")
    file: str = Field(..., description="Path relative to bundle root")


class Redaction(BaseModel):
    """Record of a redaction applied to protect sensitive data."""

    t: float = Field(..., description="Timestamp where redaction occurred")
    region: str = Field(..., description="Description of redacted region")
    applied: bool = Field(..., description="Whether redaction was successfully applied")


class CodeCandidate(BaseModel):
    """Candidate code location matched by grounding."""

    file: str
    line: int
    symbol: str | None = None
    match_reason: str = Field(
        ..., description="How this was matched (stacktrace/search/route/label)"
    )
    confidence: float = Field(..., ge=0, le=1)
    is_third_party: bool = Field(False)


class AnalysisQuality(BaseModel):
    """How much of the pipeline succeeded — the trust signal for consumers.

    Downstream agents (Copilot/Claude/tools) read ``level`` to decide whether to
    act confidently, act cautiously, or ask the user for more evidence instead of
    fabricating a fix from a near-empty bundle.
    """

    level: Literal["full", "partial", "degraded"] = Field(
        ..., description="full=all stages ran; partial=some degraded; degraded=little evidence"
    )
    degraded_stages: list[str] = Field(
        default_factory=list, description="Pipeline stages that degraded (e.g. understand, asr)"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Human-readable notes on what is missing or uncertain"
    )
    evidence_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Counts of extracted evidence (keyframes, errors, repro_steps, transcript)",
    )


class BugContextBundle(BaseModel):
    """The canonical output: complete structured bug context.

    This is the primary artifact delivered to both VS Code and Chrome surfaces.
    Schema versioning enables forward migration and compatibility checks.
    """

    schema_version: str = "1.0"
    id: str = Field(..., description="Unique job ID")
    source_video: str = Field(..., description="Original video filename")
    duration_s: float = Field(...)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    classification: Classification
    reproducibility: Reproducibility
    title: str = Field(..., max_length=200)
    severity: Severity
    priority: Priority
    suspected_component: str = Field(..., max_length=200)

    environment: dict[str, str] = Field(
        ..., description="OS, app, version, browser from OCR or sidecar"
    )
    preconditions: str = Field(..., description="Prerequisites for reproduction")
    repro_steps: list[ReproStep] = Field(..., min_length=1)
    expected_behavior: str = Field(...)
    actual_behavior: str = Field(...)

    error_evidence: list[ErrorEvidenceItem] = Field(default_factory=list)
    keyframe_refs: list[KeyframeRef] = Field(default_factory=list)

    analysis_quality: AnalysisQuality = Field(
        default_factory=lambda: AnalysisQuality(level="full"),
        description="Pipeline completeness/confidence signal for downstream consumers",
    )

    summary: str = Field(
        default="",
        description="Narrative summary of the recording (video + audio), per the chosen skill",
    )
    skill: str | None = Field(
        default=None,
        description="Skill/system-prompt label used for the summary (e.g. 'summary', 'custom')",
    )

    user_intent: str | None = Field(
        default=None, description="The user's natural-language request to act on, if any"
    )

    action: str | None = Field(
        default=None,
        description="Resolved action mode shaping the fix-prompt (e.g. 'fix', 'explain', 'custom')",
    )
    action_prompt: str | None = Field(
        default=None,
        description="Custom action task text to render (set only when action == 'custom')",
    )
    suggested_actions: list[dict[str, str]] = Field(
        default_factory=list,
        description="Machine-readable next-step menu (action/label/rationale/ref) for consumers",
    )

    transcript_path: str | None = Field(
        None, description="Path to transcript.json relative to bundle"
    )
    timeline_path: str | None = Field(None, description="Path to timeline.json relative to bundle")
    redactions: list[Redaction] = Field(default_factory=list, description="Redactions applied")
    code_candidates: list[CodeCandidate] = Field(
        default_factory=list, description="Ranked code locations"
    )

    @field_validator("repro_steps")
    @classmethod
    def validate_repro_steps(cls, v: list[ReproStep]) -> list[ReproStep]:
        """Validate that all repro steps are numbered sequentially."""
        for i, step in enumerate(v, 1):
            if step.n != i:
                raise ValueError(
                    f"Repro steps must be numbered sequentially, got {step.n} at position {i}"
                )
        return v

    def validate_claims_cited(self) -> list[str]:
        """Validate that every claim is cited.

        Returns:
            List of uncited claims (should be empty for valid bundle).
        """
        # For now, basic validation - can be expanded
        uncited = []
        if not self.title:
            uncited.append("title")
        if not self.repro_steps:
            uncited.append("repro_steps")
        return uncited
