"""Classification routing for video analysis outputs."""

from __future__ import annotations

import json
from typing import Any, Protocol

from framesleuth.config import Settings, get_settings
from framesleuth.logging_config import get_logger
from framesleuth.schemas import Classification, ClassificationLabel, SceneRecord, Transcript

logger = get_logger("pipeline.classify")

_BUG_MARKERS = ("error", "exception", "failed", "traceback", "typeerror")

# Verbs/phrases that signal a build/feature request ("add a dark mode", "build this
# screen", "let's create a settings page"). Strong verbs weigh more than nouns.
_FEATURE_VERBS = (
    "add",
    "build",
    "create",
    "implement",
    "design",
    "redesign",
    "prototype",
    "scaffold",
    "generate",
    "make a",
    "make the",
)
_FEATURE_PHRASES = (
    "i want",
    "i'd like",
    "i would like",
    "could we",
    "can we add",
    "let's add",
    "lets add",
    "let's build",
    "lets build",
    "let's create",
    "lets create",
    "going to build",
    "going to add",
    "want to add",
    "want to build",
    "need to add",
    "new feature",
    "feature request",
    "would be nice",
    "should have",
    "build this",
    "make this",
    "turn this into",
)
# When the request is really a fix, don't mistake it for a feature.
_FIX_MARKERS = ("fix", "bug", "broken", "not working", "doesn't work", "regression", "crash")


def feature_intent_signal(user_intent: str | None, transcript_text: str = "") -> float:
    """Score 0-1 that this is a build/feature request.

    The user's explicit intent is the strongest signal; narration is secondary.
    A clearly fix-flavored intent suppresses the feature score so a "fix the save
    button" request is never misread as a feature build.
    """
    intent = (user_intent or "").lower()
    transcript = (transcript_text or "").lower()

    score = 0.0
    if any(v in intent for v in _FEATURE_VERBS) or any(p in intent for p in _FEATURE_PHRASES):
        score += 0.6
    # A narrated build *phrase* ("let's build a settings page") is a strong signal
    # on its own; a bare verb in narration is weaker (it may be incidental).
    if any(p in transcript for p in _FEATURE_PHRASES):
        score += 0.5
    elif any(v in transcript for v in _FEATURE_VERBS):
        score += 0.3
    if any(m in intent for m in _FIX_MARKERS):
        score -= 0.5  # it's a fix, not a feature
    return max(0.0, min(1.0, score))


def looks_like_build_intent(user_intent: str | None, transcript_text: str = "") -> bool:
    """Whether to run build-aware visual extraction (structured UI capture).

    Decided up front from intent/narration because it must be known *before* the
    understanding stage (classification only happens afterwards).
    """
    return feature_intent_signal(user_intent, transcript_text) >= 0.5


def classify_video(
    transcript: Transcript,
    scenes: list[SceneRecord],
    *,
    settings: Settings | None = None,
    error_signals: list[str] | None = None,
    user_intent: str | None = None,
) -> Classification:
    """Classify session as bug/feature/tutorial/demo/feedback/other (deterministic).

    Args:
        transcript: Narration transcript (may be empty).
        scenes: Visual scene records (empty when the VLM is unavailable).
        settings: Configuration overrides.
        error_signals: Extra error texts from sidecars (console/network) that
            count as strong bug evidence even without any visual analysis.
        user_intent: The user's natural-language request — the strongest signal for
            a build/feature task ("add a dark-mode toggle", "build this screen").
    """
    settings = settings or get_settings()

    text_blob = " ".join(segment.text.lower() for segment in transcript.segments)
    ocr_blob = " ".join(scene.ocr_text.lower() for scene in scenes)
    signal_blob = " ".join(s.lower() for s in (error_signals or []))

    score = 0.0

    if any(scene.is_error_state for scene in scenes):
        score += 0.45
    if error_signals:
        # Console/network failures captured by the extension are strong evidence.
        score += 0.6
    if any(
        marker in text_blob or marker in ocr_blob or marker in signal_blob
        for marker in _BUG_MARKERS
    ):
        score += 0.45
    if text_blob.count("how to") > 0:
        score -= 0.35

    confidence = max(0.0, min(1.0, score))
    feature_score = feature_intent_signal(user_intent, text_blob)

    # Bugs win when there's real failure evidence (an error is an error). Otherwise,
    # a confident build/feature request is labelled FEATURE so it routes to the
    # implement action and drives BuildContext extraction — instead of "other".
    if confidence >= settings.CLASSIFY_CONFIDENCE_THRESHOLD:
        label = ClassificationLabel.BUG
    elif feature_score >= 0.5:
        label = ClassificationLabel.FEATURE
        confidence = feature_score
    else:
        label = ClassificationLabel.OTHER

    # Only assert an alternative label when there is a real signal for it. A bare
    # ``1 - confidence`` guess produced misleading output like "tutorial: 1.0" on
    # degraded/no-evidence runs, implying a confident alternative that doesn't
    # exist. An empty list honestly says "no competing hypothesis".
    alt_labels: list[tuple[ClassificationLabel, float]] = []
    if label is ClassificationLabel.FEATURE and confidence < 1.0:
        alt_labels = [(ClassificationLabel.DEMO, round(max(0.0, 1.0 - confidence), 2))]
    elif "how to" in text_blob:
        alt_labels = [(ClassificationLabel.TUTORIAL, round(max(0.35, 1.0 - confidence), 2))]
    elif label is ClassificationLabel.BUG:
        alt_labels = [(ClassificationLabel.OTHER, round(max(0.0, 1.0 - confidence), 2))]

    return Classification(label=label, confidence=confidence, alt_labels=alt_labels)


class SupportsComplete(Protocol):
    """Minimal text-completion interface for model-based refinement."""

    async def complete(
        self,
        system_prompt: str,
        user_content: str,
        *,
        max_tokens: int = ...,
        temperature: float = ...,
    ) -> str:
        """Run a system+user chat turn and return the text."""


def is_ambiguous(classification: Classification, settings: Settings) -> bool:
    """Whether the deterministic result sits in the uncertain band.

    Below ``CLASSIFY_AMBIGUOUS_FLOOR`` there is no signal worth a model call;
    at/above ``CLASSIFY_CONFIDENCE_THRESHOLD`` the heuristic is already confident.
    """
    return (
        settings.CLASSIFY_AMBIGUOUS_FLOOR
        <= classification.confidence
        < settings.CLASSIFY_CONFIDENCE_THRESHOLD
    )


def _classification_signals(
    scenes: list[SceneRecord],
    error_signals: list[str] | None,
    user_intent: str | None = None,
) -> dict[str, Any]:
    """Build the diagnostic signal block for the model classifier prompt."""
    errors = error_signals or []
    blob = " ".join(s.lower() for s in errors)
    return {
        "error_count": len(errors),
        "has_stack_trace": "traceback" in blob or "stack" in blob,
        "error_frames": sum(1 for scene in scenes if scene.is_error_state),
        "feature_intent": feature_intent_signal(user_intent) >= 0.5,
    }


def _parse_model_classification(raw: str) -> Classification | None:
    """Parse a model JSON classification, tolerating markdown fences.

    Returns ``None`` when the text cannot be mapped to a valid label so the
    caller can keep the deterministic result instead of guessing.
    """
    content = raw.strip()
    if "```" in content:
        # Pull the body of the first fenced block (``` or ```json).
        fenced = content.split("```", 2)
        if len(fenced) >= 2:
            content = fenced[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
    try:
        data = json.loads(content)
        label = ClassificationLabel(str(data["label"]).lower())
        confidence = max(0.0, min(1.0, float(data["confidence"])))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning("Model classification unparseable, keeping heuristic: %s", exc)
        return None

    alt_labels: list[tuple[ClassificationLabel, float]] = []
    for item in data.get("alt_labels", []) or []:
        try:
            alt_labels.append(
                (ClassificationLabel(str(item[0]).lower()), max(0.0, min(1.0, float(item[1]))))
            )
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return Classification(label=label, confidence=confidence, alt_labels=alt_labels)


async def refine_classification_with_model(
    base: Classification,
    *,
    summary_text: str,
    scenes: list[SceneRecord],
    error_signals: list[str] | None,
    client: SupportsComplete,
    settings: Settings | None = None,
    user_intent: str | None = None,
) -> Classification:
    """Break an ambiguous-band tie with a model classification.

    Falls back to ``base`` whenever the classification is already confident, the
    band gate is off, there is nothing to summarize, or the model is unavailable
    or returns something unparseable — so this never weakens the deterministic
    result, it only sharpens genuinely uncertain ones.
    """
    settings = settings or get_settings()
    if not settings.CLASSIFY_USE_MODEL or not is_ambiguous(base, settings):
        return base
    if not summary_text.strip():
        return base

    from framesleuth.prompts import ClassificationPrompts

    prompt = ClassificationPrompts.classify_video(
        summary_text, _classification_signals(scenes, error_signals, user_intent)
    )
    try:
        raw = await client.complete(
            "You are a precise classifier. Return ONLY the requested JSON.",
            prompt,
            max_tokens=256,
            temperature=0.0,
        )
    except Exception as exc:  # model/network failure must not weaken the result
        logger.warning("Model classification degraded, keeping heuristic: %s", exc)
        return base

    refined = _parse_model_classification(raw)
    if refined is None:
        return base
    logger.info(
        "Classification refined by model: %s (%.2f) <- heuristic %s (%.2f)",
        refined.label.value,
        refined.confidence,
        base.label.value,
        base.confidence,
    )
    return refined
