"""Per-field confidence and task-aware actionability for a Context Bundle.

``analysis_quality.level`` says how much of the *pipeline* ran. These signals say
something finer and more useful to a downstream agent: how much to trust each
individual claim (``field_confidence``), and whether the evidence is sufficient
for the *resolved action* (``actionability``) — a bundle can have "full" pipeline
quality yet be "insufficient" to implement a feature if no UI was captured.
"""

from __future__ import annotations

from typing import Literal

from framesleuth.schemas import ContextBundle

Actionability = Literal["ready", "thin", "insufficient"]


def compute_field_confidence(bundle: ContextBundle) -> dict[str, float]:
    """Confidence 0-1 for the key bundle fields, from the evidence behind each."""
    has_errors = bool(bundle.error_evidence)
    conf: dict[str, float] = {}

    # Title: strong when anchored to extracted error text, weaker when paraphrasing
    # a scene caption.
    conf["title"] = 0.85 if has_errors else 0.55

    # Repro/observed steps: the mean of the per-step confidences the extractor set.
    if bundle.repro_steps:
        conf["repro_steps"] = round(
            sum(s.confidence for s in bundle.repro_steps) / len(bundle.repro_steps), 2
        )
    else:
        conf["repro_steps"] = 0.0

    # Severity/priority are bug-shaped guesses — high only when a failure is in hand.
    conf["severity"] = 0.8 if has_errors else 0.35

    conf["suspected_component"] = 0.7 if bundle.suspected_component not in ("", "unknown") else 0.3

    if bundle.code_candidates:
        conf["code_candidates"] = round(
            sum(c.confidence for c in bundle.code_candidates) / len(bundle.code_candidates), 2
        )

    bc = bundle.build_context
    if bc is not None:
        signal = len(bc.screens) + len(bc.components) + len(bc.user_flow)
        # 0 signal -> 0.3, saturating toward 0.9 as structured evidence accumulates.
        conf["build_context"] = round(min(0.9, 0.3 + 0.1 * signal), 2)

    return conf


def assess_actionability(bundle: ContextBundle) -> Actionability:
    """Whether the evidence suffices for the bundle's resolved action.

    Independent of which stages ran: it asks "given what we extracted, can the
    downstream agent actually do the requested action?" A degraded pipeline can
    never be more than ``thin``.
    """
    action = (bundle.action or "").lower()
    degraded = bundle.analysis_quality.level == "degraded"
    has_errors = bool(bundle.error_evidence)
    has_candidates = bool(bundle.code_candidates)
    bc = bundle.build_context
    has_build = bc is not None and bool(bc.screens or bc.components)

    if action in {"implement", "design"}:
        level: Actionability = (
            "ready" if has_build else ("thin" if (bc and bc.components) else "insufficient")
        )
    elif action == "fix":
        if has_errors and has_candidates:
            level = "ready"
        elif has_errors or has_candidates:
            level = "thin"
        else:
            level = "insufficient"
    elif action == "test":
        level = "ready" if (has_errors and bundle.repro_steps) else "thin"
    else:  # explain / report / triage / reproduce / custom
        any_evidence = has_errors or has_build or len(bundle.repro_steps) > 0
        level = "ready" if any_evidence else "insufficient"

    if degraded and level == "ready":
        level = "thin"
    return level
