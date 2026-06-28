"""Assemble structured build/feature context from analyzed scenes.

This is the build counterpart to bug extraction: where ``bug_extract`` produces
repro steps and error evidence, this module distills the *buildable* model of a
video — its screens, UI components, the user flow between screens, design notes,
and data shapes — so a coding agent can implement what was shown instead of
reading a flat caption. It is deterministic (no model calls) and degrades to an
empty/None result when the video carries no structured UI signal.
"""

from __future__ import annotations

from framesleuth.schemas import (
    BuildContext,
    Classification,
    ClassificationLabel,
    CodeCandidate,
    FlowStep,
    SceneRecord,
    Screen,
    UiComponent,
)

# Labels whose videos are about building/showing functionality rather than a failure.
_BUILD_LABELS = {
    ClassificationLabel.FEATURE,
    ClassificationLabel.DEMO,
    ClassificationLabel.TUTORIAL,
}


def _has_ui_signal(scenes: list[SceneRecord]) -> bool:
    """Whether any scene carries structured UI signal worth a build context."""
    return any(scene.ui_elements or scene.screen_name for scene in scenes)


def derive_user_flow(scenes: list[SceneRecord]) -> list[FlowStep]:
    """Build the screen-to-screen flow from time-ordered scenes.

    A transition is emitted whenever the named screen changes between consecutive
    distinct screens; the causing action is the UI action observed around the
    change. Scenes without a screen name are skipped (they cannot anchor a node).
    """
    ordered = sorted(scenes, key=lambda s: s.t0)
    steps: list[FlowStep] = []
    prev_screen: str | None = None
    last_action: str | None = None
    n = 0
    for scene in ordered:
        if scene.ui_action:
            last_action = scene.ui_action
        screen = scene.screen_name
        if not screen or screen == prev_screen:
            continue
        if prev_screen is not None:
            n += 1
            steps.append(
                FlowStep(
                    n=n,
                    from_screen=prev_screen,
                    action=last_action,
                    to_screen=screen,
                    t=scene.t0,
                )
            )
        prev_screen = screen
    return steps


def _aggregate_screens(scenes: list[SceneRecord]) -> list[Screen]:
    """One Screen per distinct screen name, first-seen, with its components."""
    screens: dict[str, Screen] = {}
    for i, scene in enumerate(sorted(scenes, key=lambda s: s.t0)):
        name = scene.screen_name
        if not name:
            continue
        screen = screens.get(name)
        if screen is None:
            screen = Screen(name=name, summary=scene.caption.strip(), t=scene.t0)
            screens[name] = screen
        for el in scene.ui_elements:
            if el.label not in screen.components:
                screen.components.append(el.label)
        cite = f"frame:{i}"
        if cite not in screen.evidence:
            screen.evidence.append(cite)
    return list(screens.values())


def _aggregate_components(scenes: list[SceneRecord]) -> list[UiComponent]:
    """Distinct UI components across frames, keyed by (kind, label)."""
    components: dict[tuple[str, str], UiComponent] = {}
    for i, scene in enumerate(sorted(scenes, key=lambda s: s.t0)):
        cite = f"frame:{i}"
        for el in scene.ui_elements:
            key = (el.kind.lower(), el.label.lower())
            comp = components.get(key)
            if comp is None:
                comp = UiComponent(kind=el.kind, label=el.label, screen=scene.screen_name)
                components[key] = comp
            if el.state and el.state not in comp.states:
                comp.states.append(el.state)
            if cite not in comp.evidence:
                comp.evidence.append(cite)
    return list(components.values())


def _dedupe_preserve(values: list[str | None]) -> list[str]:
    """Order-preserving de-duplication, dropping blanks."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        text = (v or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def build_build_context(
    scenes: list[SceneRecord],
    classification: Classification,
    code_candidates: list[CodeCandidate],
) -> BuildContext | None:
    """Assemble a :class:`BuildContext`, or ``None`` for pure bug reports.

    Produced when the video is a build/feature/demo/tutorial *or* any scene carried
    structured UI signal. ``is_greenfield`` is true when grounding found no existing
    code to extend; ``target_locations`` lists the existing files to extend, or a
    hint to create new ones.
    """
    is_build = classification.label in _BUILD_LABELS
    if not is_build and not _has_ui_signal(scenes):
        return None

    screens = _aggregate_screens(scenes)
    components = _aggregate_components(scenes)
    user_flow = derive_user_flow(scenes)
    design_notes = _dedupe_preserve([s.design_notes for s in scenes])
    data_models = _dedupe_preserve([s.data_shown for s in scenes])

    is_greenfield = not code_candidates
    if code_candidates:
        target_locations = _dedupe_preserve([c.file for c in code_candidates[:5]])
    else:
        target_locations = [
            "No existing code matched — this looks net-new. Create files for the "
            "screens/components above following the repo's conventions."
        ]

    return BuildContext(
        screens=screens,
        components=components,
        user_flow=user_flow,
        design_notes=design_notes,
        data_models=data_models,
        is_greenfield=is_greenfield,
        target_locations=target_locations,
    )
