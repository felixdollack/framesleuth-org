"""Tests for schema validation and serialization."""

import pytest

from framesleuth.schemas import (
    BugContextBundle,
    Classification,
    ClassificationLabel,
    KeyframeRef,
    Priority,
    Reproducibility,
    ReproStep,
    Severity,
)


class TestBugContextBundle:
    """Test canonical bundle schema."""

    def test_bundle_round_trip(self) -> None:
        """Test bundle can serialize and deserialize."""
        bundle = BugContextBundle(
            id="bug_2026-06-20_001",
            source_video="test.webm",
            duration_s=60.0,
            classification=Classification(
                label=ClassificationLabel.BUG,
                confidence=0.92,
            ),
            reproducibility=Reproducibility.SHOWN_ONCE,
            title="Test bug",
            severity=Severity.HIGH,
            priority=Priority.P1,
            suspected_component="test/component",
            environment={"os": "Linux", "browser": "Chrome"},
            preconditions="Logged in",
            repro_steps=[
                ReproStep(
                    n=1,
                    t=5.0,
                    action="Click button",
                    evidence=["frame:1"],
                    confidence=0.9,
                )
            ],
            expected_behavior="Should work",
            actual_behavior="Fails",
        )

        # Serialize
        data = bundle.model_dump_json()
        assert isinstance(data, str)

        # Deserialize
        restored = BugContextBundle.model_validate_json(data)
        assert restored.id == bundle.id
        assert restored.title == bundle.title
        assert restored.repro_steps[0].n == 1

    def test_bundle_validation_repro_steps_sequential(self) -> None:
        """Test repro steps must be numbered sequentially."""
        with pytest.raises(ValueError, match="sequentially"):
            BugContextBundle(
                id="bug_001",
                source_video="test.webm",
                duration_s=60.0,
                classification=Classification(
                    label=ClassificationLabel.BUG,
                    confidence=0.9,
                ),
                reproducibility=Reproducibility.SHOWN_ONCE,
                title="Test",
                severity=Severity.HIGH,
                priority=Priority.P1,
                suspected_component="test",
                environment={},
                preconditions="",
                repro_steps=[
                    ReproStep(n=1, t=5.0, action="Step 1", evidence=[], confidence=0.9),
                    ReproStep(
                        n=3, t=10.0, action="Step 3", evidence=[], confidence=0.9
                    ),  # Wrong: should be 2
                ],
                expected_behavior="",
                actual_behavior="",
            )

    def test_bundle_validate_claims_cited(self) -> None:
        """Test citation validation helper."""
        bundle = BugContextBundle(
            id="bug_001",
            source_video="test.webm",
            duration_s=60.0,
            classification=Classification(
                label=ClassificationLabel.BUG,
                confidence=0.9,
            ),
            reproducibility=Reproducibility.SHOWN_ONCE,
            title="Test bug",
            severity=Severity.HIGH,
            priority=Priority.P1,
            suspected_component="test",
            environment={},
            preconditions="",
            repro_steps=[ReproStep(n=1, t=5.0, action="Test", evidence=[], confidence=0.9)],
            expected_behavior="works",
            actual_behavior="fails",
        )
        uncited = bundle.validate_claims_cited()
        assert len(uncited) == 0

    def test_keyframe_ref(self) -> None:
        """Test keyframe reference validation."""
        ref = KeyframeRef(
            index=1,
            t=5.0,
            shows="Initial state",
            file="keyframes/001.png",
        )
        assert ref.index == 1
        assert ref.t == 5.0
        assert "keyframes" in ref.file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
