"""Tests for scene cut and keyframe selection."""

from framesleuth.pipeline.scenes import detect_scene_cuts, select_keyframes


def test_detect_scene_cuts_threshold() -> None:
    """Scene cuts should be selected from deltas crossing threshold."""
    cuts = detect_scene_cuts([0.1, 0.4, 0.2, 0.9], threshold=0.35)
    assert cuts == [1, 3]


def test_select_keyframes_zero_cut_fallback() -> None:
    """Zero-cut inputs must still produce one fallback keyframe."""
    frame_times = [0.0, 1.0, 2.0, 3.0]
    frame_files = [f"frames/{i}.png" for i in range(4)]

    keyframes = select_keyframes(
        frame_times,
        frame_files,
        change_scores=[0.1, 0.1, 0.1, 0.1],
        error_hints=[False, False, False, False],
    )

    assert len(keyframes) == 1
    assert keyframes[0].index == 2


def test_select_keyframes_prioritizes_error_hints() -> None:
    """Error-hinted frames should be included even without scene cuts."""
    frame_times = [0.0, 1.0, 2.0]
    frame_files = [f"frames/{i}.png" for i in range(3)]

    keyframes = select_keyframes(
        frame_times,
        frame_files,
        change_scores=[0.0, 0.0, 0.0],
        error_hints=[False, True, False],
    )

    assert len(keyframes) == 1
    assert keyframes[0].index == 1
    assert keyframes[0].shows == "error_state_candidate"
