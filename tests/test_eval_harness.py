"""Tests for eval harness metrics."""

from scripts.eval_harness import evaluate_bundle


def test_eval_metrics_computation() -> None:
    predicted = {
        "repro_steps": [{"action": "Click Save"}, {"action": "Wait"}],
        "error_evidence": [{"text": "TypeError"}],
        "code_candidates": [{"file": "a.py"}],
    }
    expected = {
        "repro_steps": [{"action": "Click Save"}],
        "error_evidence": [{"text": "TypeError"}],
        "code_candidates": [{"file": "a.py"}],
    }

    metrics = evaluate_bundle(predicted, expected, k=3)

    assert metrics.repro_step_precision == 0.5
    assert metrics.repro_step_recall == 1.0
    assert metrics.error_capture_rate == 1.0
    assert metrics.grounding_hit_rate_at_k == 1.0
