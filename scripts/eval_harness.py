#!/usr/bin/env python3
"""Simple evaluation harness for bundle quality metrics on fixtures."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalMetrics:
    """Core eval metrics for generated bundle quality."""

    repro_step_precision: float
    repro_step_recall: float
    error_capture_rate: float
    grounding_hit_rate_at_k: float

    def to_dict(self) -> dict[str, float]:
        return {
            "repro_step_precision": self.repro_step_precision,
            "repro_step_recall": self.repro_step_recall,
            "error_capture_rate": self.error_capture_rate,
            "grounding_hit_rate_at_k": self.grounding_hit_rate_at_k,
        }


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def evaluate_bundle(predicted: dict, expected: dict, k: int = 5) -> EvalMetrics:
    """Compute deterministic quality metrics against golden fixture."""
    pred_steps = {step.get("action", "") for step in predicted.get("repro_steps", [])}
    exp_steps = {step.get("action", "") for step in expected.get("repro_steps", [])}

    overlap_steps = len(pred_steps.intersection(exp_steps))
    precision = _safe_div(overlap_steps, len(pred_steps))
    recall = _safe_div(overlap_steps, len(exp_steps))

    pred_errors = {item.get("text", "") for item in predicted.get("error_evidence", [])}
    exp_errors = {item.get("text", "") for item in expected.get("error_evidence", [])}
    error_overlap = len(pred_errors.intersection(exp_errors))
    error_capture = _safe_div(error_overlap, len(exp_errors))

    pred_candidates = predicted.get("code_candidates", [])[:k]
    exp_candidates = {item.get("file", "") for item in expected.get("code_candidates", [])}
    hit = any(candidate.get("file", "") in exp_candidates for candidate in pred_candidates)
    grounding = 1.0 if hit else 0.0

    return EvalMetrics(
        repro_step_precision=precision,
        repro_step_recall=recall,
        error_capture_rate=error_capture,
        grounding_hit_rate_at_k=grounding,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Framesleuth eval harness")
    parser.add_argument("--predicted", required=True, help="Path to predicted bundle json")
    parser.add_argument("--expected", required=True, help="Path to expected golden bundle json")
    parser.add_argument("--output", default="-", help="Output JSON path or '-' for stdout")
    parser.add_argument("--k", type=int, default=5, help="Top-k for grounding hit rate")
    args = parser.parse_args()

    predicted = json.loads(Path(args.predicted).read_text(encoding="utf-8"))
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))

    metrics = evaluate_bundle(predicted, expected, k=args.k).to_dict()
    payload = json.dumps(metrics, indent=2)

    if args.output == "-":
        print(payload)
    else:
        Path(args.output).write_text(payload, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
