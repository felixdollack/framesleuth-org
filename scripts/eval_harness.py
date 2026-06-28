#!/usr/bin/env python3
"""Evaluation harness CLI for Framesleuth.

Two modes:
- ``--predicted/--expected``: bundle-vs-golden metrics (precision/recall/grounding).
- ``--behavioral``: model-free suites for classification, grounding, and citation
  integrity (the same checks gated in ``tests/test_eval_harness.py``).

The metric logic lives in :mod:`framesleuth.eval.harness`; this file is the CLI.
``evaluate_bundle`` / ``EvalMetrics`` are re-exported for backward compatibility.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from framesleuth.eval.harness import EvalMetrics, evaluate_bundle, run_all

__all__ = ["EvalMetrics", "evaluate_bundle", "main"]


def _run_behavioral() -> int:
    """Run the model-free behavioral suites and print a metrics summary."""
    with tempfile.TemporaryDirectory() as tmp:
        results = run_all(Path(tmp))
    for result in results.values():
        print(result)
    worst = min(r.metric for r in results.values())
    return 0 if worst >= 0.8 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Framesleuth eval harness")
    parser.add_argument(
        "--behavioral",
        action="store_true",
        help="Run model-free classification/grounding/citation suites",
    )
    parser.add_argument("--predicted", help="Path to predicted bundle json")
    parser.add_argument("--expected", help="Path to expected golden bundle json")
    parser.add_argument("--output", default="-", help="Output JSON path or '-' for stdout")
    parser.add_argument("--k", type=int, default=5, help="Top-k for grounding hit rate")
    args = parser.parse_args()

    if args.behavioral:
        return _run_behavioral()

    if not args.predicted or not args.expected:
        parser.error("--predicted and --expected are required (or use --behavioral)")

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
