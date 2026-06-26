"""Repository grounding: rank candidate code locations from evidence."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from framesleuth.schemas import CodeCandidate


def _score_match(line: str, query: str) -> float:
    if query.lower() in line.lower():
        return min(1.0, 0.5 + (len(query) / max(20, len(line))))
    return 0.0


def locate_in_code(
    workspace_root: Path, queries: Iterable[str], max_results: int = 10
) -> list[CodeCandidate]:
    """Locate code candidates via deterministic text matching across workspace files."""
    candidates: list[CodeCandidate] = []

    files = [p for p in workspace_root.rglob("*.py") if p.is_file()]
    normalized_queries = [query.strip() for query in queries if query.strip()]

    for file_path in files:
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue

        for line_no, line in enumerate(lines, start=1):
            for query in normalized_queries:
                score = _score_match(line, query)
                if score <= 0:
                    continue
                candidates.append(
                    CodeCandidate(
                        file=str(file_path.relative_to(workspace_root)),
                        line=line_no,
                        symbol=None,
                        match_reason="verbatim_search",
                        confidence=score,
                        is_third_party="site-packages" in str(file_path),
                    )
                )

    # Deterministic ordering: confidence desc, file asc, line asc.
    ordered = sorted(candidates, key=lambda c: (-c.confidence, c.file, c.line))
    unique: list[CodeCandidate] = []
    seen: set[tuple[str, int]] = set()
    for candidate in ordered:
        key = (candidate.file, candidate.line)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= max_results:
            break

    return unique
