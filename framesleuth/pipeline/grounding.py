"""Repository grounding: rank candidate code locations from evidence.

Works for both bugs (ground error symbols to their source) and build/feature work
(ground intent + on-screen UI nouns to the files to extend). Ranking prefers
*definition* lines — function/class/component declarations — over incidental
matches in comments or strings, so the candidate is a place to act, not just a
place a word appears.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from framesleuth.schemas import CodeCandidate

# Source extensions worth grounding against — Python plus the common web/app stacks
# a feature video is likely about. Keeps the scan bounded vs. reading every file.
_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".vue",
    ".svelte",
    ".go",
    ".rb",
    ".java",
    ".kt",
    ".php",
    ".cs",
    ".rs",
    ".swift",
}
# Vendored / generated dirs never worth grounding into.
_SKIP_DIRS = {
    "node_modules",
    ".venv",
    "venv",
    ".git",
    "dist",
    "build",
    ".next",
    "__pycache__",
    ".mypy_cache",
    "site-packages",
    "target",
    "vendor",
}
# Lines that *declare* a symbol — strong grounding anchors across languages.
_DEFINITION = re.compile(
    r"\b(def|class|function|func|const|let|var|export|interface|type|component|struct|fn)\b"
    r"|=>|=\s*\(",
)


def _score_match(line: str, query: str) -> tuple[float, str]:
    """Score a query/line match and report why (definition vs. plain search)."""
    if query.lower() not in line.lower():
        return (0.0, "")
    base = min(1.0, 0.5 + (len(query) / max(20, len(line))))
    stripped = line.lstrip()
    if stripped.startswith(("#", "//", "*", "/*")):
        return (max(0.1, base - 0.25), "comment_search")  # incidental mention
    if _DEFINITION.search(line):
        return (min(1.0, base + 0.2), "definition")  # a place to act
    return (base, "search")


def _iter_code_files(workspace_root: Path) -> Iterable[Path]:
    """Yield source files, skipping vendored/generated directories."""
    for path in workspace_root.rglob("*"):
        if not path.is_file() or path.suffix not in _CODE_EXTENSIONS:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        yield path


def locate_in_code(
    workspace_root: Path, queries: Iterable[str], max_results: int = 10
) -> list[CodeCandidate]:
    """Locate code candidates via deterministic text matching across workspace files."""
    candidates: list[CodeCandidate] = []
    normalized_queries = [query.strip() for query in queries if query.strip()]

    for file_path in _iter_code_files(workspace_root):
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue

        for line_no, line in enumerate(lines, start=1):
            for query in normalized_queries:
                score, reason = _score_match(line, query)
                if score <= 0:
                    continue
                candidates.append(
                    CodeCandidate(
                        file=str(file_path.relative_to(workspace_root)),
                        line=line_no,
                        symbol=None,
                        match_reason=reason,
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
