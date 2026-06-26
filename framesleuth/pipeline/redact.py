"""Redaction pipeline for OCR text and sidecar-like sensitive fields."""

from __future__ import annotations

import re

from framesleuth.schemas import Redaction

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "api_key",
        re.compile(r"\b(sk|rk|pk)_[A-Za-z0-9]{8,}\b"),
    ),
    (
        "bearer_token",
        re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}\b", re.IGNORECASE),
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    (
        "password_assignment",
        re.compile(r"\b(password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
    ),
    (
        "token_assignment",
        re.compile(
            r"\b(token|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*\S+",
            re.IGNORECASE,
        ),
    ),
]


def redact_text(text: str, timestamp: float = 0.0) -> tuple[str, list[Redaction]]:
    """Redact likely secrets from OCR/text streams using regex detectors."""
    redactions: list[Redaction] = []
    output = text

    for region, pattern in _SECRET_PATTERNS:
        if pattern.search(output):
            output = pattern.sub("[REDACTED]", output)
            redactions.append(Redaction(t=timestamp, region=region, applied=True))

    return output, redactions
