#!/usr/bin/env python3
"""Download/check local model prerequisites with checksum verification stubs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

MODELS = {
    "qwen3_vl": {
        "path": Path.home() / ".cache" / "huggingface" / "qwen3-vl-placeholder.gguf",
        "sha256": "PLACEHOLDER_SHA256"
    },
    "whisper": {
        "path": Path.home() / ".cache" / "whisper" / "model.bin",
        "sha256": "PLACEHOLDER_SHA256"
    }
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path, expected_sha: str) -> bool:
    if not path.exists():
        print(f"[missing] {path}")
        return False
    if expected_sha == "PLACEHOLDER_SHA256":
        print(f"[warn] checksum placeholder for {path}")
        return True
    actual = sha256sum(path)
    if actual != expected_sha:
        print(f"[fail] checksum mismatch for {path}")
        return False
    print(f"[ok] {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Framesleuth model checker")
    parser.add_argument("--strict", action="store_true", help="Fail when any model is missing")
    args = parser.parse_args()

    all_ok = True
    for model in MODELS.values():
        model["path"].parent.mkdir(parents=True, exist_ok=True)
        ok = verify_model(model["path"], model["sha256"])
        all_ok = all_ok and ok

    if args.strict and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
