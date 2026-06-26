#!/usr/bin/env bash
#
# Start the Framesleuth local stack via Docker Compose.
# RUN it, do not SOURCE it:  ./scripts/dev_up.sh
#
# (Sourcing would run `set -e` in your interactive shell, so any failed command
#  — e.g. Docker not installed — would close your terminal.)

# Refuse to be sourced: return without touching the parent shell's options.
if (return 0 2>/dev/null); then
  echo "Don't source this script — run it:  ./scripts/dev_up.sh" >&2
  return 1 2>/dev/null || exit 1
fi

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Virtualenv not found. Run: uv venv && source .venv/bin/activate && uv pip install -e \".[dev]\""
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found — this script starts the stack with Docker Compose."
  echo
  echo "No Docker? Use the Ollama path instead (README → 'Start & stop the stack'):"
  echo "  ollama serve &            # if not already running"
  echo "  ollama pull qwen2.5vl"
  echo "  framesleuth-api           # backend on 127.0.0.1:8010"
  exit 1
fi

echo "Starting Framesleuth local stack"

docker compose --profile local-default up -d

echo "Services started."
echo "Backend: http://127.0.0.1:8010/v1/healthz"
echo "VLM: http://127.0.0.1:8080/v1/models"
echo "Coder: http://127.0.0.1:11434/api/tags"
echo
echo "Stop the stack with: docker compose --profile local-default down"
