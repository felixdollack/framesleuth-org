<p align="center">
  <img src="docs/logo/framesleuth-logo-256.png" alt="Framesleuth" width="128" height="128" />
</p>

# Framesleuth

**Local bug-reproduction video analysis, exposed over MCP.**

Framesleuth takes a bug-recording video (plus optional browser sidecars), understands it
frame-by-frame, and produces a structured **Bug Context Bundle**. It is **MCP-ready**, so
any MCP client — a VS Code agent, another coding agent, or a custom system — can drive the
analysis and consume the result to fix the bug directly.

Capture happens in a separate Chrome extension,
[inkwell](https://github.com/santoshshinde2012/inkwell), which records the bug and posts the
video + sidecars to this agent's local API. This repo is the analysis agent only.

Everything runs locally. Nothing leaves your machine.

## Quick start

> **Want to fix a bug from a video inside VS Code?** Follow
> [Use with VS Code & Claude (MCP)](docs/use-with-vscode-and-claude.md) — connect
> the bundled MCP server and go from a recording to a grounded fix.

### Try it end-to-end in 2 minutes (no models required)

Framesleuth **degrades gracefully**: with no vision model or ffmpeg installed, it
still produces a Bug Context Bundle from the browser sidecars (console errors,
failed network requests, clicks). This is enough to record a bug in Chrome and
get a structured report.

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 1. Start the backend (binds 127.0.0.1:8010 from config; 8000 is left for inkwell)
framesleuth-api          # or: uvicorn framesleuth.service.api:app --port 8010

# 2. Install the inkwell Chrome extension (separate repo) to capture a bug:
#    https://github.com/santoshshinde2012/inkwell
#    Record → reproduce the bug → Stop. It posts the video + sidecars here.

# 3. Or drive the agent directly with your own video file — over the HTTP API
#    (see the Postman collection / runbook) or the videobug MCP server
#    (see docs/use-with-vscode-and-claude.md).
```

Check `GET /v1/healthz`: `status` is *healthy* when the vision + coder models
are up, or *unhealthy* when running sidecar-only (with `vlm`/`coder` reported
`unavailable` and `storage` `ready`). Add the model servers (below) to enable
frame-level OCR/visual understanding.

### Prerequisites (full pipeline)
- Python 3.11+
- A local VLM server (llama.cpp `:8080` or Ollama `:11434`) for frame understanding
- 8GB+ RAM (for models)

> ffmpeg is **not** a separate prerequisite — frame/audio decoding uses PyAV,
> which bundles its own ffmpeg libraries. (`ffprobe`, if present, is used
> opportunistically to detect whether a recording has an audio stream.)

### Setup

```bash
# Clone and navigate
git clone https://github.com/santoshshinde2012/framesleuth.git
cd framesleuth

# Create environment and install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Download models (one-time, ~10-20GB)
python scripts/download_models.py

# Copy environment template
cp .env.example .env
# Edit .env to match your setup (see runbook.md for details)

# Start services (Docker Compose). Run it — do NOT `source` it.
./scripts/dev_up.sh
# No Docker? Skip this and use the Ollama path in "Start & stop the stack" below.
```

### Start & stop the stack (Ollama — verified working)

`scripts/dev_up.sh` uses Docker Compose. If you don't run Docker, use the
engine-agnostic Ollama path below. `.env.example` ships the llama.cpp defaults
(`VLM_URL=http://127.0.0.1:8080`, `VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct-GGUF`); for
the Ollama path, copy it to `.env` and set `VLM_URL=http://127.0.0.1:11434` and
`VLM_MODEL=qwen2.5vl`. The vision model is the
piece that powers frame-level understanding; without it reachable, analyses come
back **degraded** (no on-screen evidence read).

**Start**

```bash
# 1. Vision model — start Ollama and pull the VLM once (~6 GB; or qwen2.5vl:3b)
ollama serve &                       # skip if Ollama is already running
ollama pull qwen2.5vl

# 2. Backend — run from the repo root so it loads .env (binds 127.0.0.1:8010)
source .venv/bin/activate
framesleuth-api                      # or: uvicorn framesleuth.service.api:app --port 8010

# 3. Verify BEFORE recording — both should report ready
curl -s http://127.0.0.1:11434/v1/models | grep -q qwen2.5vl && echo "VLM ready"
curl -s http://127.0.0.1:8010/v1/healthz   # expect status: healthy, vlm: ready
```

When `/v1/healthz` shows `vlm: ready`, recordings analyze with a real
classification (`analysis_quality.level` = `full`/`partial`). If the VLM is down,
you get the degraded "evidence was thin" report instead. Record **with narration**
so the audio transcript (`asr`) stage contributes too.

**Stop**

```bash
# Stop the backend: Ctrl+C in its terminal, or
pkill -f framesleuth-api

# Stop Ollama (optional — leaving it running keeps the model warm)
pkill -f "ollama serve"              # macOS app users: quit Ollama from the menu bar
```

## Architecture

```
Bug video (mp4/webm) + sidecars
    ↓
Local Analysis Service (pipeline)
    ├─ Preprocess (PyAV: duration/fps/dims)
    ├─ Transcript (faster-whisper)
    ├─ Keyframes (visual-delta change scoring)
    ├─ Understanding (Qwen3-VL)
    ├─ Fusion + Classification
    ├─ Extraction → Bug Context Bundle
    ├─ Summarize (skill/system-prompt-driven)
    └─ Grounding (workspace search)
    ↓
Bug Context Bundle
    ↓
MCP server + local HTTP API
    └─ consumed by any MCP client (VS Code agent, other agents, inkwell extension)
```

## Features

- **Frame-by-frame understanding** using Qwen3-VL vision model
- **Automatic keyframe selection** via frame-to-frame visual-delta change scoring
- **Error detection and extraction** from console, OCR, and UI state
- **Redaction-first design** — sensitive data (passwords, tokens) redacted before models see it
- **No data leaves your machine** — fully local, no telemetry or cloud APIs
- **Engine-agnostic** — swap Ollama, llama.cpp, or vLLM via config only
- **Structured output** — canonical Bug Context Bundle with evidence citations
- **Configurable response** — pick a summary **skill** *and* an **action mode**
  (`fix`/`explain`/`triage`/`test`/`report`/`reproduce`, auto-picked from the
  classification), plus a machine-readable `suggested_actions` menu and on-demand
  artifact renderers (markdown / GitHub issue / test plan)
- **Resilient** — handles no-audio videos, weak local models, low-confidence cases

## Project structure

```
framesleuth/
├── framesleuth/              # Main package
│   ├── config.py            # Typed config (pydantic-settings)
│   ├── schemas.py           # Data contracts (Bug Context Bundle, enums)
│   ├── errors.py            # Exception taxonomy
│   ├── logging_config.py    # Structured JSON logging, job-id correlation
│   ├── prompts.py           # VLM / classify / summary / fix prompt templates
│   ├── skills.py            # Built-in summary skills (summary, bug_report, ...)
│   ├── actions.py           # Action modes (fix/explain/triage/...) + suggested-actions menu
│   ├── render.py            # Artifact renderers (markdown / GitHub issue / test plan)
│   ├── clients/             # VLM, coder HTTP clients (OpenAI-compatible)
│   ├── pipeline/            # preprocess, asr, scenes, understand, fusion, classify, bug_extract, redact, summarize, sidecars, grounding
│   ├── orchestrator/        # graph.py — linear async stage pipeline
│   ├── jobs/                # store.py — SQLite job state + bundle index
│   ├── service/             # FastAPI HTTP endpoints
│   └── mcp_server/          # videobug MCP server (VS Code + any MCP client)
├── tests/                   # pytest tests + fixtures
├── scripts/                 # download_models.py, dev_up.sh
├── postman/                 # HTTP API collection + environment
├── docs/                    # capabilities, use-with-vscode-and-claude, web-integration
└── pyproject.toml           # Dependencies and tool config
```

## Development

### Run tests
```bash
pytest tests/ -v --cov=framesleuth
```

### Code quality
```bash
ruff check framesleuth tests
black --check framesleuth tests
mypy --strict framesleuth
```

### Set up pre-commit hooks
```bash
pre-commit install
```

A short, focused set:

- [Capabilities](docs/capabilities.md) — the single reference: every input, output, skill, action, renderer, HTTP endpoint, and MCP tool
- [Use with VS Code & Claude (MCP)](docs/use-with-vscode-and-claude.md) — connect the `videobug` MCP server to Copilot, Claude Code, and Claude Desktop
- [Web App Integration (end-to-end)](docs/web-integration.md) — embed Framesleuth behind your own backend with an agent loop
- [Postman Collection](postman/README.md) — exercise the HTTP API end-to-end (import or run headless with Newman)
- [Runbook & Troubleshooting](runbook.md) — setup, health checks, and common issues

## License

Apache-2.0

---

## Capture client (inkwell)

Bug capture lives in a separate Chrome extension,
[inkwell](https://github.com/santoshshinde2012/inkwell). It records the bug, collects
browser sidecars (console errors, failed requests, clicks), and posts the video + sidecars
to this agent's local API. The agent's CORS is already scoped to `chrome-extension://`
origins and the loopback bind, so inkwell works against a locally running backend with no
extra setup.

**Status:** Backend + pipeline + MCP server completed.  
**Questions?** Open an issue or check [runbook.md](runbook.md) for common questions.
