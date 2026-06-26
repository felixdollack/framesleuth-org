# Framesleuth Runbook

**Setup, health checks, troubleshooting, and common operational tasks.**

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Initial setup](#initial-setup)
3. [Model downloading](#model-downloading)
4. [Starting services](#starting-services)
5. [Health checks](#health-checks)
6. [Common issues](#common-issues)
7. [Testing the system](#testing-the-system)
8. [Logs and debugging](#logs-and-debugging)

---

## Prerequisites

### macOS

```bash
brew install python@3.11 ffmpeg
# Install uv for Python management
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv ffmpeg
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
# Download Python 3.11 from python.org
# Install ffmpeg via scoop or download from ffmpeg.org
scoop install ffmpeg
# Install uv
irm https://astral.sh/uv/install.ps1 | iex
```

### NVIDIA GPU support (Linux)

If you plan to use vLLM for higher throughput:
```bash
sudo apt install -y nvidia-cuda-toolkit nvidia-utils
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Initial setup

### 1. Clone and install

```bash
git clone https://github.com/santoshshinde2012/framesleuth.git
cd framesleuth
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your preferences
nano .env
```

Key settings:
- `ENGINE_PROFILE`: `local-default` (Ollama+llama.cpp) or `local-onestack` (all llama.cpp)
- Model URLs and model names
- Storage paths
- Concurrency limits

### 3. Validate configuration

```bash
python -c "from framesleuth.config import get_settings; s = get_settings(); s.validate_paths(); print('✓ Config OK')"
```

---

## Model downloading

### One-time setup

```bash
python scripts/download_models.py
```

This script currently **checks for** the expected local model files and reports
what is missing; checksum verification is stubbed (placeholder SHA256) and it does
**not** itself download weights or run `ollama pull`. Fetch the models via the
engine you use:
- VLM (llama.cpp): `llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF ...` (pulls on first run)
- VLM (Ollama path): `ollama pull qwen2.5vl`
- Coder (Ollama): `ollama pull qwen2.5-coder:7b`
- Whisper: downloaded automatically by `faster-whisper` on first transcription

### What gets downloaded

| Model | Size | Purpose | Local path |
|---|---|---|---|
| Qwen3-VL-8B-Instruct-GGUF | ~8GB | Frame understanding | `~/.cache/huggingface/` |
| Qwen3-VL-8B-mmproj-fp16.gguf | ~2GB | Vision encoder | `~/.cache/huggingface/` |
| Whisper | ~1GB | Speech-to-text | `~/.cache/whisper/` |
| Qwen2.5-Coder:7b | ~5GB | Code fixing | Ollama blob store |

**Total: ~16-20GB**

---

## Starting services

### Option 0: Vision via Ollama (lightest — no llama.cpp, no GGUF download)

The pipeline is engine-agnostic: the VLM only needs an OpenAI-compatible
`/v1/chat/completions` endpoint that accepts images. Ollama provides that, so you
can skip `llama-server` and the ~20 GB Qwen3-VL GGUF entirely:

```bash
ollama pull qwen2.5vl            # ~6 GB vision model (or qwen2.5vl:3b, ~3 GB)
```

Point the backend at Ollama in `.env`:

```env
VLM_URL=http://127.0.0.1:11434
VLM_MODEL=qwen2.5vl
```

Then `framesleuth-api`. `GET /v1/healthz` should report `vlm: ready`, and runs
will populate `keyframes/` and produce real frame OCR/captions (`analysis_quality.level: full`).
**Without a vision model, `understand` degrades** and — unless the recording
carries browser sidecars — the bundle has nothing to summarize.

### Option A: All-in-one (local-default)

```bash
# Terminal 1: VLM server (llama.cpp)
llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF \
  --mmproj ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct-GGUF/snapshots/*/qwen3-vl-mmproj.gguf \
  --n-gpu-layers 99 -c 32768 --port 8080

# Terminal 2: Coder server (Ollama)
OLLAMA_KEEP_ALIVE=-1 ollama serve
# In separate shell: ollama pull qwen2.5-coder:7b

# Terminal 3: Framesleuth backend
framesleuth-api

# Terminal 4: MCP server
framesleuth-mcp
```

### Option B: Docker Compose (all services)

```bash
docker compose --profile local-default up
# or
docker compose --profile vllm up  # for server profile
```

### Option C: Helper script

```bash
./scripts/dev_up.sh   # run it — do NOT `source` it (sourcing can close your shell)
# Starts all services in the background (detached). Stop with:
#   docker compose --profile local-default down
```

---

## Health checks

### Backend health

```bash
curl http://127.0.0.1:8010/v1/healthz
```

Expected response (all services up):
```json
{
  "status": "healthy",
  "services": {
    "vlm": { "name": "vlm", "status": "ready", "latency_ms": null, "error": null },
    "coder": { "name": "coder", "status": "ready", "latency_ms": null, "error": null },
    "storage": { "name": "storage", "status": "ready", "latency_ms": null, "error": null }
  },
  "queue_depth": 0,
  "timestamp": "2026-06-22T08:25:19.936358+00:00"
}
```

Without the model servers running you'll instead see `"status": "unhealthy"`
with `vlm`/`coder` reporting `"status": "unavailable"` and `storage: ready`.
That is expected — analysis still runs in degraded (sidecar-only) mode, and the
resulting bundle's `analysis_quality` records what was skipped.

### Model servers

```bash
# VLM (llama.cpp)
curl http://127.0.0.1:8080/v1/models

# Coder (Ollama)
curl http://127.0.0.1:11434/api/tags
```

### MCP server

```bash
# Check logs
tail -f .framesleuth-mcp.log
```

---

## Common issues

### Issue: "VLM server not responding (503)"

**Cause:** llama.cpp not running or model not loaded.

**Fix:**
```bash
# Check if running
lsof -i :8080

# Restart and wait for model load (~1-2 min)
llama-server ... --n-gpu-layers 99 -c 32768 --port 8080
# Wait for "llama_server: server is listening"
```

### Issue: "Coder unavailable (503)"

**Cause:** Ollama not running or model not pulled.

**Fix:**
```bash
# Check if running
pgrep ollama || OLLAMA_KEEP_ALIVE=-1 ollama serve

# Pull model
ollama pull qwen2.5-coder:7b
```

### Issue: "Upload too large (413)"

**Cause:** Video exceeds `MAX_UPLOAD_MB` limit.

**Fix:**
- Increase limit in `.env` (`MAX_UPLOAD_MB=1024`)
- Or record shorter videos

### Issue: "Out of memory (OOM)"

**Cause:** Models competing for VRAM.

**Fix:**
- Reduce `MAX_CONCURRENT_JOBS` in `.env` (default 2)
- Use smaller quant (8B instead of 13B)
- Enable GPU offloading with `--n-gpu-layers 99`

### Issue: "FFmpeg: No such file"

**Cause:** ffmpeg not installed or not on PATH.

**Fix:**
```bash
# Install
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Linux

# Verify
ffmpeg -version
```

---

## Testing the system

### Run unit tests

```bash
pytest tests/ -v --cov=framesleuth
```

### Run integration tests (requires services)

```bash
pytest tests/ -v -m integration
```

### Create a fixture report (end-to-end)

Analysis is **asynchronous**: `POST /v1/analyze` returns `202` with a `job_id`
immediately and runs the pipeline in the background. Poll `/v1/jobs/{id}` until
`state` is `done`, then read the bundle.

```bash
# 1. Queue a sample (returns 202 {job_id, status: "queued"}):
JOB=$(curl -s -X POST http://127.0.0.1:8010/v1/analyze \
  -F "video=@samples/flash_bug.mp4" \
  -F "intent=Find why the Save button hangs and fix it." \
  | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "job_id=$JOB"

# 2. Poll until the job reports state: done (or failed):
curl -s http://127.0.0.1:8010/v1/jobs/$JOB | python -m json.tool

# 3. Read the Bug Context Bundle:
curl -s http://127.0.0.1:8010/v1/report/$JOB | python -m json.tool
```

> Idempotency: re-posting the *same bytes* returns the existing job
> (`idempotent: true`) without re-running. Use a different file, or clear the
> scratch store (`rm -rf bug-reports/*`) to force a fresh run.

Prefer a GUI? Import the Postman collection in [`postman/`](postman/README.md).

---

## Logs and debugging

### Backend logs

**Real-time:**
```bash
tail -f bug-reports/jobs.db-journal
# or in the backend terminal
```

**Per-job:**
```bash
# After a job completes, check:
cat bug-reports/{job-id}/job.log
cat bug-reports/{job-id}/metrics.json
```

### Model server logs

**llama.cpp:**
```bash
# Logs to stdout; check terminal where llama-server started
```

**Ollama:**
```bash
# macOS: ~/Library/Application Support/Ollama/logs
# Linux: ~/.ollama/logs
```

### Debug mode

```bash
LOG_LEVEL=DEBUG framesleuth-api
```

---

## Configuration tuning

### For low-end hardware (4GB RAM)

```env
MAX_CONCURRENT_JOBS=1
FRAME_LOWRES_HEIGHT=360
MAX_FRAMES_PER_MIN=15
VLM_TIMEOUT_S=120  # Give more time
```

### For production (shared GPU)

```env
ENGINE_PROFILE=server
MAX_CONCURRENT_JOBS=4
FRAME_LOWRES_HEIGHT=480
MAX_FRAMES_PER_MIN=60
CLASSIFY_CONFIDENCE_THRESHOLD=0.8  # Higher bar
```

---

## Next steps

- [Capabilities](docs/capabilities.md) — full reference for inputs, outputs, endpoints, and MCP tools
- [Use with VS Code & Claude (MCP)](docs/use-with-vscode-and-claude.md) — connect an MCP client
- [Postman Collection](postman/README.md) — exercise the HTTP API end-to-end
