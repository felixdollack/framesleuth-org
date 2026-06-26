"""Tests for the latency/quality improvements.

Covers:
- VLM payload shape (JSON mode, deterministic temperature, configurable tokens, JPEG).
- Pooled HTTP session reuse + cleanup on the VLM/Coder clients.
- Bounded-concurrency keyframe analysis with parallel overlap.
- Model-based classification refinement on the ambiguous band.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest

from framesleuth.clients.vlm import FrameAnalysisResponse, VLMClient, _encode_image
from framesleuth.config import Settings
from framesleuth.orchestrator.graph import AnalysisOrchestrator
from framesleuth.pipeline.classify import (
    classify_video,
    is_ambiguous,
    refine_classification_with_model,
)
from framesleuth.pipeline.understand import analyze_keyframes
from framesleuth.schemas import (
    Classification,
    ClassificationLabel,
    KeyframeRef,
    SceneRecord,
    Transcript,
)

# A real 1x1 PNG so OpenCV can decode + transcode it.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# ===== VLM payload shape =====


class _CapturingVLM(VLMClient):
    """VLM client that captures the outgoing payload instead of calling the network."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("http://127.0.0.1:8080", "test-model", **kwargs)
        self.payload: dict[str, Any] | None = None

    async def _call_with_retry(self, payload: dict[str, Any]) -> str:
        self.payload = payload
        import json

        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "caption": "ok",
                                    "ocr_text": "",
                                    "ui_action": None,
                                    "is_error_state": False,
                                    "reason": None,
                                }
                            )
                        }
                    }
                ]
            }
        )


@pytest.mark.asyncio
async def test_payload_uses_json_mode_zero_temp_and_token_cap(tmp_path: Path) -> None:
    """JSON mode on, deterministic temperature, and the configured token cap are sent."""
    image = tmp_path / "f.png"
    image.write_bytes(_PNG_1X1)
    client = _CapturingVLM(max_tokens=512, json_mode=True)

    await client.analyze_frame(str(image), 1.0)

    assert client.payload is not None
    assert client.payload["response_format"] == {"type": "json_object"}
    assert client.payload["temperature"] == 0.0
    assert client.payload["max_tokens"] == 512


@pytest.mark.asyncio
async def test_error_retry_uses_larger_token_budget(tmp_path: Path) -> None:
    """A per-call max_tokens override (used by the error retry) wins over the default."""
    image = tmp_path / "f.png"
    image.write_bytes(_PNG_1X1)
    client = _CapturingVLM(max_tokens=512)

    await client.analyze_frame(str(image), 1.0, prompt_override="focus on errors", max_tokens=1024)

    assert client.payload is not None
    assert client.payload["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_json_mode_can_be_disabled(tmp_path: Path) -> None:
    """response_format is omitted for engines that reject it."""
    image = tmp_path / "f.png"
    image.write_bytes(_PNG_1X1)
    client = _CapturingVLM(json_mode=False)

    await client.analyze_frame(str(image), 1.0)

    assert client.payload is not None
    assert "response_format" not in client.payload


def test_encode_image_transcodes_png_to_jpeg(tmp_path: Path) -> None:
    """A decodable PNG is sent as JPEG when send_jpeg is set; smaller and image/jpeg."""
    image = tmp_path / "f.png"
    image.write_bytes(_PNG_1X1)

    data, mime = _encode_image(str(image), send_jpeg=True, jpeg_quality=85)
    assert mime == "image/jpeg"
    # Decodes back to a valid base64 blob.
    assert base64.b64decode(data)


def test_encode_image_falls_back_to_raw_bytes_on_undecodable(tmp_path: Path) -> None:
    """A non-image file is sent unchanged with a best-effort mime, never raising."""
    image = tmp_path / "f.png"
    image.write_bytes(b"not really a png")

    data, mime = _encode_image(str(image), send_jpeg=True, jpeg_quality=85)
    assert mime == "image/png"  # extension-based fallback
    assert base64.b64decode(data) == b"not really a png"


# ===== Session reuse / cleanup =====


@pytest.mark.asyncio
async def test_vlm_session_reused_and_closed() -> None:
    """The pooled session is created once, reused, and released by aclose()."""
    client = VLMClient("http://127.0.0.1:8080", "m")
    s1 = await client._get_session()
    s2 = await client._get_session()
    assert s1 is s2
    assert not s1.closed
    await client.aclose()
    assert s1.closed
    assert client._session is None


@pytest.mark.asyncio
async def test_vlm_async_context_closes_session() -> None:
    """Using the client as an async context manager closes its session on exit."""
    async with VLMClient("http://127.0.0.1:8080", "m") as client:
        session = await client._get_session()
        assert not session.closed
    assert session.closed


# ===== Bounded-concurrency keyframe analysis =====


class _SlowConcurrencyVLM:
    """Records max concurrent in-flight calls to prove frames overlap."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    async def analyze_frame(
        self,
        image_path: str,
        timestamp: float,
        prompt_override: str | None = None,
        *,
        max_tokens: int | None = None,
        send_jpeg: bool | None = None,
    ) -> FrameAnalysisResponse:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.02)
        finally:
            self.in_flight -= 1
        return FrameAnalysisResponse(caption=f"f{timestamp}", ocr_text="")


@pytest.mark.asyncio
async def test_keyframes_analyzed_concurrently_in_order(tmp_path: Path) -> None:
    """Frames overlap (concurrency > 1) yet scenes come back in keyframe order."""
    for i in range(6):
        (tmp_path / f"{i}.png").write_bytes(_PNG_1X1)
    keyframes = [KeyframeRef(index=i, t=float(i), shows="scene", file=f"{i}.png") for i in range(6)]
    client = _SlowConcurrencyVLM()

    scenes = await analyze_keyframes(keyframes, tmp_path, client, max_concurrency=3)

    assert [s.caption for s in scenes] == [f"f{float(i)}" for i in range(6)]
    assert client.max_in_flight > 1  # genuinely overlapped
    assert client.max_in_flight <= 3  # but bounded by the semaphore


# ===== High-res, uncompressed error re-OCR =====


class _ErrorReOcrVLM:
    """Returns a sparse error frame first, then good OCR on the focused retry."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, bool | None]] = []

    async def analyze_frame(
        self,
        image_path: str,
        timestamp: float,
        prompt_override: str | None = None,
        *,
        max_tokens: int | None = None,
        send_jpeg: bool | None = None,
    ) -> FrameAnalysisResponse:
        self.calls.append((image_path, prompt_override is not None, send_jpeg))
        if prompt_override is None:
            # Looks like an error but the OCR is too sparse → triggers the retry.
            return FrameAnalysisResponse(caption="error dialog", ocr_text="", is_error_state=True)
        return FrameAnalysisResponse(
            caption="error dialog",
            ocr_text="TypeError: cannot read 'id' at profileController.js:142",
            is_error_state=True,
        )


@pytest.mark.asyncio
async def test_error_retry_uses_hires_uncompressed_frame(tmp_path: Path) -> None:
    """The error re-OCR reads the rescued full-res frame, sent uncompressed (PNG)."""
    (tmp_path / "0.png").write_bytes(_PNG_1X1)
    keyframes = [KeyframeRef(index=0, t=2.0, shows="failure", file="0.png")]
    vlm = _ErrorReOcrVLM()
    hires_path = str(tmp_path / "hires.png")

    scenes = await analyze_keyframes(keyframes, tmp_path, vlm, rescue_frame=lambda _t: hires_path)

    assert len(vlm.calls) == 2
    first_path, first_is_retry, first_jpeg = vlm.calls[0]
    retry_path, retry_is_retry, retry_jpeg = vlm.calls[1]
    # First pass: the normal 480p frame, default compression.
    assert first_path.endswith("0.png") and first_is_retry is False and first_jpeg is None
    # Retry: the rescued hi-res frame, forced uncompressed.
    assert retry_path == hires_path and retry_is_retry is True and retry_jpeg is False
    assert "profileController.js:142" in scenes[0].ocr_text


@pytest.mark.asyncio
async def test_error_retry_falls_back_when_no_rescue(tmp_path: Path) -> None:
    """With no rescue available, the retry re-reads the original frame (still uncompressed)."""
    (tmp_path / "0.png").write_bytes(_PNG_1X1)
    keyframes = [KeyframeRef(index=0, t=1.0, shows="failure", file="0.png")]
    vlm = _ErrorReOcrVLM()

    scenes = await analyze_keyframes(keyframes, tmp_path, vlm, rescue_frame=lambda _t: None)

    retry_path, _, retry_jpeg = vlm.calls[1]
    assert retry_path.endswith("0.png") and retry_jpeg is False
    assert "TypeError" in scenes[0].ocr_text


def test_rescue_frame_returns_none_on_undecodable_video(tmp_path: Path) -> None:
    """_rescue_frame degrades to None when the frame can't be decoded (no raise)."""
    from framesleuth.jobs.store import JobStore
    from framesleuth.orchestrator.graph import AnalysisOrchestrator

    settings = _settings()
    orch = AnalysisOrchestrator(settings, JobStore(tmp_path / "j.db"), None)  # type: ignore[arg-type]
    video = tmp_path / "bug.webm"
    video.write_bytes(b"not a real video")
    assert orch._rescue_frame(video, tmp_path / "work", 2.0) is None


# ===== Classification refinement =====


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


class _FakeCompleter:
    """Returns a canned completion and records the call."""

    def __init__(self, reply: str, *, boom: bool = False) -> None:
        self.reply = reply
        self.boom = boom
        self.called = False

    async def complete(
        self,
        system_prompt: str,
        user_content: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        self.called = True
        if self.boom:
            raise RuntimeError("model down")
        return self.reply


def test_is_ambiguous_band() -> None:
    """Only the [floor, threshold) band counts as ambiguous."""
    settings = _settings(CLASSIFY_AMBIGUOUS_FLOOR=0.3, CLASSIFY_CONFIDENCE_THRESHOLD=0.7)
    assert not is_ambiguous(
        Classification(label=ClassificationLabel.OTHER, confidence=0.0), settings
    )
    assert is_ambiguous(Classification(label=ClassificationLabel.OTHER, confidence=0.45), settings)
    assert not is_ambiguous(Classification(label=ClassificationLabel.BUG, confidence=0.9), settings)


@pytest.mark.asyncio
async def test_refine_replaces_ambiguous_with_model_label() -> None:
    """An ambiguous heuristic result is replaced by the model's parsed classification."""
    base = Classification(label=ClassificationLabel.OTHER, confidence=0.45)
    client = _FakeCompleter('{"label": "tutorial", "confidence": 0.88, "alt_labels": []}')

    refined = await refine_classification_with_model(
        base,
        summary_text="A walkthrough showing how to configure the app.",
        scenes=[],
        error_signals=None,
        client=client,
        settings=_settings(),
    )

    assert client.called
    assert refined.label == ClassificationLabel.TUTORIAL
    assert refined.confidence == 0.88


@pytest.mark.asyncio
async def test_refine_skips_confident_classification() -> None:
    """A confident heuristic result is never sent to the model."""
    base = Classification(label=ClassificationLabel.BUG, confidence=0.95)
    client = _FakeCompleter("unused")

    refined = await refine_classification_with_model(
        base,
        summary_text="anything",
        scenes=[],
        error_signals=None,
        client=client,
        settings=_settings(),
    )

    assert not client.called
    assert refined is base


@pytest.mark.asyncio
async def test_refine_keeps_base_when_model_fails() -> None:
    """A model error or unparseable reply must not weaken the deterministic result."""
    base = Classification(label=ClassificationLabel.OTHER, confidence=0.45)

    boom = await refine_classification_with_model(
        base,
        summary_text="x",
        scenes=[],
        error_signals=None,
        client=_FakeCompleter("", boom=True),
        settings=_settings(),
    )
    assert boom is base

    garbage = await refine_classification_with_model(
        base,
        summary_text="x",
        scenes=[],
        error_signals=None,
        client=_FakeCompleter("not json at all"),
        settings=_settings(),
    )
    assert garbage is base


@pytest.mark.asyncio
async def test_refine_disabled_by_flag() -> None:
    """CLASSIFY_USE_MODEL=false keeps the deterministic result untouched."""
    base = Classification(label=ClassificationLabel.OTHER, confidence=0.45)
    client = _FakeCompleter("unused")

    refined = await refine_classification_with_model(
        base,
        summary_text="x",
        scenes=[SceneRecord(t0=0.0, t1=0.0, caption="c", ocr_text="")],
        error_signals=None,
        client=client,
        settings=_settings(CLASSIFY_USE_MODEL=False),
    )

    assert not client.called
    assert refined is base


# ===== Bounded resample loop (MAX_RESAMPLE_RETRIES) =====


def _orchestrator(settings: Settings) -> AnalysisOrchestrator:
    """Build an orchestrator whose store/VLM are unused by the resample path."""
    return AnalysisOrchestrator(settings, store=None, vlm_client=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_resample_adds_frames_and_sharpens_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ambiguous error frame triggers a resample that adds evidence and sharpens the label."""
    settings = _settings(MAX_RESAMPLE_RETRIES=2)
    orch = _orchestrator(settings)

    # One error-state frame with no OCR markers scores 0.45 — squarely ambiguous.
    scenes = [SceneRecord(t0=1.0, t1=1.0, caption="spinner", ocr_text="", is_error_state=True)]
    analyzed = [KeyframeRef(index=0, t=1.0, shows="failure", file="frames/0.png")]
    transcript = Transcript(segments=[], words=[])
    base = classify_video(transcript, scenes, settings=settings)
    assert is_ambiguous(base, settings)

    async def _fake_extra(
        video_path: Path, work_dir: Path, times: list[float], start_index: int
    ) -> tuple[list[SceneRecord], list[KeyframeRef]]:
        new_scenes = [
            SceneRecord(
                t0=t, t1=t, caption="error", ocr_text="TypeError: boom", is_error_state=True
            )
            for t in times
        ]
        new_frames = [
            KeyframeRef(
                index=start_index + i,
                t=t,
                shows="failure",
                file=f"frames_resample_{start_index}/{i}.png",
            )
            for i, t in enumerate(times)
        ]
        return new_scenes, new_frames

    monkeypatch.setattr(orch, "_analyze_extra", _fake_extra)

    metrics: dict[str, Any] = {"stages": {}, "degraded": []}
    refined = await orch._maybe_resample(
        Path("/x/bug.webm"), Path("/x/.work"), 12.0, scenes, analyzed, transcript, [], base, metrics
    )

    # Resampled OCR markers pushed confidence over the threshold.
    assert refined.label == ClassificationLabel.BUG
    assert len(scenes) > 1 and len(analyzed) > 1  # extended in lockstep
    assert metrics["resample_attempts"] >= 1
    assert "resample" in metrics["stages"]


@pytest.mark.asyncio
async def test_resample_disabled_when_retries_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """MAX_RESAMPLE_RETRIES=0 is a hard off switch — no extra analysis at all."""
    settings = _settings(MAX_RESAMPLE_RETRIES=0)
    orch = _orchestrator(settings)
    scenes = [SceneRecord(t0=1.0, t1=1.0, caption="spinner", ocr_text="", is_error_state=True)]
    transcript = Transcript(segments=[], words=[])
    base = classify_video(transcript, scenes, settings=settings)

    called = False

    async def _boom(*_a: object, **_k: object) -> tuple[list[SceneRecord], list[KeyframeRef]]:
        nonlocal called
        called = True
        return [], []

    monkeypatch.setattr(orch, "_analyze_extra", _boom)

    refined = await orch._maybe_resample(
        Path("/x/bug.webm"),
        Path("/x/.work"),
        12.0,
        scenes,
        [],
        transcript,
        [],
        base,
        {"stages": {}, "degraded": []},
    )
    assert refined is base
    assert not called


@pytest.mark.asyncio
async def test_resample_terminates_when_no_new_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """If resampled frames add nothing, the loop stops instead of spinning."""
    settings = _settings(MAX_RESAMPLE_RETRIES=3)
    orch = _orchestrator(settings)
    scenes = [SceneRecord(t0=1.0, t1=1.0, caption="spinner", ocr_text="", is_error_state=True)]
    transcript = Transcript(segments=[], words=[])
    base = classify_video(transcript, scenes, settings=settings)

    calls = 0

    async def _empty(*_a: object, **_k: object) -> tuple[list[SceneRecord], list[KeyframeRef]]:
        nonlocal calls
        calls += 1
        return [], []  # no new frames decode

    monkeypatch.setattr(orch, "_analyze_extra", _empty)

    refined = await orch._maybe_resample(
        Path("/x/bug.webm"),
        Path("/x/.work"),
        12.0,
        scenes,
        [KeyframeRef(index=0, t=1.0, shows="failure", file="frames/0.png")],
        transcript,
        [],
        base,
        {"stages": {}, "degraded": []},
    )
    assert refined is base
    assert calls == 1  # tried once, got nothing, stopped
