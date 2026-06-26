"""Video preprocessing pipeline: probe, frame sampling, and optional audio extraction.

Heavy media work (container probing, frame decoding, audio export) is done with
PyAV (``av``), which bundles its own ffmpeg libraries — no system ffmpeg binary
is required. Every PyAV call is wrapped so that a missing ``av``/``cv2`` wheel or
an unreadable file degrades to a safe empty result rather than aborting the run.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from framesleuth.config import Settings, get_settings
from framesleuth.errors import DurationExceededError, UnsupportedMediaError
from framesleuth.logging_config import get_logger
from framesleuth.schemas import PreprocessResult

logger = get_logger("pipeline.preprocess")


_SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


@dataclass(frozen=True)
class ExtractedFrame:
    """A single frame decoded to disk for visual understanding."""

    t: float
    file: str  # path relative to the frames directory's parent, e.g. "frames/0.png"
    change_score: float  # visual delta vs the previous frame (0.0 if unknown)


def compute_sample_count(duration_s: float, max_frames_per_min: int) -> int:
    """Compute deterministic frame budget constrained by duration and limits."""
    if duration_s <= 0:
        return 1
    max_frames = max(1, math.ceil((duration_s / 60.0) * max_frames_per_min))
    return max_frames


def compute_sample_timestamps(duration_s: float, max_frames_per_min: int) -> list[float]:
    """Compute evenly spaced timestamps with stable deterministic ordering."""
    frame_budget = compute_sample_count(duration_s, max_frames_per_min)
    if frame_budget == 1:
        return [0.0]

    step = duration_s / frame_budget
    return [round(i * step, 3) for i in range(frame_budget)]


def _run_ffprobe_has_audio(video_path: Path) -> bool:
    """Best-effort ffprobe audio stream check; returns False if ffprobe is missing."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return False

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return bool(proc.stdout.strip())


def scan_last_timestamp(container: Any) -> float:
    """Return the end timestamp (seconds) of the last packet across all streams.

    Duration fallback for containers whose header omits it — notably Chrome's
    ``MediaRecorder`` WebM, which writes no container/stream duration, no frame
    count and no average rate. Reads packet timestamps only (no full decode), so
    the audio track alone is enough to recover a realistic duration even when the
    video track is sparse. Never raises: returns ``0.0`` if nothing can be read.
    """
    best = 0.0
    try:
        for packet in container.demux():
            if packet.pts is None or packet.stream.time_base is None:
                continue
            # pts marks the packet start; add its duration to reach the end.
            span = packet.pts + (packet.duration or 0)
            best = max(best, float(span * packet.stream.time_base))
    except Exception as exc:
        logger.warning("Timestamp scan failed: %s", exc)
    return best


def probe_video(video_path: Path) -> dict[str, Any]:
    """Probe container metadata with PyAV; return zeros if PyAV is unavailable.

    Returns a dict with ``duration_s``, ``fps``, ``width``, ``height``,
    ``frame_count`` and ``has_audio``. Never raises: a missing ``av`` wheel or an
    unreadable container yields an all-zero/``has_audio=False`` result so the
    caller can degrade to sidecar-only analysis.
    """
    try:
        import av  # lazy: only needed when real probing is requested
    except Exception as exc:  # pragma: no cover - exercised only without PyAV
        logger.warning("PyAV unavailable, skipping probe: %s", exc)
        return {}

    try:
        with av.open(str(video_path)) as container:
            vstreams = container.streams.video
            astreams = container.streams.audio
            duration_s = float(container.duration / av.time_base) if container.duration else 0.0
            fps = 0.0
            width = height = frame_count = 0
            if vstreams:
                vstream = vstreams[0]
                if vstream.average_rate:
                    fps = float(vstream.average_rate)
                width = int(vstream.width or 0)
                height = int(vstream.height or 0)
                frame_count = int(vstream.frames or 0)
                if duration_s <= 0 and vstream.duration and vstream.time_base:
                    duration_s = float(vstream.duration * vstream.time_base)
            # Last resort when the header carries no duration metadata at all:
            # scan packet timestamps. This rescues Chrome MediaRecorder WebM,
            # where leaving duration at 0 would collapse sampling to one frame.
            if duration_s <= 0:
                duration_s = scan_last_timestamp(container)
            return {
                "duration_s": duration_s,
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_count,
                "has_audio": bool(astreams),
            }
    except Exception as exc:
        logger.warning("PyAV probe failed for %s: %s", video_path, exc)
        return {}


def _decode_frame_at(container: Any, vstream: Any, time_base: Any, t: float) -> Any:
    """Seek then decode forward to the frame at/after ``t``.

    A bare seek lands on the nearest preceding keyframe, so without decoding
    forward every timestamp would reuse the same keyframe. Returns the frame, or
    ``None`` if nothing decodes.
    """
    if time_base:
        container.seek(int(t / time_base), stream=vstream, backward=True)
    frame = None
    for candidate in container.decode(vstream):
        frame = candidate
        if candidate.pts is None or not time_base:
            break
        if float(candidate.pts * time_base) >= t - 1e-3:
            break
    return frame


def extract_frames(
    video_path: Path,
    timestamps: list[float],
    out_dir: Path,
    *,
    height: int = 480,
) -> list[ExtractedFrame]:
    """Decode frames at ``timestamps`` to ``out_dir`` as PNGs via PyAV + OpenCV.

    Returns one :class:`ExtractedFrame` per successfully written frame, each with
    a change score (mean absolute delta vs the previous frame) so downstream
    scene selection can find cuts. Degrades to ``[]`` when ``av``/``cv2`` are
    missing or the video cannot be decoded — never raises.
    """
    if not timestamps:
        return []
    try:
        import av  # lazy: heavy media deps only present in the full stack
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - exercised only without media deps
        logger.warning("Frame extraction unavailable (av/cv2/numpy missing): %s", exc)
        return []

    try:
        container = av.open(str(video_path))
    except Exception as exc:
        logger.warning("Could not open %s for frame extraction: %s", video_path, exc)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[ExtractedFrame] = []
    prev: Any = None
    try:
        vstream = container.streams.video[0]
        time_base = vstream.time_base
        for index, t in enumerate(timestamps):
            try:
                frame = _decode_frame_at(container, vstream, time_base, t)
                if frame is None:
                    continue
                arr = frame.to_ndarray(format="bgr24")
                if height and arr.shape[0] > height:
                    width = int(arr.shape[1] * height / arr.shape[0])
                    arr = cv2.resize(arr, (width, height))
                score = 0.0 if prev is None else float(np.mean(cv2.absdiff(arr, prev)) / 255.0)
                prev = arr
                target = out_dir / f"{index}.png"
                if cv2.imwrite(str(target), arr):
                    rel = f"{out_dir.name}/{index}.png"
                    frames.append(ExtractedFrame(t=float(t), file=rel, change_score=score))
            except Exception as exc:  # one bad frame must not sink the batch
                logger.debug("Frame %s extract failed at t=%.3f: %s", index, t, exc)
    finally:
        container.close()
    logger.info("Extracted %d/%d frames to %s", len(frames), len(timestamps), out_dir)
    return frames


def extract_audio(video_path: Path, out_dir: Path) -> Path | None:
    """Export the first audio stream to a 16 kHz mono WAV for ASR, or ``None``.

    Returns ``None`` when there is no audio stream, when PyAV is unavailable, or
    on any decode/encode error — ASR treats ``None`` as "no audio" and yields an
    empty transcript.
    """
    try:
        import av  # lazy
    except Exception as exc:  # pragma: no cover - exercised only without PyAV
        logger.warning("PyAV unavailable, skipping audio extraction: %s", exc)
        return None

    try:
        with av.open(str(video_path)) as in_container:
            if not in_container.streams.audio:
                return None
            out_dir.mkdir(parents=True, exist_ok=True)
            audio_path = out_dir / "audio.wav"
            in_stream = in_container.streams.audio[0]
            with av.open(str(audio_path), mode="w") as out_container:
                out_stream = out_container.add_stream("pcm_s16le", rate=16000)
                out_stream.layout = "mono"
                for frame in in_container.decode(in_stream):
                    frame.pts = None
                    for packet in out_stream.encode(frame):
                        out_container.mux(packet)
                for packet in out_stream.encode(None):
                    out_container.mux(packet)
            return audio_path
    except Exception as exc:
        logger.warning("Audio extraction failed for %s: %s", video_path, exc)
        return None


def preprocess_video(
    video_path: Path,
    *,
    settings: Settings | None = None,
    metadata_override: dict[str, Any] | None = None,
) -> PreprocessResult:
    """Preprocess a video into typed metadata for downstream perception stages.

    When ``metadata_override`` is absent the container is probed with PyAV; the
    override path is intended for deterministic tests and fixture mode.
    """
    settings = settings or get_settings()

    if not video_path.exists():
        raise UnsupportedMediaError(str(video_path), "file does not exist")
    if video_path.suffix.lower() not in _SUPPORTED_VIDEO_EXTS:
        raise UnsupportedMediaError(str(video_path), "unsupported extension")

    # Real probing only when no deterministic override was supplied.
    probed = {} if metadata_override is not None else probe_video(video_path)
    metadata = metadata_override if metadata_override is not None else probed
    duration_s = float(metadata.get("duration_s", 0.0))
    fps = float(metadata.get("fps", 0.0))
    width = int(metadata.get("width", 0))
    height = int(metadata.get("height", 0))
    frame_count = int(metadata.get("frame_count", 0))
    has_audio = bool(metadata.get("has_audio", _run_ffprobe_has_audio(video_path)))

    if duration_s <= 0 and fps > 0 and frame_count > 0:
        duration_s = frame_count / fps
    if duration_s > settings.MAX_DURATION_S:
        raise DurationExceededError(duration_s, settings.MAX_DURATION_S)

    sampled_timestamps = compute_sample_timestamps(duration_s, settings.MAX_FRAMES_PER_MIN)

    return PreprocessResult(
        video_path=video_path,
        duration_s=duration_s,
        fps=fps,
        width=width,
        height=height,
        has_audio=has_audio,
        audio_path=None,
        frame_count=frame_count,
        metadata={
            **metadata,
            "sampled_timestamps": sampled_timestamps,
            "sample_count": len(sampled_timestamps),
        },
    )
