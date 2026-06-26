"""Tests for preprocess pipeline utilities."""

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from framesleuth.config import Settings
from framesleuth.errors import DurationExceededError
from framesleuth.pipeline.preprocess import (
    compute_sample_count,
    compute_sample_timestamps,
    extract_audio,
    extract_frames,
    preprocess_video,
    probe_video,
    scan_last_timestamp,
)


class _FakePacket:
    """Minimal stand-in for an ``av.Packet`` for timestamp-scan tests."""

    def __init__(self, pts: int | None, duration: int, time_base: Fraction) -> None:
        self.pts = pts
        self.duration = duration
        self.stream = SimpleNamespace(time_base=time_base)


class _FakeContainer:
    """Container whose ``demux()`` replays a fixed packet sequence."""

    def __init__(self, packets: list[_FakePacket]) -> None:
        self._packets = packets

    def demux(self) -> list[_FakePacket]:
        return self._packets


def test_scan_last_timestamp_recovers_duration_from_packets() -> None:
    """Duration is the end (pts + duration) of the latest packet across streams."""
    tb = Fraction(1, 1000)  # millisecond timebase, like MediaRecorder WebM
    packets = [
        _FakePacket(pts=0, duration=20, time_base=tb),
        _FakePacket(pts=3500, duration=20, time_base=tb),  # audio runs to ~3.52s
        _FakePacket(pts=4, duration=0, time_base=tb),  # sparse video track near t=0
    ]
    assert scan_last_timestamp(_FakeContainer(packets)) == pytest.approx(3.52)


def test_scan_last_timestamp_ignores_packets_without_timing() -> None:
    """Flush/None-pts packets must not raise and must not affect the estimate."""
    tb = Fraction(1, 1000)
    packets = [
        _FakePacket(pts=None, duration=0, time_base=tb),
        _FakePacket(pts=1000, duration=0, time_base=None),  # type: ignore[arg-type]
    ]
    assert scan_last_timestamp(_FakeContainer(packets)) == 0.0


def test_scan_last_timestamp_never_raises_on_broken_demux() -> None:
    """A demux that blows up degrades to 0.0 rather than aborting the probe."""

    class _Exploding:
        def demux(self) -> list[_FakePacket]:
            raise RuntimeError("corrupt container")

    assert scan_last_timestamp(_Exploding()) == 0.0


def test_compute_sample_count_caps_by_duration() -> None:
    """Frame budget should scale linearly with duration and configured cap."""
    # 2 minutes at 30 fpm => 60 sampled frames
    assert compute_sample_count(120.0, 30) == 60


def test_compute_sample_timestamps_deterministic() -> None:
    """Sampling should be deterministic for reproducible fixtures."""
    a = compute_sample_timestamps(10.0, 30)
    b = compute_sample_timestamps(10.0, 30)
    assert a == b
    assert len(a) == 5


def test_preprocess_no_audio_metadata_override(tmp_path: Path) -> None:
    """Preprocess should support deterministic metadata override for tests."""
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"dummy")

    settings = Settings(MAX_FRAMES_PER_MIN=30, MAX_DURATION_S=600)
    result = preprocess_video(
        video,
        settings=settings,
        metadata_override={
            "duration_s": 60.0,
            "fps": 30.0,
            "width": 1280,
            "height": 720,
            "frame_count": 1800,
            "has_audio": False,
        },
    )

    assert result.has_audio is False
    assert result.frame_count == 1800
    assert result.metadata["sample_count"] == 30


def test_preprocess_rejects_long_videos(tmp_path: Path) -> None:
    """Duration limits should be enforced with typed exceptions."""
    video = tmp_path / "long.mp4"
    video.write_bytes(b"dummy")

    with pytest.raises(DurationExceededError):
        preprocess_video(
            video,
            settings=Settings(MAX_DURATION_S=10, MAX_FRAMES_PER_MIN=30),
            metadata_override={"duration_s": 11.0, "fps": 30.0, "frame_count": 330},
        )


def test_probe_video_degrades_on_undecodable_input(tmp_path: Path) -> None:
    """Probing a non-video file must never raise — it returns an empty dict."""
    fake = tmp_path / "broken.mp4"
    fake.write_bytes(b"not a real container")
    assert probe_video(fake) == {}


def test_preprocess_without_override_does_not_crash(tmp_path: Path) -> None:
    """The real (probe) path must degrade to a zero-duration result, not raise."""
    fake = tmp_path / "clip.mp4"
    fake.write_bytes(b"not a real container")
    result = preprocess_video(fake, settings=Settings())
    assert result.duration_s == 0.0
    assert result.metadata["sample_count"] == 1


def test_extract_frames_degrades_without_media_deps(tmp_path: Path) -> None:
    """Frame extraction returns an empty list when av/cv2 or the file are unusable."""
    fake = tmp_path / "clip.mp4"
    fake.write_bytes(b"not a real container")
    assert extract_frames(fake, [0.0, 1.0], tmp_path / "frames") == []
    assert extract_frames(fake, [], tmp_path / "frames") == []


def test_extract_audio_degrades_without_media_deps(tmp_path: Path) -> None:
    """Audio extraction returns None when there is no decodable audio stream."""
    fake = tmp_path / "clip.mp4"
    fake.write_bytes(b"not a real container")
    assert extract_audio(fake, tmp_path / "out") is None
