"""Tests for the GIF preview pipeline."""

from pathlib import Path

import numpy as np

from framesleuth.pipeline.gif import GifOptions, encode_gif, normalize_options


def _write_sample_video(path: Path, *, frames: int = 20, fps: int = 10) -> None:
    """Encode a tiny synthetic mp4 so the test is self-contained (no fixtures)."""
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 128
        stream.height = 96
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            arr = np.full((96, 128, 3), (i * 10) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_normalize_options_clamps_to_safe_ranges() -> None:
    """Out-of-range fps/width and an unbounded window are clamped, not trusted."""
    opts = normalize_options(fps=999, width=99999, start=-5, end=None, max_duration_s=30.0)
    assert 1 <= opts.fps <= 30
    assert 64 <= opts.width <= 1280
    assert opts.start == 0.0
    # No end given → start + max_duration.
    assert opts.end == 30.0


def test_normalize_options_caps_window_to_max_duration() -> None:
    """An explicit end longer than the cap is shortened to the cap."""
    opts = normalize_options(fps=10, width=640, start=2.0, end=100.0, max_duration_s=5.0)
    assert opts.end == 7.0


def test_cache_key_is_stable_and_parameter_specific() -> None:
    a = GifOptions(fps=10, width=640, start=0.0, end=3.0)
    b = GifOptions(fps=10, width=640, start=0.0, end=3.0)
    c = GifOptions(fps=8, width=640, start=0.0, end=3.0)
    assert a.cache_key() == b.cache_key()
    assert a.cache_key() != c.cache_key()


def test_encode_gif_produces_valid_animated_gif(tmp_path: Path) -> None:
    """A real source video encodes to a non-empty GIF with a valid header."""
    src = tmp_path / "source.mp4"
    _write_sample_video(src)
    out = tmp_path / "preview.gif"

    opts = normalize_options(fps=8, width=96, start=0.0, end=None, max_duration_s=30.0)
    result = encode_gif(src, out, options=opts)

    assert result == out
    assert out.exists()
    data = out.read_bytes()
    assert data[:6] in (b"GIF87a", b"GIF89a")
    assert len(data) > 64
    # The atomic temp file is cleaned up, never left behind.
    assert not list(tmp_path.glob("*.tmp"))


def test_encode_gif_downscales_to_requested_width(tmp_path: Path) -> None:
    """Width is honored by downscaling; the source is never upscaled."""
    src = tmp_path / "source.mp4"
    _write_sample_video(src)
    out = tmp_path / "preview.gif"

    opts = GifOptions(fps=8, width=64, start=0.0, end=2.0)
    assert encode_gif(src, out, options=opts) == out

    import av

    with av.open(str(out)) as container:
        frame = next(container.decode(container.streams.video[0]))
        assert frame.width == 64


def test_encode_gif_returns_none_for_missing_source(tmp_path: Path) -> None:
    """A missing source degrades to None rather than raising."""
    opts = GifOptions(fps=8, width=96, start=0.0, end=2.0)
    assert encode_gif(tmp_path / "nope.mp4", tmp_path / "out.gif", options=opts) is None


def test_encode_gif_returns_none_for_unreadable_source(tmp_path: Path) -> None:
    """A non-video file degrades to None and leaves no partial output."""
    src = tmp_path / "source.mp4"
    src.write_bytes(b"not a real video")
    out = tmp_path / "out.gif"
    opts = GifOptions(fps=8, width=96, start=0.0, end=2.0)
    assert encode_gif(src, out, options=opts) is None
    assert not out.exists()
    assert not list(tmp_path.glob("*.tmp"))
