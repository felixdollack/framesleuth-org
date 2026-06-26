"""Tests for FastAPI service endpoints."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from framesleuth.config import Settings
from framesleuth.schemas import JobState
from framesleuth.service.api import create_app


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        BUNDLE_DIR=tmp_path / "bundles",
        DATABASE_PATH=tmp_path / "jobs.db",
        MAX_UPLOAD_MB=10,
        CHROME_EXTENSION_ORIGIN="chrome-extension://test",
    )


def test_analyze_and_report_roundtrip(tmp_path: Path) -> None:
    """Analyze endpoint should create idempotent job and report should be retrievable."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(
            json.dumps({"id": job_id, "title": "test", "source_video": source_video}),
            encoding="utf-8",
        )
        await state.store.update_job(
            job_id,
            state=JobState.DONE,
            progress_pct=100,
            bundle_path=str(bundle_path),
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run

        response = client.post(
            "/v1/analyze",
            files={"video": ("sample.mp4", b"video-content", "video/mp4")},
        )
        # Analysis is accepted (202) and runs in the background; the TestClient
        # drains the background task before returning, so the job is done by now.
        assert response.status_code == 202
        payload = response.json()
        assert payload["status"] == "queued"
        job_id = payload["job_id"]

        job = client.get(f"/v1/jobs/{job_id}")
        assert job.status_code == 200
        assert job.json()["state"] == "done"

        report = client.get(f"/v1/report/{job_id}")
        assert report.status_code == 200
        assert report.json()["id"] == job_id

        # The temp upload is cleaned up, not left accumulating in the bundle dir.
        assert not list(app.state.app_state.settings.BUNDLE_DIR.glob("upload-*"))

        # Same upload should hit idempotency path.
        second = client.post(
            "/v1/analyze",
            files={"video": ("sample.mp4", b"video-content", "video/mp4")},
        )
        assert second.status_code == 202
        assert second.json()["idempotent"] == "true"


def test_analyze_records_typed_failure_in_background(tmp_path: Path) -> None:
    """A typed pipeline failure marks the job FAILED with a structured error."""
    from framesleuth.errors import UnsupportedMediaError

    app = create_app(_make_settings(tmp_path))

    async def boom(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        raise UnsupportedMediaError("bad codec")

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = boom
        resp = client.post("/v1/analyze", files={"video": ("x.mp4", b"vid", "video/mp4")})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        job = client.get(f"/v1/jobs/{job_id}").json()
        assert job["state"] == "failed"
        assert job["error"]["code"] == "unsupported_media"

        # The temp upload is still cleaned up on the failure path.
        assert not list(app.state.app_state.settings.BUNDLE_DIR.glob("upload-*"))


def test_get_video_serves_correct_media_type(tmp_path: Path) -> None:
    """The stored source is served with a content-type matching its container."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        # An mp4 source must not be advertised as webm.
        (bundle_dir / "source.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        job_id = client.post(
            "/v1/analyze", files={"video": ("bug.mp4", b"vid", "video/mp4")}
        ).json()["job_id"]

        video = client.get(f"/v1/video/{job_id}")
        assert video.status_code == 200
        assert video.headers["content-type"] == "video/mp4"


def _write_sample_video(path: Path, *, frames: int = 16, fps: int = 8) -> None:
    """Encode a tiny synthetic mp4 for endpoints that need a real recording."""
    import av
    import numpy as np

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 128
        stream.height = 96
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            arr = np.full((96, 128, 3), (i * 12) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_get_gif_renders_and_caches_preview(tmp_path: Path) -> None:
    """GET /v1/gif encodes a GIF from the stored source and caches it on disk."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        _write_sample_video(bundle_dir / "source.mp4")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        job_id = client.post(
            "/v1/analyze", files={"video": ("bug.mp4", b"vid", "video/mp4")}
        ).json()["job_id"]

        resp = client.get(f"/v1/gif/{job_id}", params={"fps": 6, "width": 96})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/gif"
        assert resp.content[:6] in (b"GIF87a", b"GIF89a")

        # The render is cached on disk keyed by its parameters.
        bundle_dir = app.state.app_state.settings.BUNDLE_DIR / job_id
        cached = list(bundle_dir.glob("preview-*.gif"))
        assert len(cached) == 1

        # A second identical request reuses the cache (still 200, same file).
        again = client.get(f"/v1/gif/{job_id}", params={"fps": 6, "width": 96})
        assert again.status_code == 200
        assert len(list(bundle_dir.glob("preview-*.gif"))) == 1


def test_get_gif_returns_404_when_no_source(tmp_path: Path) -> None:
    """A job without a stored recording yields a 404, not a 500."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        job_id = client.post(
            "/v1/analyze", files={"video": ("bug.mp4", b"vid", "video/mp4")}
        ).json()["job_id"]

        resp = client.get(f"/v1/gif/{job_id}")
        assert resp.status_code == 404
        assert resp.json()["detail"]["code"] == "missing_video"


def test_skills_endpoint_lists_builtins(tmp_path: Path) -> None:
    """GET /v1/skills returns the default and the built-in catalog."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        r = client.get("/v1/skills")
        assert r.status_code == 200
        body = r.json()
        assert body["default"] == "summary"
        names = {s["name"] for s in body["skills"]}
        assert {"summary", "bug_report", "tutorial"} <= names


def test_actions_endpoint_lists_builtins(tmp_path: Path) -> None:
    """GET /v1/actions returns the default, auto flag, and the built-in catalog."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        r = client.get("/v1/actions")
        assert r.status_code == 200
        body = r.json()
        assert body["default"] == "fix"
        assert body["auto"] is True
        names = {a["name"] for a in body["actions"]}
        assert {"fix", "explain", "triage", "test", "report", "reproduce"} <= names


def test_analyze_forwards_action_fields(tmp_path: Path) -> None:
    """action/action_prompt form fields are passed through to the orchestrator."""
    app = create_app(_make_settings(tmp_path))
    captured: dict[str, object] = {}

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        captured.update(kwargs)
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        resp = client.post(
            "/v1/analyze",
            files={"video": ("bug.mp4", b"vid", "video/mp4")},
            data={"action": "triage"},
        )
        assert resp.status_code == 202
        assert captured["action"] == "triage"
        assert captured["action_prompt"] is None


def test_analyze_forwards_skill_and_system_prompt(tmp_path: Path) -> None:
    """skill/system_prompt form fields are passed through to the orchestrator."""
    app = create_app(_make_settings(tmp_path))
    captured: dict[str, object] = {}

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        captured.update(kwargs)
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        resp = client.post(
            "/v1/analyze",
            files={"video": ("bug.mp4", b"vid", "video/mp4")},
            data={"skill": "tutorial", "intent": "explain it"},
        )
        assert resp.status_code == 202
        assert captured["skill"] == "tutorial"
        assert captured["user_intent"] == "explain it"
        assert captured["system_prompt"] is None


def test_cors_allows_any_chrome_extension_origin(tmp_path: Path) -> None:
    """A real (dynamic) extension origin must be allowed without prior config."""
    app = create_app(_make_settings(tmp_path))
    ext_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
    with TestClient(app) as client:
        # Preflight from the extension origin.
        preflight = client.options(
            "/v1/analyze",
            headers={
                "Origin": ext_origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.headers.get("access-control-allow-origin") == ext_origin

        # Actual request echoes the allowed origin too.
        health = client.get("/v1/healthz", headers={"Origin": ext_origin})
        assert health.headers.get("access-control-allow-origin") == ext_origin

    # A normal web origin is NOT allowed.
    with TestClient(app) as client:
        denied = client.get("/v1/healthz", headers={"Origin": "https://evil.example.com"})
        assert denied.headers.get("access-control-allow-origin") is None


def test_upload_limit_enforced(tmp_path: Path) -> None:
    """Oversized uploads should be rejected with 413."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        large = b"a" * (11 * 1024 * 1024)
        response = client.post(
            "/v1/analyze",
            files={"video": ("big.mp4", large, "video/mp4")},
        )
        assert response.status_code == 413
