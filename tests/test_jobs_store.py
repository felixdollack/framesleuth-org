"""Tests for SQLite job store and idempotency behavior."""

from pathlib import Path

import pytest

from framesleuth.jobs.store import JobStore
from framesleuth.schemas import JobState


@pytest.mark.asyncio
async def test_job_store_create_get_and_hash_lookup(tmp_path: Path) -> None:
    """Store should persist, fetch, and find jobs by content hash."""
    db_path = tmp_path / "jobs.db"
    store = JobStore(db_path)
    await store.initialize()

    await store.create_job("job-1", "hash-abc", "video.mp4")

    by_id = await store.get_job("job-1")
    by_hash = await store.find_by_content_hash("hash-abc")

    assert by_id is not None
    assert by_hash is not None
    assert by_id.id == by_hash.id == "job-1"
    assert by_id.state == JobState.QUEUED


@pytest.mark.asyncio
async def test_job_store_update_transitions(tmp_path: Path) -> None:
    """Store updates should persist state/progress and optional bundle path."""
    db_path = tmp_path / "jobs.db"
    store = JobStore(db_path)
    await store.initialize()

    await store.create_job("job-2", "hash-def", "video.mp4")
    await store.update_job(
        "job-2", state=JobState.DONE, progress_pct=100, bundle_path="bundle.json"
    )

    job = await store.get_job("job-2")
    assert job is not None
    assert job.state == JobState.DONE
    assert job.progress_pct == 100
    assert job.bundle_path == "bundle.json"
