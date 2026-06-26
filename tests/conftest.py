"""Test infrastructure and fixtures."""

import pytest


@pytest.fixture
def settings():
    """Provide test configuration."""
    from framesleuth.config import Settings

    return Settings(
        BACKEND_PORT=8001,
        LOG_LEVEL="DEBUG",
        MAX_CONCURRENT_JOBS=1,
    )
