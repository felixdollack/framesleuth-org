"""Tests for configuration loading and validation."""

import os

import pytest

from framesleuth.config import EngineProfile, Settings


class TestConfig:
    """Test configuration management."""

    def test_settings_default(self) -> None:
        """Test default settings load."""
        # Create a temporary Settings instance without reading .env
        settings = Settings(_env_file=None)
        assert settings.BACKEND_HOST == "127.0.0.1"
        assert settings.BACKEND_PORT == 8010
        assert settings.ENGINE_PROFILE == EngineProfile.LOCAL_DEFAULT

    def test_settings_engine_profile(self) -> None:
        """Test engine profile selection."""
        settings = Settings(
            ENGINE_PROFILE=EngineProfile.SERVER,
            _env_file=None,
        )
        assert settings.ENGINE_PROFILE == EngineProfile.SERVER

    def test_settings_validation_paths(self) -> None:
        """Test path creation on validation."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                BUNDLE_DIR=Path(tmpdir) / "bundles",
                DATABASE_PATH=Path(tmpdir) / "db" / "jobs.db",
                _env_file=None,
            )
            settings.validate_paths()
            assert settings.BUNDLE_DIR.exists()
            assert settings.DATABASE_PATH.parent.exists()

    def test_settings_rejects_unknown_vars(self) -> None:
        """Test that unknown env vars are rejected."""
        os.environ["FRAMESLEUTH_UNKNOWN_VAR"] = "test"
        try:
            # This should raise due to extra="forbid"
            with pytest.raises(ValueError):
                Settings(
                    UNKNOWN_VAR="test",  # type: ignore
                    _env_file=None,
                )
        finally:
            del os.environ["FRAMESLEUTH_UNKNOWN_VAR"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
