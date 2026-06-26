"""Tests for health check functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from framesleuth.clients.health import (
    check_coder_health,
    check_storage_health,
    check_vlm_health,
    get_health_status,
)


class TestHealthChecks:
    """Test health check functions."""

    @pytest.mark.asyncio
    async def test_vlm_health_ready(self) -> None:
        """Test VLM health check when ready."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            session = mock_session.return_value.__aenter__.return_value
            session.get.return_value.__aenter__.return_value = mock_resp

            health = await check_vlm_health("http://127.0.0.1:8080")
            assert health.status == "ready"
            assert health.name == "vlm"

    @pytest.mark.asyncio
    async def test_vlm_health_unavailable(self) -> None:
        """Test VLM health check when unavailable."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 500
            session = mock_session.return_value.__aenter__.return_value
            session.get.return_value.__aenter__.return_value = mock_resp

            health = await check_vlm_health("http://127.0.0.1:8080")
            assert health.status == "unavailable"

    @pytest.mark.asyncio
    async def test_coder_health_ready(self) -> None:
        """Test coder health check when ready."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            session = mock_session.return_value.__aenter__.return_value
            session.get.return_value.__aenter__.return_value = mock_resp

            health = await check_coder_health("http://127.0.0.1:11434")
            assert health.status == "ready"

    @pytest.mark.asyncio
    async def test_storage_health(self) -> None:
        """Test storage health check."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            health = await check_storage_health(tmpdir)
            assert health.status == "ready"
            assert health.name == "storage"

    @pytest.mark.asyncio
    async def test_overall_health_status(self) -> None:
        """Test overall health status aggregation."""
        with (
            patch("framesleuth.clients.health.check_vlm_health") as mock_vlm,
            patch("framesleuth.clients.health.check_coder_health") as mock_coder,
            patch("framesleuth.clients.health.check_storage_health") as mock_storage,
        ):
            from framesleuth.clients.health import ServiceHealth

            mock_vlm.return_value = ServiceHealth(name="vlm", status="ready")
            mock_coder.return_value = ServiceHealth(name="coder", status="ready")
            mock_storage.return_value = ServiceHealth(name="storage", status="ready")

            status = await get_health_status(
                vlm_url="http://127.0.0.1:8080",
                coder_url="http://127.0.0.1:11434",
                bundle_dir="/tmp",
                queue_depth=0,
            )

            assert status.status == "healthy"
            assert status.queue_depth == 0
            assert len(status.services) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
