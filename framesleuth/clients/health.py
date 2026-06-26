"""Health checks for external dependencies.

Provides /healthz endpoint response contract and startup checks.
"""

import inspect
from typing import Any

import aiohttp
from pydantic import BaseModel

from framesleuth.logging_config import get_logger

logger = get_logger("clients.health")


async def _resolve_response(request: Any) -> Any:
    """Resolve aiohttp request object or coroutine into a response object."""
    if inspect.isawaitable(request):
        return await request
    return request


class ServiceHealth(BaseModel):
    """Health status of a single service."""

    name: str
    status: str  # "ready", "unavailable", "degraded"
    latency_ms: float | None = None
    error: str | None = None


class HealthStatus(BaseModel):
    """Overall health status."""

    status: str  # "healthy", "degraded", "unhealthy"
    services: dict[str, ServiceHealth]
    queue_depth: int = 0
    timestamp: str = ""


async def check_vlm_health(vlm_url: str, timeout_s: float = 5.0) -> ServiceHealth:
    """Check VLM (llama.cpp) health.

    Args:
        vlm_url: Base URL of VLM server.
        timeout_s: Timeout for health check.

    Returns:
        Health status of VLM.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            request = session.get(f"{vlm_url}/v1/models")
            resolved = await _resolve_response(request)
            if hasattr(resolved, "__aenter__"):
                async with resolved as resp:
                    if resp.status == 200:
                        return ServiceHealth(name="vlm", status="ready")
                    return ServiceHealth(
                        name="vlm", status="unavailable", error=f"HTTP {resp.status}"
                    )

            resp = resolved
            if resp.status == 200:
                return ServiceHealth(name="vlm", status="ready")
            return ServiceHealth(name="vlm", status="unavailable", error=f"HTTP {resp.status}")
    except Exception as e:
        logger.warning(f"VLM health check failed: {e}")
        return ServiceHealth(name="vlm", status="unavailable", error=str(e))


async def check_coder_health(coder_url: str, timeout_s: float = 5.0) -> ServiceHealth:
    """Check coder (Ollama/llama.cpp) health.

    Args:
        coder_url: Base URL of coder server.
        timeout_s: Timeout for health check.

    Returns:
        Health status of coder.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Try Ollama endpoint first
            ollama_request = session.get(f"{coder_url}/api/tags")
            ollama_resolved = await _resolve_response(ollama_request)
            if hasattr(ollama_resolved, "__aenter__"):
                async with ollama_resolved as resp:
                    if resp.status == 200:
                        return ServiceHealth(name="coder", status="ready")
            elif ollama_resolved.status == 200:
                return ServiceHealth(name="coder", status="ready")

            # Fall back to OpenAI-compatible endpoint
            openai_request = session.get(f"{coder_url}/v1/models")
            openai_resolved = await _resolve_response(openai_request)
            if hasattr(openai_resolved, "__aenter__"):
                async with openai_resolved as resp:
                    if resp.status == 200:
                        return ServiceHealth(name="coder", status="ready")
                    return ServiceHealth(
                        name="coder", status="unavailable", error=f"HTTP {resp.status}"
                    )

            resp = openai_resolved
            if resp.status == 200:
                return ServiceHealth(name="coder", status="ready")
            return ServiceHealth(name="coder", status="unavailable", error=f"HTTP {resp.status}")
    except Exception as e:
        logger.warning(f"Coder health check failed: {e}")
        return ServiceHealth(name="coder", status="unavailable", error=str(e))


async def check_storage_health(bundle_dir: str) -> ServiceHealth:
    """Check storage (filesystem) health.

    Args:
        bundle_dir: Path to bundle directory.

    Returns:
        Health status of storage.
    """
    try:
        from pathlib import Path

        path = Path(bundle_dir)
        # Try to write a temporary file
        test_file = path / ".health-check"
        test_file.write_text("ok")
        test_file.unlink()
        return ServiceHealth(name="storage", status="ready")
    except Exception as e:
        logger.warning(f"Storage health check failed: {e}")
        return ServiceHealth(name="storage", status="unavailable", error=str(e))


async def get_health_status(
    vlm_url: str, coder_url: str, bundle_dir: str, queue_depth: int = 0
) -> HealthStatus:
    """Get overall system health status.

    Args:
        vlm_url: VLM server URL.
        coder_url: Coder server URL.
        bundle_dir: Bundle storage directory.
        queue_depth: Current job queue depth.

    Returns:
        Overall health status with service details.
    """
    from datetime import UTC, datetime

    vlm_health = await check_vlm_health(vlm_url)
    coder_health = await check_coder_health(coder_url)
    storage_health = await check_storage_health(bundle_dir)

    services = {
        "vlm": vlm_health,
        "coder": coder_health,
        "storage": storage_health,
    }

    # Overall status: healthy if all ready, degraded if some unavailable, unhealthy if critical
    status_values = [s.status for s in services.values()]
    if all(s == "ready" for s in status_values):
        overall = "healthy"
    elif any(s == "unavailable" for s in status_values if s != "coder"):
        # VLM or storage down = unhealthy
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthStatus(
        status=overall,
        services=services,
        queue_depth=queue_depth,
        timestamp=datetime.now(UTC).isoformat(),
    )
