"""OpenAI-compatible VLM client with retry, backoff, and structured output.

Follows the interface segregation principle: single focused responsibility.
Engine-agnostic HTTP abstraction per ADR-001.
"""

import asyncio
import base64
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
from pydantic import BaseModel, ValidationError

from framesleuth.errors import ModelUnavailableError
from framesleuth.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from framesleuth.config import Settings

logger = get_logger("clients.vlm")

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


async def _resolve_response(request: Any) -> Any:
    """Resolve aiohttp request object or coroutine into a response object.

    Supports both real aiohttp request context managers and AsyncMock-based
    test doubles that return awaitable responses directly.
    """
    if inspect.isawaitable(request):
        return await request
    return request


def _encode_image(image_path: str, *, send_jpeg: bool, jpeg_quality: int) -> tuple[str, str]:
    """Read an image and return ``(base64, mime_type)`` for an image_url payload.

    Transcodes to JPEG (smaller upload + fewer vision prefill tokens) when
    ``send_jpeg`` is set and OpenCV can decode the frame; otherwise the original
    bytes are sent unchanged. Never raises on a bad/opaque file — it falls back
    to the raw bytes so a single odd frame can't abort analysis.
    """
    raw = Path(image_path).read_bytes()
    if send_jpeg:
        try:
            import cv2  # lazy: heavy media dep, only in the full stack
            import numpy as np

            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if arr is not None:
                ok, buf = cv2.imencode(
                    ".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
                )
                if ok:
                    return base64.b64encode(buf.tobytes()).decode("utf-8"), "image/jpeg"
        except Exception as exc:  # pragma: no cover - exercised only without cv2/bad image
            logger.debug("JPEG transcode skipped for %s: %s", image_path, exc)
    mime = (
        "image/png"
        if raw[:8] == _PNG_MAGIC or image_path.lower().endswith(".png")
        else "image/jpeg"
    )
    return base64.b64encode(raw).decode("utf-8"), mime


class FrameAnalysisResponse(BaseModel):
    """Response from VLM frame analysis."""

    caption: str
    ocr_text: str
    ui_action: str | None = None
    is_error_state: bool = False
    reason: str | None = None


class VLMClient:
    """OpenAI-compatible VLM (Qwen3-VL) client.

    Handles:
    - Retry with exponential backoff
    - JSON parsing with fallback
    - Timeout and circuit-breaker guards
    - Structured output (JSON schema per engine)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        *,
        max_tokens: int = 768,
        json_mode: bool = True,
        send_jpeg: bool = True,
        jpeg_quality: int = 85,
    ) -> None:
        """Initialize VLM client.

        Args:
            base_url: OpenAI-compatible API base URL (e.g., http://127.0.0.1:8080).
            model: Model identifier (e.g., Qwen/Qwen3-VL-8B-Instruct-GGUF).
            timeout_s: Request timeout in seconds.
            max_retries: Maximum number of retries on 5xx/timeout.
            max_tokens: Default generation cap per frame.
            json_mode: Request OpenAI-style ``response_format`` JSON output.
            send_jpeg: Transcode frames to JPEG before upload.
            jpeg_quality: JPEG quality (1-100) when ``send_jpeg`` is set.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.send_jpeg = send_jpeg
        self.jpeg_quality = jpeg_quality
        self._session: aiohttp.ClientSession | None = None

    @classmethod
    def from_settings(cls, settings: "Settings") -> "VLMClient":
        """Build a client from app settings.

        The single construction path for both the HTTP service and the MCP
        server, so VLM tuning (timeouts, token caps, JSON mode, JPEG) stays
        consistent wherever a frame is analyzed.
        """
        return cls(
            settings.VLM_URL,
            settings.VLM_MODEL,
            timeout_s=settings.VLM_TIMEOUT_S,
            max_retries=settings.VLM_MAX_RETRIES,
            max_tokens=settings.VLM_MAX_TOKENS,
            json_mode=settings.VLM_JSON_MODE,
            send_jpeg=settings.VLM_SEND_JPEG,
            jpeg_quality=settings.VLM_JPEG_QUALITY,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a shared, reused HTTP session (created lazily).

        Reusing one session keeps the TCP/TLS connection pool warm across the
        many per-frame calls instead of paying connection setup on each request.
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def aclose(self) -> None:
        """Close the shared HTTP session, if open."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "VLMClient":
        """Enter an async context that closes the session on exit."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Close the shared session when leaving the context."""
        await self.aclose()

    async def analyze_frame(
        self,
        image_path: str,
        timestamp: float,
        prompt_override: str | None = None,
        *,
        max_tokens: int | None = None,
        send_jpeg: bool | None = None,
    ) -> FrameAnalysisResponse:
        """Analyze a single frame using the VLM.

        Args:
            image_path: Path to the frame image.
            timestamp: Timestamp of the frame in seconds (for context).
            prompt_override: Custom prompt; if None, uses default.
            max_tokens: Per-call generation cap (defaults to ``self.max_tokens``).
            send_jpeg: Per-call JPEG override (defaults to ``self.send_jpeg``). Pass
                ``False`` to send the frame uncompressed — used for error re-OCR so
                tiny error text is never degraded by lossy compression.

        Returns:
            Parsed frame analysis response.

        Raises:
            ModelUnavailableError: If VLM is unreachable or fails.
            ValidationError: If response doesn't match schema.
        """
        from framesleuth.prompts import VLMPrompts

        if prompt_override is None:
            prompt = VLMPrompts.frame_analysis(timestamp)
        else:
            prompt = prompt_override

        use_jpeg = self.send_jpeg if send_jpeg is None else send_jpeg
        image_data, mime = _encode_image(
            image_path, send_jpeg=use_jpeg, jpeg_quality=self.jpeg_quality
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_data}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,  # Deterministic OCR/structured output.
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await self._call_with_retry(payload)
        return self._parse_response(response)

    async def _call_with_retry(self, payload: dict[str, Any]) -> str:
        """Call the VLM with exponential backoff retry.

        Args:
            payload: Request payload.

        Returns:
            Raw response text.

        Raises:
            ModelUnavailableError: If all retries exhausted.
        """
        url = f"{self.base_url}/v1/chat/completions"

        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()
                resolved = await _resolve_response(session.post(url, json=payload))
                if hasattr(resolved, "__aenter__"):
                    async with resolved as resp:
                        text = await self._handle_status(resp, attempt)
                else:
                    text = await self._handle_status(resolved, attempt)
                if text is not None:
                    return text
                await asyncio.sleep(2**attempt)
            except TimeoutError:
                logger.warning(f"VLM timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise ModelUnavailableError("VLM", self.base_url) from None
            except ModelUnavailableError:
                raise
            except Exception as e:
                logger.error(f"VLM request failed: {e}")
                raise ModelUnavailableError("VLM", self.base_url) from e

        raise ModelUnavailableError("VLM", self.base_url)

    async def _handle_status(self, resp: Any, attempt: int) -> str | None:
        """Map an HTTP response to text, a retry signal, or a fatal error.

        Args:
            resp: Response object exposing ``status`` and ``text()``.
            attempt: Zero-based attempt index for retry accounting.

        Returns:
            Response text on success, or ``None`` to signal a retriable 503.

        Raises:
            ModelUnavailableError: On non-retriable or exhausted failures.
        """
        if resp.status == 200:
            text: str = await resp.text()
            logger.info(f"VLM call successful (attempt {attempt + 1})")
            return text
        # Transient server-side failures (overload, gateway, rate limit) are worth
        # retrying; 4xx client errors (bad payload, model not found) are not — they
        # would fail identically on every attempt, so surface them immediately.
        retriable = resp.status in (429, 500, 502, 503, 504)
        if retriable and attempt < self.max_retries - 1:
            logger.warning(
                f"VLM returned {resp.status} (attempt {attempt + 1}/{self.max_retries})"
            )
            return None
        if not retriable:
            body = (await resp.text())[:500]
            logger.error(f"VLM returned non-retriable {resp.status}: {body}")
        raise ModelUnavailableError("VLM", self.base_url)

    def _parse_response(self, response_text: str) -> FrameAnalysisResponse:
        """Parse and validate VLM response.

        Args:
            response_text: Raw response from VLM.

        Returns:
            Validated FrameAnalysisResponse.

        Raises:
            ValueError: If response cannot be parsed or doesn't match schema.
        """
        try:
            response_data = json.loads(response_text)
            # Extract JSON from choices[0].message.content
            if "choices" in response_data:
                content = response_data["choices"][0]["message"]["content"]
            else:
                content = response_text

            # Try to parse as JSON if it's a string
            if isinstance(content, str):
                # Extract JSON object from potential markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                content = json.loads(content)

            # Validate against schema
            parsed = FrameAnalysisResponse(**content)
            logger.debug(
                f"VLM response parsed: error_state={parsed.is_error_state}, "
                f"caption_len={len(parsed.caption)}"
            )
            return parsed

        except (json.JSONDecodeError, KeyError, ValidationError) as e:
            logger.error(f"Failed to parse VLM response: {e}")
            raise ValueError(f"VLM response validation failed: {e}") from e

    async def warmup(self) -> bool:
        """Warm up the VLM by calling it with a dummy request.

        Returns:
            True if warmup successful, False otherwise.
        """
        import tempfile

        logger.info("Warming up VLM...")
        # A dummy 1x1 PNG written to a private temp file (never a shared, fixed
        # /tmp path that could collide or linger between runs).
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            handle.write(png_data)
            dummy_frame = handle.name
        try:
            await self.analyze_frame(dummy_frame, 0.0)
            logger.info("VLM warmup complete")
            return True
        except Exception as e:
            logger.warning(f"VLM warmup failed: {e}")
            return False
        finally:
            Path(dummy_frame).unlink(missing_ok=True)
