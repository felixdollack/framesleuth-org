"""OpenAI-compatible coder model client with engine-agnostic abstraction.

Supports Ollama, llama.cpp, and vLLM via OpenAI API.
Per ADR-001: engine selection is configuration, not code.
"""

import asyncio
import inspect
import json
from typing import Any

import aiohttp
from pydantic import BaseModel

from framesleuth.errors import ModelUnavailableError
from framesleuth.logging_config import get_logger

logger = get_logger("clients.coder")


async def _resolve_response(request: Any) -> Any:
    """Resolve aiohttp request object or coroutine into a response object."""
    if inspect.isawaitable(request):
        return await request
    return request


class CoderResponse(BaseModel):
    """Response from coder model."""

    reasoning: str
    code_changes: str
    explanation: str
    regressions: str | None = None
    next_steps: str | None = None


class CoderClient:
    """OpenAI-compatible coder client (Qwen2.5-Coder).

    Handles:
    - Engine-agnostic OpenAI-compatible HTTP
    - Retry with backoff
    - Structured output
    - Context window management
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        """Initialize coder client.

        Args:
            base_url: OpenAI-compatible API base URL.
            model: Model identifier.
            timeout_s: Request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a shared, reused HTTP session (created lazily)."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def aclose(self) -> None:
        """Close the shared HTTP session, if open."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "CoderClient":
        """Enter an async context that closes the session on exit."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Close the shared session when leaving the context."""
        await self.aclose()

    async def fix_bug(
        self,
        title: str,
        severity: str,
        component: str,
        environment: dict[str, str],
        repro_steps: list[dict[str, Any]],
        expected: str,
        actual: str,
        errors: list[str],
        candidates: list[dict[str, Any]],
        keyframe_path: str | None = None,
    ) -> CoderResponse:
        """Generate a fix for the reported bug.

        Args:
            title: Bug title.
            severity: Severity level.
            component: Suspected component.
            environment: Environment info (OS, app, version, browser).
            repro_steps: Numbered reproduction steps.
            expected: Expected behavior.
            actual: Actual behavior.
            errors: Error messages and stack traces.
            candidates: Code candidates from grounding.
            keyframe_path: Path to failure keyframe (for context).

        Returns:
            Parsed coder response with fix and explanation.

        Raises:
            ModelUnavailableError: If coder is unavailable.
        """
        from framesleuth.prompts import FixPrompts

        prompt = FixPrompts.fix_from_video(
            title=title,
            severity=severity,
            component=component,
            environment=environment,
            repro_steps=repro_steps,
            expected=expected,
            actual=actual,
            errors=errors,
            candidates=candidates,
            keyframe_path=keyframe_path,
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        response = await self._call_with_retry(payload)
        return self._parse_response(response)

    async def complete(
        self,
        system_prompt: str,
        user_content: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        """Generic text completion: run a system+user chat turn, return the text.

        Used for synthesis tasks (e.g. recording summaries) where the caller
        supplies the full system prompt. Raises ``ModelUnavailableError`` if the
        model is unreachable or the response can't be parsed, so callers can
        degrade gracefully.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        raw = await self._call_with_retry(payload)
        try:
            data = json.loads(raw)
            return str(data["choices"][0]["message"]["content"]).strip()
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.error(f"Failed to parse completion response: {exc}")
            raise ModelUnavailableError("Coder", self.base_url) from exc

    async def _call_with_retry(self, payload: dict[str, Any]) -> str:
        """Call coder with exponential backoff retry.

        Args:
            payload: Request payload.

        Returns:
            Raw response text.

        Raises:
            ModelUnavailableError: If retries exhausted.
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
                logger.warning(f"Coder timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise ModelUnavailableError("Coder", self.base_url) from None
            except ModelUnavailableError:
                raise
            except Exception as e:
                logger.error(f"Coder request failed: {e}")
                raise ModelUnavailableError("Coder", self.base_url) from e

        raise ModelUnavailableError("Coder", self.base_url)

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
            logger.info(f"Coder call successful (attempt {attempt + 1})")
            return text
        if resp.status == 503 and attempt < self.max_retries - 1:
            logger.warning(f"Coder returned 503 (attempt {attempt + 1}/{self.max_retries})")
            return None
        raise ModelUnavailableError("Coder", self.base_url)

    def _parse_response(self, response_text: str) -> CoderResponse:
        """Parse coder response.

        Args:
            response_text: Raw response.

        Returns:
            Parsed coder response.

        Raises:
            ModelUnavailableError: If parsing fails.
        """
        try:
            response_data = json.loads(response_text)
            content = response_data["choices"][0]["message"]["content"]

            # Extract code block if present
            if "```" in content:
                parts = content.split("```")
                code_section = parts[1] if len(parts) > 1 else ""
                explanation = content.split("```")[0]
            else:
                code_section = ""
                explanation = content

            return CoderResponse(
                reasoning=explanation[:500],  # First 500 chars
                code_changes=code_section,
                explanation=explanation,
            )
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse coder response: {e}")
            raise ModelUnavailableError("Coder", self.base_url) from e

    async def warmup(self) -> bool:
        """Warm up coder by calling with a small prompt.

        Returns:
            True if warmup successful.
        """
        try:
            logger.info("Warming up coder...")
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Say 'Ready'."}],
                "temperature": 0.5,
                "max_tokens": 10,
            }
            await self._call_with_retry(payload)
            logger.info("Coder warmup complete")
            return True
        except Exception as e:
            logger.warning(f"Coder warmup failed: {e}")
            return False
