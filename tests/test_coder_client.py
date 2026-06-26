"""Tests for coder client behavior and parsing."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from framesleuth.clients.coder import CoderClient, CoderResponse


class TestCoderClient:
    """Test coder model client."""

    @pytest.fixture
    def coder_client(self) -> CoderClient:
        """Create a test coder client."""
        return CoderClient(
            base_url="http://127.0.0.1:11434",
            model="qwen2.5-coder:7b",
            timeout_s=10.0,
            max_retries=3,
        )

    @pytest.mark.asyncio
    async def test_fix_bug_success(self, coder_client: CoderClient) -> None:
        """Test successful bug-fix response parsing."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.text = AsyncMock(
                return_value=json.dumps(
                    {
                        "choices": [
                            {"message": {"content": "Reasoning text\n```python\nprint('fix')\n```"}}
                        ]
                    }
                )
            )

            mock_request = AsyncMock()
            mock_request.__aenter__.return_value = mock_resp
            mock_request.__aexit__.return_value = False

            session = mock_session.return_value
            session.post.return_value = mock_request

            result = await coder_client.fix_bug(
                title="Button click crashes",
                severity="high",
                component="ui/button",
                environment={"os": "macOS", "browser": "Chrome"},
                repro_steps=[{"n": 1, "action": "Click Save"}],
                expected="Save succeeds",
                actual="App crashes",
                errors=["TypeError: undefined is not a function"],
                candidates=[{"path": "src/ui/button.ts", "why": "click handler"}],
            )

            assert isinstance(result, CoderResponse)
            assert "Reasoning" in result.reasoning
            assert "print('fix')" in result.code_changes

    @pytest.mark.asyncio
    async def test_warmup_success(self, coder_client: CoderClient) -> None:
        """Test warmup delegates to retry call."""
        with patch.object(coder_client, "_call_with_retry", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"choices":[{"message":{"content":"Ready"}}]}'
            ok = await coder_client.warmup()
            assert ok is True
            mock_call.assert_called_once()

    def test_parse_response_without_code_block(self, coder_client: CoderClient) -> None:
        """Test parsing plain text response."""
        response_text = json.dumps(
            {"choices": [{"message": {"content": "No code block, explanation only."}}]}
        )
        parsed = coder_client._parse_response(response_text)
        assert parsed.code_changes == ""
        assert "explanation only" in parsed.explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
