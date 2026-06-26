"""Tests for VLM client with retry/backoff and response parsing."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from framesleuth.clients.vlm import FrameAnalysisResponse, VLMClient


class TestVLMClient:
    """Test VLM client functionality."""

    @pytest.fixture
    def vlm_client(self) -> VLMClient:
        """Create a test VLM client."""
        return VLMClient(
            base_url="http://127.0.0.1:8080",
            model="Qwen/Qwen3-VL-8B-Instruct-GGUF",
            timeout_s=10.0,
            max_retries=3,
        )

    @pytest.mark.asyncio
    async def test_analyze_frame_success(self, vlm_client: VLMClient) -> None:
        """Test successful frame analysis."""
        # Create a dummy image file
        import base64
        import tempfile

        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            image_path = f.name

        try:
            with patch("aiohttp.ClientSession") as mock_session:
                # Mock successful response
                mock_resp = AsyncMock()
                mock_resp.status = 200
                mock_resp.text = AsyncMock(
                    return_value=json.dumps(
                        {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            {
                                                "caption": "Error dialog visible",
                                                "ocr_text": "TypeError: cannot read property",
                                                "ui_action": None,
                                                "is_error_state": True,
                                                "reason": "Exception dialog",
                                            }
                                        )
                                    }
                                }
                            ]
                        }
                    )
                )
                session = mock_session.return_value
                session.post.return_value.__aenter__.return_value = mock_resp

                response = await vlm_client.analyze_frame(image_path, 5.0)
                assert isinstance(response, FrameAnalysisResponse)
                assert response.is_error_state is True
                assert "TypeError" in response.ocr_text
        finally:
            import os

            os.unlink(image_path)

    @pytest.mark.asyncio
    async def test_analyze_frame_retry_on_503(self, vlm_client: VLMClient) -> None:
        """Test retry logic on 503 (service overloaded)."""
        import base64
        import tempfile

        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            image_path = f.name

        try:
            with patch("aiohttp.ClientSession") as mock_session:
                # First call: 503, second call: 200
                mock_resp_503 = AsyncMock()
                mock_resp_503.status = 503

                mock_resp_200 = AsyncMock()
                mock_resp_200.status = 200
                mock_resp_200.text = AsyncMock(
                    return_value=json.dumps(
                        {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            {
                                                "caption": "OK",
                                                "ocr_text": "",
                                                "ui_action": None,
                                                "is_error_state": False,
                                                "reason": None,
                                            }
                                        )
                                    }
                                }
                            ]
                        }
                    )
                )

                mock_req_503 = AsyncMock()
                mock_req_503.__aenter__.return_value = mock_resp_503
                mock_req_503.__aexit__.return_value = False

                mock_req_200 = AsyncMock()
                mock_req_200.__aenter__.return_value = mock_resp_200
                mock_req_200.__aexit__.return_value = False

                session = mock_session.return_value
                session.post.side_effect = [mock_req_503, mock_req_200]

                # Should retry and succeed
                response = await vlm_client.analyze_frame(image_path, 5.0)
                assert response.is_error_state is False
        finally:
            import os

            os.unlink(image_path)

    @pytest.mark.asyncio
    async def test_analyze_frame_does_not_retry_on_4xx(self, vlm_client: VLMClient) -> None:
        """A 4xx client error is non-retriable and surfaces immediately (no second call)."""
        import base64
        import os
        import tempfile

        from framesleuth.errors import ModelUnavailableError

        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            image_path = f.name

        try:
            with patch("aiohttp.ClientSession") as mock_session:
                mock_resp = AsyncMock()
                mock_resp.status = 400
                mock_resp.text = AsyncMock(return_value="bad request: unknown field")

                mock_req = AsyncMock()
                mock_req.__aenter__.return_value = mock_resp
                mock_req.__aexit__.return_value = False

                session = mock_session.return_value
                session.post.side_effect = [mock_req]  # only ONE response available

                with pytest.raises(ModelUnavailableError):
                    await vlm_client.analyze_frame(image_path, 5.0)
                # No retry: the client must not have asked for a second response.
                assert session.post.call_count == 1
        finally:
            os.unlink(image_path)

    @pytest.mark.asyncio
    async def test_parse_response_with_markdown(self, vlm_client: VLMClient) -> None:
        """Test parsing JSON response wrapped in markdown code blocks."""
        inner = json.dumps(
            {
                "caption": "Test",
                "ocr_text": "",
                "ui_action": None,
                "is_error_state": False,
                "reason": None,
            }
        )
        response_text = json.dumps(
            {"choices": [{"message": {"content": f"```json\n{inner}\n```"}}]}
        )

        parsed = vlm_client._parse_response(response_text)
        assert parsed.caption == "Test"
        assert parsed.is_error_state is False


class TestVLMClientHealth:
    """Test VLM client health check."""

    @pytest.mark.asyncio
    async def test_warmup_success(self) -> None:
        """Test successful warmup."""
        client = VLMClient(
            base_url="http://127.0.0.1:8080",
            model="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        )

        with patch.object(client, "analyze_frame", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = FrameAnalysisResponse(
                caption="Test", ocr_text="", is_error_state=False
            )
            result = await client.warmup()
            assert result is True
            mock_analyze.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
