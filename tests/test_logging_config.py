"""Tests for structured logging configuration."""

import io
import json
import logging

from framesleuth.logging_config import (
    JSONFormatter,
    clear_context,
    get_logger,
    set_correlation_id,
    set_job_id,
    setup_logging,
)


class TestLoggingConfig:
    """Test logging module behavior."""

    def test_json_formatter_includes_context(self) -> None:
        """JSON formatter should include correlation context fields."""
        clear_context()
        set_job_id("job-123")
        set_correlation_id("corr-abc")

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="framesleuth.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        payload = json.loads(formatted)

        assert payload["message"] == "hello"
        assert payload["job_id"] == "job-123"
        assert payload["correlation_id"] == "corr-abc"

    def test_setup_logging_with_stream_handler(self) -> None:
        """setup_logging should attach a JSON formatter to handlers."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)

        logger = setup_logging(level="INFO", json_format=True, handlers=[handler])
        logger.info("structured log")

        output = stream.getvalue().strip()
        assert output
        parsed = json.loads(output)
        assert parsed["message"] == "structured log"
        assert parsed["level"] == "INFO"

    def test_get_logger_uses_framesleuth_namespace(self) -> None:
        """Module logger names should be rooted under framesleuth."""
        logger = get_logger("pipeline.test")
        assert logger.name == "framesleuth.pipeline.test"

    def test_set_correlation_id_generates_when_missing(self) -> None:
        """set_correlation_id should generate an ID when not provided."""
        clear_context()
        generated = set_correlation_id()
        assert isinstance(generated, str)
        assert len(generated) > 10
