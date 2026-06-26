"""Structured logging for Framesleuth.

Provides JSON-formatted logging with job_id correlation for observability.
Follows the single responsibility principle with a dedicated logging module.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


class ContextVar:
    """Thread-safe context variable for correlation IDs."""

    def __init__(self) -> None:
        """Initialize context storage."""
        self._context: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Set a context value."""
        self._context[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._context.get(key, default)

    def clear(self) -> None:
        """Clear all context."""
        self._context.clear()


# Global context for correlation IDs
_context = ContextVar()


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured output and correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string with all relevant fields.
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add job_id if available
        job_id = _context.get("job_id")
        if job_id:
            log_data["job_id"] = job_id

        # Add correlation ID
        correlation_id = _context.get("correlation_id")
        if correlation_id:
            log_data["correlation_id"] = correlation_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add custom fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    handlers: list[logging.Handler] | None = None,
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, use JSON formatting; else use plain text.
        handlers: Optional list of handlers; defaults to stderr.

    Returns:
        The root logger instance.
    """
    root_logger = logging.getLogger("framesleuth")
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    if handlers is None:
        handlers = [logging.StreamHandler(sys.stderr)]

    for handler in handlers:
        formatter: logging.Formatter
        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__).

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(f"framesleuth.{name}")


def set_job_id(job_id: str) -> None:
    """Set the current job ID for correlation.

    Args:
        job_id: The job identifier.
    """
    _context.set("job_id", job_id)


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Set or generate a correlation ID.

    Args:
        correlation_id: Optional correlation ID; generates UUID if not provided.

    Returns:
        The correlation ID being used.
    """
    if correlation_id is None:
        correlation_id = str(uuid4())
    _context.set("correlation_id", correlation_id)
    return correlation_id


def clear_context() -> None:
    """Clear all context variables."""
    _context.clear()
