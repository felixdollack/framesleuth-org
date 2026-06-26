"""Framesleuth: Local bug video analysis for VS Code and Chrome."""

__version__ = "0.1.0"
__author__ = "Framesleuth Contributors"

from framesleuth.config import EngineProfile, Settings, get_settings
from framesleuth.errors import (
    DurationExceededError,
    FramesleutheException,
    JobNotFoundError,
    JobTimeoutError,
    LowEvidenceWarning,
    ModelUnavailableError,
    PreprocessingFailedError,
    UnsupportedMediaError,
    UploadTooLargeError,
)
from framesleuth.logging_config import get_logger, set_correlation_id, set_job_id

__all__ = [
    "DurationExceededError",
    "EngineProfile",
    "FramesleutheException",
    "JobNotFoundError",
    "JobTimeoutError",
    "LowEvidenceWarning",
    "ModelUnavailableError",
    "PreprocessingFailedError",
    "Settings",
    "UnsupportedMediaError",
    "UploadTooLargeError",
    "get_logger",
    "get_settings",
    "set_correlation_id",
    "set_job_id",
]
