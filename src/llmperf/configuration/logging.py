"""Shared logging configuration for LLMPerf processes and command-line tools."""

import logging
import os
import sys
from typing import Optional, TextIO


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
LOG_LEVELS = ("critical", "error", "warning", "info", "debug")
LOG_COLOR_MODES = ("auto", "always", "never")
RESET = "\033[0m"
LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}
LOGGER_ALIASES = {
    # Uvicorn uses "uvicorn.error" for all lifecycle messages, including INFO.
    # A neutral name prevents the logger name from looking like the record level.
    "uvicorn.error": "llmperf_backend.server",
    "uvicorn.access": "llmperf_backend.access",
}


class LoggerNameFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.name = LOGGER_ALIASES.get(record.name, record.name)
        return True


class ColorFormatter(logging.Formatter):
    """Highlight the level name without coloring machine-readable messages."""

    def __init__(self, *, color: bool):
        super().__init__(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        color = LEVEL_COLORS.get(record.levelno) if self.color else None
        if not color:
            return rendered
        level = record.levelname
        return rendered.replace(f" {level} ", f" {color}{level}{RESET} ", 1)


def normalize_log_level(level: str) -> int:
    """Map public log-level names onto standard-library logging levels."""

    normalized = level.strip().lower()
    if normalized == "trace":
        return logging.DEBUG
    if normalized not in LOG_LEVELS:
        raise ValueError(f"Unsupported log level: {level}")
    return getattr(logging, normalized.upper())


def configure_logging(
    level: str = "info",
    *,
    color: str = "auto",
    stream: Optional[TextIO] = None,
    force: bool = True,
) -> None:
    """Configure one consistent stderr log stream for the current process."""

    if color not in LOG_COLOR_MODES:
        raise ValueError(f"Unsupported log color mode: {color}")
    output = stream or sys.stderr
    use_color = color == "always" or (
        color == "auto"
        and "NO_COLOR" not in os.environ
        and bool(getattr(output, "isatty", lambda: False)())
    )
    logging.basicConfig(
        level=normalize_log_level(level),
        stream=output,
        force=force,
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(ColorFormatter(color=use_color))
        handler.addFilter(LoggerNameFilter())

    # Hugging Face logs every retry twice at WARNING. At normal verbosity the
    # LLMPerf tokenizer resolver emits one final actionable ERROR instead.
    retry_logger = logging.getLogger("huggingface_hub.utils._http")
    retry_logger.setLevel(
        logging.DEBUG if normalize_log_level(level) <= logging.DEBUG else logging.ERROR
    )


def route_library_logs(*logger_names: str) -> None:
    """Route third-party private handlers through the configured root logger."""

    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
