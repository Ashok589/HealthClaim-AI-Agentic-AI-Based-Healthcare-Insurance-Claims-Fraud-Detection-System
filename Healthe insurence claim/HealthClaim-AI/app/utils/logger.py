"""
app/utils/logger.py
Structured logging factory with coloured console output and rotating file handler.
Usage:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Hello %s", "world")
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI colour codes (no external dependency required)
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}


class _ColourFormatter(logging.Formatter):
    """Formatter that prepends a coloured level-name to each log record."""

    _FMT = "%(asctime)s | {colour}%(levelname)-8s{reset} | %(name)s | %(message)s"
    _DATEFMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = _COLOURS.get(record.levelname, "")
        fmt = self._FMT.format(colour=colour, reset=_RESET)
        formatter = logging.Formatter(fmt, datefmt=self._DATEFMT)
        return formatter.format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI codes)."""

    _FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT)


# ---------------------------------------------------------------------------
# Module-level defaults (can be overridden by env vars before first import)
# ---------------------------------------------------------------------------
_LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
_LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_STR, logging.INFO)
_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
_BACKUP_COUNT = 5                # keep 5 rotated files

# Registry so we never configure the same logger twice
_configured: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for *name*.

    The logger gets:
    - A coloured StreamHandler writing to stderr.
    - A RotatingFileHandler writing to logs/<name>.log.

    Subsequent calls with the same *name* return the already-configured logger.
    """
    logger = logging.getLogger(name)

    if name in _configured:
        return logger

    logger.setLevel(_LOG_LEVEL)
    logger.propagate = False   # avoid double-printing via root logger

    # ------------------------------------------------------------------ #
    # Console handler
    # ------------------------------------------------------------------ #
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(_LOG_LEVEL)
    console_handler.setFormatter(_ColourFormatter())
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------ #
    # File handler
    # ------------------------------------------------------------------ #
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        # Replace dots in logger name with underscores for a valid filename
        safe_name = name.replace(".", "_")
        log_file = _LOG_DIR / f"{safe_name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(_LOG_LEVEL)
        file_handler.setFormatter(_PlainFormatter())
        logger.addHandler(file_handler)
    except OSError as exc:
        # Non-fatal — log to console only if we cannot create the file
        logger.warning("Could not create log file: %s", exc)

    _configured.add(name)
    return logger


# ---------------------------------------------------------------------------
# Root application logger convenience instance
# ---------------------------------------------------------------------------
root_logger = get_logger("healthclaim")
