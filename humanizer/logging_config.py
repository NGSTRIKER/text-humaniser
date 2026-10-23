"""Logging configuration for the rewriter pipeline.

Sets up a ``rich``-powered console handler and an optional file handler.
Call :func:`setup_logging` once at startup.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure the root ``humanizer`` logger.

    Parameters:
        level: Python log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: If given, also write log records to this file.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    logger = logging.getLogger("humanizer")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Rich console handler — human-friendly output.
    console = Console(stderr=True, width=120)
    console_handler = RichHandler(
        console=console,
        show_path=False,
        show_time=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(console_handler)

    # Optional file handler — machine-parseable.
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
        )
        logger.addHandler(file_handler)

    # Suppress noisy third-party loggers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``humanizer`` namespace."""
    return logging.getLogger(f"humanizer.{name}")
