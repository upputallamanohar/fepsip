"""
FEPSIP Logging Configuration
Uses loguru for structured, colorful logs.
"""
from __future__ import annotations

import sys
from loguru import logger


def setup_logger(level: str = "INFO", log_file: str | None = "logs/fepsip.log") -> None:
    """Configure loguru logger for the application."""
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler
    if log_file:
        import os
        os.makedirs("logs", exist_ok=True)
        logger.add(
            log_file,
            level=level,
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        )

    logger.info("FEPSIP Logger initialized at level={}", level)


# Default logger export
__all__ = ["logger", "setup_logger"]
