"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.utils.config import config


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Default format
    if format_string is None:
        format_string = config.get(
            "logging.format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default application logger
logger = setup_logger(
    "churn_predictor",
    log_file=config.get("logging.file", "logs/churn_predictor.log"),
    level=config.get("logging.level", "INFO"),
)
