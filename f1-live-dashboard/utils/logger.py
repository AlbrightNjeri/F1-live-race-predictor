"""
utils/logger.py — Structured logging for F1 Live Predictor Dashboard
"""

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def get_logger(name: str = "f1_dashboard") -> logging.Logger:
    """Return a named logger, configuring root logger on first call."""
    global _configured
    if not _configured:
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FMT))

        # File handler — rotates so the file doesn't grow forever
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "f1_dashboard.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FMT))

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(console)
        root.addHandler(file_handler)
        _configured = True

    return logging.getLogger(name)
