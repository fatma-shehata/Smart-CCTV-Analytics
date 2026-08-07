"""
logger.py
Unified logging setup used across the whole project.
"""

import logging


def setup_logger(name: str = "smart_cctv") -> logging.Logger:
    """Returns a configured logger. Safe to call multiple times from
    different files without duplicating log output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger