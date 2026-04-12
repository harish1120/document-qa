import logging
import sys


def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger instance."""

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
