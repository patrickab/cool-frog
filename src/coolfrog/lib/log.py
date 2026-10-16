"""Package logging: Rich console output, optional plain log file, swappable handler for TUIs."""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

PACKAGE_LOGGER_NAME = "coolfrog"
FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger, use as `log = get_logger(__name__)`."""
    return logging.getLogger(name)


def setup_logging(
    level: int | str = "INFO",
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure the `coolfrog` logger, safe to call repeatedly.

    - Rich auto-disables colors when not on a TTY (pipes, CI)
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    package_logger.setLevel(level)
    package_logger.propagate = False

    for existing_handler in list(package_logger.handlers):
        package_logger.removeHandler(existing_handler)
        existing_handler.close()

    handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_path=False,
        omit_repeated_times=False,
    )
    package_logger.addHandler(handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        package_logger.addHandler(file_handler)

    return package_logger
