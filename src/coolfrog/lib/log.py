"""Package logging and progress bars, both Rich on one shared stderr console so bars stay below log lines."""

from collections.abc import Iterable, Iterator
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

PACKAGE_LOGGER_NAME = "coolfrog"
FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
CONSOLE = Console(stderr=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger, use as `log = get_logger(__name__)`."""
    return logging.getLogger(name)


def setup_logging(level: int | str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure the `coolfrog` logger, safe to call repeatedly.

    - Console: Rich, level and message only, colors off when not on a TTY (pipes, CI)
    - `log_file`: plain text with timestamps
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    package_logger.setLevel(level)
    package_logger.propagate = False

    for existing_handler in list(package_logger.handlers):
        package_logger.removeHandler(existing_handler)
        existing_handler.close()

    package_logger.addHandler(RichHandler(console=CONSOLE, show_time=False, show_path=False, rich_tracebacks=True))

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        package_logger.addHandler(file_handler)

    return package_logger


def progress_bar() -> Progress:
    """Rich progress bar: description, bar, done/total, percent, elapsed and remaining time.

    >>> with progress_bar() as progress:
    ...     task = progress.add_task("Voxelizing designs", total=100)
    ...     progress.advance(task, 64)
    """
    return Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TextColumn("({task.percentage:.0f}%)"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    )


def track[T](items: Iterable[T], description: str, total: int | None = None) -> Iterator[T]:
    """Yield `items` under a `progress_bar`, `total` defaults to `len(items)`: `for batch in track(loader, "Loading")`."""
    with progress_bar() as progress:
        yield from progress.track(items, total=total, description=description)
