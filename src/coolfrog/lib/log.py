"""Package logging and progress bars, both Rich on one shared stderr console so bars stay below log lines."""

from collections.abc import Iterable, Iterator
import logging
from pathlib import Path
from typing import ClassVar

from rich.console import Console, RenderableType
from rich.highlighter import RegexHighlighter
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class NumberHighlighter(RegexHighlighter):
    """Colors every number, also each axis of `1600x320x64`, but not digits within names like `tmp1o7.tmp`."""

    base_style = "repr."
    highlights: ClassVar[list[str]] = [
        r"(?:(?<=\dx)|(?<![A-Za-z_.\d]))(?P<number>\d+(?:\.\d+)?)(?:(?=x\d)|(?![A-Za-z_\d]))",
    ]


PACKAGE_LOGGER_NAME = "coolfrog"
FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
CONSOLE = Console(stderr=True)
NUMBER_HIGHLIGHTER = NumberHighlighter()


def setup_logging(level: int | str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure the `coolfrog` logger, safe to call repeatedly.

    - Console: Rich, message only with numbers colored, colors off when not on a TTY (pipes, CI)
    - `log_file`: plain text with timestamps
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    package_logger.setLevel(level)
    package_logger.propagate = False

    for existing_handler in list(package_logger.handlers):
        package_logger.removeHandler(existing_handler)
        existing_handler.close()

    handler = RichHandler(
        console=CONSOLE, show_time=False, show_level=False, show_path=False, highlighter=NUMBER_HIGHLIGHTER, rich_tracebacks=True
    )
    package_logger.addHandler(handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        package_logger.addHandler(file_handler)

    return package_logger


class StatusProgress(Progress):
    """Rich progress bars plus one status line below them, hidden until `set_status`."""

    def __init__(self, *columns: ProgressColumn, console: Console) -> None:
        self.status = Text()  # before super().__init__, which already renders
        super().__init__(*columns, console=console)

    def set_status(self, message: str) -> None:
        """Replace the status line, numbers colored like log lines."""
        self.status = NUMBER_HIGHLIGHTER(message)

    def get_renderables(self) -> Iterator[RenderableType]:
        yield from super().get_renderables()
        if self.status.plain:
            yield self.status


def progress_bar() -> StatusProgress:
    """Rich progress bar: description, bar, done/total, percent, elapsed and remaining time, optional status line below.

    >>> with progress_bar() as progress:
    ...     task = progress.add_task("Voxelizing designs", total=100)
    ...     progress.advance(task, 64)
    ...     progress.set_status("170.0 GB needed · 258.0 GB free")
    """
    return StatusProgress(
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
