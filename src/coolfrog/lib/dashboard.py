"""Terminal dashboard for any `MetricTracker`: Rich table, plotext charts and a Textual app that runs the work in a thread.

- `DashboardSpec` declares what to show, the app only reads the tracker
- Charts and tables are plain Rich renderables, reusable outside Textual
"""

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
import logging
import math
import re
import sys
import threading
import time
from typing import Literal

import plotext
from rich import box
from rich.highlighter import Highlighter
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.content import Content
from textual.geometry import Size
from textual.widgets import DataTable, Footer, Header, Label, ProgressBar, RichLog, Static

from coolfrog.lib.log import CONSOLE, NUMBER_HIGHLIGHTER, PACKAGE_LOGGER_NAME
from coolfrog.lib.metrics import MetricTracker, Progress

MIN_CHART_SIZE = Size(20, 6)
REFRESH_SECONDS = 0.5
GRID_COLUMNS = 2
MIN_TICKS = 2
MAX_TICKS = 6
MAX_LINEAR_TICKS = 5
MIN_LOG_DECADES = 1.0  # smallest span of a log axis
PLOTEXT_LOCK = threading.Lock()  # plotext keeps one global figure
# Background and 256-color foreground codes of the plotext theme: stripped, so axes inherit the terminal colors,
# series keep their 24-bit colors
THEME_COLORS = re.compile(r"\x1b\[(?:48;[\d;]+|38;5;\d+)m")

PALETTE = ("#00d7ff", "#5fd75f", "#ffaf00", "#ff5faf", "#af87ff", "#ff5f5f", "#d7d75f")  # distinct on dark and light terminals


@dataclass(frozen=True)
class SeriesSpec:
    """One plotted line, `color` is `#rrggbb` and shared by legend and plot."""

    name: str
    label: str
    color: str
    side: Literal["left", "right"] = "left"


@dataclass(frozen=True)
class ChartSpec:
    """One chart, `log_y` drops non-positive values instead of crashing plotext's log scale."""

    title: str
    series: tuple[SeriesSpec, ...]
    y_label: str = ""
    log_y: bool = True


@dataclass(frozen=True)
class ColumnSpec:
    """One table column showing series `name`, `fmt` is a `str.format` pattern for its value."""

    header: str
    name: str
    style: str = ""
    fmt: str = "{:.2e}"


@dataclass(frozen=True)
class DashboardSpec:
    """What the dashboard shows.

    Attributes
    ----------
    title : str
        App title.
    charts : tuple[ChartSpec, ...]
        Charts in reading order, laid out two per row after the table.
    columns : tuple[ColumnSpec, ...]
        Table columns, a row per step at which any column's series was logged.
    step_header : str
        Header of the step column.
    progress_bars : tuple[str, ...]
        Labels of the bars shown on top, fed by `MetricTracker.set_progress`, e.g. ("Training", "Validation").
    summary : str
        Rich markup shown under the progress bars, e.g. the key hyperparameters.
    """

    title: str
    charts: tuple[ChartSpec, ...]
    columns: tuple[ColumnSpec, ...]
    step_header: str = "Step"
    progress_bars: tuple[str, ...] = ()
    summary: str = ""


# ---------------------------------------------------------------------------
# Rich renderables
# ---------------------------------------------------------------------------


def render_chart(tracker: MetricTracker, spec: ChartSpec, size: Size) -> Text:
    """Plotext chart of `spec` at exactly `size` cells, a dim hint when too small or plotext fails."""
    if size.width < MIN_CHART_SIZE.width or size.height < MIN_CHART_SIZE.height:
        return Text("terminal too small", style="dim")
    lines = [(line, *tracker.series(line.name)) for line in spec.series]
    if not any(steps for _, steps, _ in lines):
        return Text("waiting for data", style="dim")
    figure = plotext.figure  # one global figure, hence the lock
    with PLOTEXT_LOCK:
        try:
            figure.clear.all()
            figure.plot_size(size.width, size.height)
            figure.theme("dark")
            _draw(figure, spec, lines, tracker.last_step or 0)
            return Text.from_ansi(THEME_COLORS.sub("", figure.build().string()).rstrip("\n"))
        except Exception as exception:  # a chart bug must never kill a training run
            return Text(f"chart unavailable: {exception!r}", style="dim")
        finally:
            figure.clear.all()


def chart_legend(spec: ChartSpec) -> Content:
    """`● label` per series, the dot in the series color."""
    parts: list[tuple[str, str] | str] = []
    for line in spec.series:
        parts += [("●", line.color), f" {line.label}  "]
    return Content.assemble(*parts).rstrip()


def render_table(tracker: MetricTracker, spec: DashboardSpec, max_rows: int) -> Table:
    """Rich table with the last `max_rows` steps, `-` where a series skipped the step."""
    table = Table(show_header=True, header_style="bold cyan", expand=True, padding=(0, 1), box=box.SIMPLE_HEAD)
    table.add_column(spec.step_header, justify="right", style="cyan", no_wrap=True)
    for column in spec.columns:
        table.add_column(column.header, justify="right", style=column.style, no_wrap=True)
    for step in tracker.recent_steps([column.name for column in spec.columns], max(max_rows, 1)):
        cells = [_format(tracker.value_at(column.name, step), column.fmt) for column in spec.columns]
        table.add_row(str(step), *cells)
    return table


def _format(value: float | None, pattern: str) -> str:
    return "-" if value is None else pattern.format(value)


def _draw(figure: "plotext.figure", spec: ChartSpec, lines: list[tuple[SeriesSpec, list[int], list[float]]], last_step: int) -> None:
    """Draw `lines` into `figure` over steps 0 to `last_step`, log axes as log10 values with round 1-2-5 tick labels.

    - plotext's own log scale drops the signal once custom ticks are set
    """
    values_per_side: dict[str, list[float]] = {"left": [], "right": []}
    steps: list[int] = [last_step]
    for line, line_steps, line_values in lines:
        keep = [i for i, value in enumerate(line_values) if math.isfinite(value) and (value > 0 or not spec.log_y)]
        if not keep:
            continue
        kept_steps, kept_values = [line_steps[i] for i in keep], [line_values[i] for i in keep]
        plotted = [math.log10(value) for value in kept_values] if spec.log_y else kept_values
        signal = figure.signal(kept_steps, plotted, marker=plotext.marker("braille", _rgb(line.color)), yside=line.side)
        signal.lines()
        figure.draw(signal)
        steps += kept_steps
        values_per_side[line.side] += kept_values
    for side, values in values_per_side.items():
        if values and spec.log_y:
            low, high = log_axis_range(values)
            figure.ruler("y", side).lim(low, high)
            ticks, labels = log_ticks([10**low, 10**high])
            figure.ruler("y", side).ticks([math.log10(tick) for tick in ticks], labels)
    last_step = max(steps, default=0)
    if last_step > 0:  # every chart spans 0 to the tracker's last step, so validation points line up with training
        figure.ruler("x").lim(0, last_step)
        figure.ruler("x").ticks(*linear_ticks([0, last_step]))
    figure.label("step", "x")
    if spec.y_label:
        figure.label(spec.y_label, "y")


def _rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def duration(seconds: float) -> str:
    """`h:mm:ss`, hours unbounded: `1:02:03`, `26:00:00`."""
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def progress_times(progress: Progress, now: float) -> str:
    """Elapsed, remaining and projected total time of a pass, paused time excluded, `done in` once finished."""
    elapsed = progress.elapsed(now)
    if progress.finished is not None:
        return f"done in {duration(elapsed)}"
    remaining = progress.remaining(now)
    times = f"{duration(elapsed)} elapsed"
    if remaining is not None:
        times += f" · ~{duration(remaining)} left · ~{duration(elapsed + remaining)} total"
    return times + (" · paused" if progress.paused_at is not None else "")


def scientific(value: float) -> str:
    """Compact scientific notation, `2e-3`, `1.5e-5`."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def log_axis_range(values: list[float]) -> tuple[float, float]:
    """log10 range covering `values` (all > 0), widened around its middle to at least `MIN_LOG_DECADES`.

    - Keeps a nearly flat series from filling the chart height, its noise would look like large swings
    """
    low, high = math.log10(min(values)), math.log10(max(values))
    if high - low < MIN_LOG_DECADES:
        middle = (low + high) / 2
        low, high = middle - MIN_LOG_DECADES / 2, middle + MIN_LOG_DECADES / 2
    return low, high


def log_ticks(values: list[float]) -> tuple[list[float], list[str]]:
    """1-2-5 ticks per decade covering `values` (all > 0) thinned to `MAX_TICKS`, min and max when no round tick fits."""
    low, high = min(values), max(values)
    ticks = [m * 10.0**k for k in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1) for m in (1, 2, 5)]
    ticks = [tick for tick in ticks if low <= tick <= high]
    if len(ticks) < MIN_TICKS:
        ticks = sorted({low, high})
    ticks = ticks[:: math.ceil(len(ticks) / MAX_TICKS)]
    return ticks, [scientific(tick) for tick in ticks]


def linear_ticks(values: list[float]) -> tuple[list[float], list[str]]:
    """At most 5 round ticks (1-2-5 steps) covering `values`."""
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    step = 10.0 ** math.floor(math.log10(span / MAX_LINEAR_TICKS))
    for multiplier in (1, 2, 5, 10):
        if span / (step * multiplier) <= MAX_LINEAR_TICKS:
            step *= multiplier
            break
    ticks = [index * step for index in range(math.ceil(low / step), math.floor(high / step) + 1)]
    return ticks, [f"{tick:g}" for tick in ticks]


# ---------------------------------------------------------------------------
# Textual app
# ---------------------------------------------------------------------------


class LivePane(Static):
    """Bordered pane redrawn only when the tracker changed or the pane was resized."""

    def __init__(self, tracker: MetricTracker) -> None:
        super().__init__(classes="pane")
        self._tracker = tracker
        self._drawn: tuple[Size, int] | None = None

    def build(self, size: Size) -> Text | Table:
        raise NotImplementedError

    def redraw(self) -> None:
        """Rebuild unless nothing changed since the last draw."""
        key = (self.content_size, self._tracker.version)
        if key != self._drawn and self.content_size.area:
            self._drawn = key
            self.update(self.build(self.content_size))

    def on_resize(self) -> None:
        self.redraw()


class ChartPane(LivePane):
    def __init__(self, tracker: MetricTracker, spec: ChartSpec) -> None:
        super().__init__(tracker)
        self._spec = spec
        self.border_title = spec.title
        self.border_subtitle = chart_legend(spec)

    def build(self, size: Size) -> Text:
        return render_chart(self._tracker, self._spec, size)


class MetricsTable(DataTable):
    """Scrollable table of every logged step, follows new rows unless scrolled up."""

    def __init__(self, tracker: MetricTracker, spec: DashboardSpec) -> None:
        super().__init__(classes="pane", cursor_type="none")
        self._tracker, self._spec = tracker, spec
        self._drawn_version = -1
        self._steps: list[int] = []
        self.border_title = "Metrics"

    def on_mount(self) -> None:
        self.add_column(Text(self._spec.step_header, style="bold cyan", justify="right"), key="step")
        for column in self._spec.columns:
            self.add_column(Text(column.header, style="bold cyan", justify="right"), key=column.name)

    def redraw(self) -> None:
        """Append new steps and refresh the last rows, values can arrive later at a step, e.g. its validation."""
        if self._tracker.version == self._drawn_version:
            return
        self._drawn_version = self._tracker.version
        is_following = self.scroll_y >= self.max_scroll_y
        for step in self._steps[-2:]:
            for column in self._spec.columns:
                self.update_cell(str(step), column.name, self._cell(column, step))
        names = [column.name for column in self._spec.columns]
        for step in self._tracker.recent_steps(names, count=sys.maxsize):
            if self._steps and step <= self._steps[-1]:
                continue
            cells = [self._cell(column, step) for column in self._spec.columns]
            self.add_row(Text(str(step), style="cyan", justify="right"), *cells, key=str(step))
            self._steps.append(step)
        if is_following:
            self.call_after_refresh(self.scroll_end, animate=False)  # after the new rows are laid out

    def _cell(self, column: ColumnSpec, step: int) -> Text:
        return Text(_format(self._tracker.value_at(column.name, step), column.fmt), style=column.style, justify="right")


class DashboardApp(App[None]):
    """Runs `work` in a daemon thread while showing live progress, table, charts and log lines (`l`)."""

    BINDINGS = [  # noqa: RUF012
        Binding("l", "toggle_log", "Log"),
        Binding("q", "stop", "Stop"),
    ]
    CSS = """
    .progress-row { height: 1; padding: 0 1; }
    .progress-label { width: 12; }
    .progress-count { width: 17; text-align: right; padding: 0 2 0 0; }
    .progress-time { width: auto; padding: 0 0 0 2; color: $text-muted; }
    #summary { height: auto; padding: 0 1; }
    #grid { layout: grid; grid-size: 2; height: 1fr; }
    .pane { height: 100%; border: round $primary; padding: 0 1; }
    .pane:focus { border: round $accent; }
    #log { display: none; height: 30%; border: round $primary; }
    """

    def __init__(self, tracker: MetricTracker, spec: DashboardSpec, work: Callable[[], object]) -> None:
        super().__init__()
        self.tracker, self.spec, self._work = tracker, spec, work
        self.result: object = None
        self.error: BaseException | None = None
        self._ui_thread = threading.get_ident()
        self._panes: list[LivePane | MetricsTable] = []
        self._work_thread: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        for index, label in enumerate(self.spec.progress_bars):
            yield Horizontal(
                Label(label, classes="progress-label"),
                Label("", id=f"progress-count-{index}", classes="progress-count"),
                ProgressBar(id=f"progress-{index}", show_eta=False),
                Label("", id=f"progress-time-{index}", classes="progress-time"),
                classes="progress-row",
            )
        yield Static(self.spec.summary, id="summary")
        self._panes = [MetricsTable(self.tracker, self.spec), *(ChartPane(self.tracker, chart) for chart in self.spec.charts)]
        yield Container(*self._panes, id="grid")
        yield RichLog(id="log", markup=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.spec.title
        self.query_one("#grid").styles.grid_size_rows = math.ceil(len(self._panes) / GRID_COLUMNS)
        self.set_interval(REFRESH_SECONDS, self._refresh)
        self._work_thread = threading.Thread(target=self._run_work, name="dashboard-work", daemon=True)
        self._work_thread.start()

    def _run_work(self) -> None:
        try:
            self.result = self._work()
        except BaseException as exception:  # re-raised by `run_dashboard` in the caller's thread
            self.error = exception
        finally:
            with suppress(RuntimeError):  # app already closed after a forced quit
                self.call_from_thread(self.exit)

    @property
    def is_working(self) -> bool:
        """Whether `work` is still running."""
        return self._work_thread is not None and self._work_thread.is_alive()

    def wait_for_work(self) -> None:
        """Block until `work` ends, Ctrl+C asks it to stop at its next `log` or `set_progress` and waits for that."""
        try:
            while self.is_working:
                self._work_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            self.tracker.request_stop()
            while self.is_working:
                self._work_thread.join(timeout=1.0)
            raise

    def _refresh(self) -> None:
        step = self.tracker.last_step
        if step is not None:
            self.sub_title = f"step {step}"
        now = time.monotonic()
        for index, label in enumerate(self.spec.progress_bars):
            progress = self.tracker.progress(label)
            if progress is None:
                continue
            self.query_one(f"#progress-{index}", ProgressBar).update(total=progress.total, progress=progress.completed)
            self.query_one(f"#progress-count-{index}", Label).update(f"{progress.completed:,} / {progress.total:,}")
            self.query_one(f"#progress-time-{index}", Label).update(progress_times(progress, now))
        for pane in self._panes:
            pane.redraw()

    def action_toggle_log(self) -> None:
        log = self.query_one("#log", RichLog)
        log.display = not log.display

    def action_stop(self) -> None:
        """First press asks the work to stop at its next `log`, the second quits at once."""
        if self.tracker.stop_requested:
            self.exit()
            return
        self.tracker.request_stop()
        self.notify("Stopping at the next logged step, press q again to quit immediately", timeout=8)

    def write_log(self, text: Text) -> None:
        """Append a line to the log pane from any thread."""
        log = self.query_one("#log", RichLog)
        if threading.get_ident() == self._ui_thread:
            log.write(text)
        else:
            self.call_from_thread(log.write, text)


class _AppLogHandler(logging.Handler):
    """Shows `coolfrog` log records in the app's log pane, as stderr writes would corrupt the screen."""

    def __init__(self, app: DashboardApp, highlighter: Highlighter = NUMBER_HIGHLIGHTER) -> None:
        super().__init__()
        self._app, self._highlighter = app, highlighter

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        try:
            self._app.write_log(self._highlighter(message))
        except Exception:  # closed app or full queue, never break the work for a log line
            self.handleError(record)


def run_dashboard[T](tracker: MetricTracker, spec: DashboardSpec, work: Callable[[], T], *, enabled: bool = True) -> T:
    """Run `work` while showing `spec` live, returning its result or re-raising its exception.

    - Without a terminal (pipes, CI) or `enabled=False` -> `work` runs plainly, logging unchanged
    - Console log handlers move into the app's log pane meanwhile, file handlers stay
    - Last table rows stay in the scrollback afterwards
    - `q` stops at the next `tracker.log` or `tracker.set_progress` as `KeyboardInterrupt`
    - The dashboard crashing never ends `work` early: it runs on with plain logging until done
    """
    if not enabled or not (sys.stdout.isatty() and sys.stderr.isatty()):
        return work()
    app = DashboardApp(tracker, spec, work)
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    console_handlers = [handler for handler in package_logger.handlers if isinstance(handler, RichHandler)]
    app_handler = _AppLogHandler(app)
    app_handler.setFormatter(logging.Formatter("%(message)s"))
    for handler in console_handlers:
        package_logger.removeHandler(handler)
    package_logger.addHandler(app_handler)
    try:
        app.run()
    finally:
        package_logger.removeHandler(app_handler)
        for handler in console_handlers:
            package_logger.addHandler(handler)
    if app.tracker.stop_requested:
        raise KeyboardInterrupt("stopped from the dashboard")
    if app.is_working:  # the UI ended on its own, e.g. a rendering error: keep the work, finish without the dashboard
        logging.getLogger(__name__).warning("Dashboard closed, the work continues with plain logging, Ctrl+C stops it")
        app.wait_for_work()
    CONSOLE.print(render_table(tracker, spec, max_rows=20))
    if app.error is not None:
        raise app.error
    return app.result
