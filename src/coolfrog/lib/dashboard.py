"""Terminal dashboard of any `Tracker`: plotext charts, a table and a Textual app that runs the work in a thread.

- Shows what the web monitor shows, from the same tables and state, no per-run layout
- Charts follow `chart_groups`: per table the first column is x, the others group by name up to the last `/`
- Charts and tables are plain Rich renderables, reusable outside Textual
"""

from collections.abc import Callable
from contextlib import suppress
from itertools import cycle
import json
import logging
import math
import re
import sys
import threading
import time
from typing import Any

import plotext
from rich import box
from rich.logging import RichHandler
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.content import Content
from textual.geometry import Size
from textual.widgets import DataTable, Footer, Header, RichLog, Static

from coolfrog.lib.log import CONSOLE, NUMBER_HIGHLIGHTER, PACKAGE_LOGGER_NAME
from coolfrog.lib.metrics import Progress, Tracker

MIN_CHART_SIZE = Size(20, 6)
REFRESH_SECONDS = 0.5
CHART_ROWS = 16  # terminal rows per chart, the grid scrolls beyond the screen
MIN_TICKS = 2
MAX_TICKS = 6
MAX_LINEAR_TICKS = 5
MIN_LOG_DECADES = 1.0  # smallest span of a log axis
LOG_SCALE_RATIO = 10.0  # positive values spanning more than this factor get a log axis
PLOTEXT_LOCK = threading.Lock()  # plotext keeps one global figure
# Background and 256-color foreground codes of the plotext theme: stripped, so axes inherit the terminal colors,
# series keep their 24-bit colors
THEME_COLORS = re.compile(r"\x1b\[(?:48;[\d;]+|38;5;\d+)m")

PALETTE = ("#00d7ff", "#5fd75f", "#ffaf00", "#ff5faf", "#af87ff", "#ff5f5f", "#d7d75f")  # distinct on dark and light terminals
HIDDEN_STATE_KEYS = ("title", "progress", "tables")  # shown elsewhere, not as state lines


def chart_groups(columns: dict[str, str]) -> list[tuple[str, list[str]]]:
    """Charts of one table: (group, its columns), x is the first column, none when a column holds strings or bools.

    - `mse` -> its own chart, `nrmse/median/Vx` and `nrmse/median/Vy` -> one chart `nrmse/median`
    """
    names = list(columns)[1:]  # the first column is x
    if {"str", "bool"} & set(columns.values()):
        return []
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(name.rpartition("/")[0] or name, []).append(name)
    return list(groups.items())


def series_label(group: str, column: str) -> str:
    """`nrmse/median/Vx` in group `nrmse/median` -> `Vx`, a single-column group keeps its name."""
    return column.removeprefix(f"{group}/")


def is_log_scale(values: list[float]) -> bool:
    """Whether `values` are all positive and span more than `LOG_SCALE_RATIO`."""
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return bool(finite) and min(finite) > 0 and max(finite) / min(finite) > LOG_SCALE_RATIO


def charted_tables(infos: dict[str, dict[str, Any]]) -> list[str]:
    """Tables with charts, most rows first: the main log, e.g. `train`, before rarer ones."""
    charted = [name for name, info in infos.items() if chart_groups(info["columns"])]
    return sorted(charted, key=lambda name: -infos[name]["rows"])


def last_step(tracker: Tracker) -> dict[str, int]:
    """Largest x per x column name of the charted tables, charts sharing an x span 0 to it so validations line up."""
    maxima: dict[str, int] = {}
    for name, info in tracker.table_infos().items():
        if chart_groups(info["columns"]):
            x_name = next(iter(info["columns"]))
            xs = [x for x in tracker.column(name, x_name) if x is not None and math.isfinite(x)]
            maxima[x_name] = max([maxima.get(x_name, 0), *xs])
    return maxima


def format_cell(value: Any) -> str:  # noqa: ANN401, any table or state value
    """Table cell or state value: `-` for None, floats to 4 significant digits, nested values as JSON."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return value if isinstance(value, str) else json.dumps(value)


def render_chart(tracker: Tracker, table: str, group: str, size: Size, x_max: int) -> Text:
    """Plotext chart of one `chart_groups` group at exactly `size` cells, a dim hint when too small or plotext fails."""
    if size.width < MIN_CHART_SIZE.width or size.height < MIN_CHART_SIZE.height:
        return Text("terminal too small", style="dim")
    columns = tracker.table_infos()[table]["columns"]
    x_name, series = next(iter(columns)), dict(chart_groups(columns))[group]
    xs, *values = tracker.columns(table, [x_name, *series])
    lines = list(zip(cycle(PALETTE), values, strict=False))
    figure = plotext.figure  # one global figure, hence the lock
    with PLOTEXT_LOCK:
        try:
            figure.clear.all()
            figure.plot_size(size.width, size.height)
            figure.theme("dark")
            _draw(figure, xs, lines, x_max)
            figure.label(x_name, "x")
            return Text.from_ansi(THEME_COLORS.sub("", figure.build().string()).rstrip("\n"))
        except Exception as exception:  # a chart bug must never kill a training run
            return Text(f"chart unavailable: {exception!r}", style="dim")
        finally:
            figure.clear.all()


def _draw(figure: "plotext.figure", xs: list[Any], lines: list[tuple[str, list[Any]]], x_max: int) -> None:
    """Draw `lines` over x from 0 to `x_max`, log axes as log10 values with round 1-2-5 tick labels.

    - plotext's own log scale drops the signal once custom ticks are set
    - A None, NaN or inf value splits its line, each run of finite points is drawn separately
    """
    log_y = is_log_scale([value for _, values in lines for value in values])
    plotted_values: list[float] = []
    for color, values in lines:
        keep = [
            i
            for i, value in enumerate(values)
            if value is not None and xs[i] is not None and math.isfinite(value) and (value > 0 or not log_y)
        ]
        if not keep:
            continue
        for run in _consecutive_runs(keep):
            kept_xs, kept_values = [xs[i] for i in run], [values[i] for i in run]
            plotted = [math.log10(value) for value in kept_values] if log_y else kept_values
            signal = figure.signal(kept_xs, plotted, marker=plotext.marker("braille", _rgb(color)))
            signal.lines()
            figure.draw(signal)
            plotted_values += kept_values
    if plotted_values and log_y:
        low, high = log_axis_range(plotted_values)
        figure.ruler("y").lim(low, high)
        ticks, labels = log_ticks([10**low, 10**high])
        figure.ruler("y").ticks([math.log10(tick) for tick in ticks], labels)
    if x_max > 0:
        figure.ruler("x").lim(0, x_max)
        figure.ruler("x").ticks(*linear_ticks([0, x_max]))


def _consecutive_runs(indices: list[int]) -> list[list[int]]:
    """`[0, 1, 3, 4, 5]` -> `[[0, 1], [3, 4, 5]]`."""
    runs: list[list[int]] = []
    for index in indices:
        if runs and runs[-1][-1] == index - 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def render_table(tracker: Tracker, table: str, max_rows: int) -> Table:
    """Rich table of the last `max_rows` rows of `table`."""
    frame = tracker.frame(table, tail=max(max_rows, 1))
    rich_table = Table(show_header=True, header_style="bold cyan", expand=True, padding=(0, 1), box=box.SIMPLE_HEAD)
    for column in frame.columns:
        rich_table.add_column(column, justify="right", no_wrap=True)
    for row in frame.rows():
        rich_table.add_row(*(format_cell(value) for value in row))
    return rich_table


def render_progress(progress: dict[str, Progress]) -> Table:
    """One line per bar: label, count, bar, elapsed and remaining time."""
    now = time.monotonic()
    grid = Table.grid(padding=(0, 2))
    for _ in range(4):
        grid.add_column()
    for label, bar in progress.items():
        grid.add_row(
            label,
            f"{bar.completed:,} / {bar.total:,}",
            ProgressBar(total=bar.total, completed=bar.completed, width=40),
            Text(progress_times(bar, now), style="dim"),
        )
    return grid


def render_state(snapshot: dict[str, Any]) -> Text:
    """State lines below the progress: one line per nested dict, `[dim]key[/] value` pairs, e.g. one per config."""
    lines: list[str] = []

    def line(name: str, values: dict[str, Any]) -> str:
        return f"[dim]{name}[/] " + " · ".join(f"[dim]{key}[/] {format_cell(value)}" for key, value in values.items())

    for key, value in snapshot.items():
        if key in HIDDEN_STATE_KEYS:
            continue
        if isinstance(value, dict) and value and all(isinstance(inner, dict) for inner in value.values()):
            lines += [line(name, inner) for name, inner in value.items()]
        elif isinstance(value, dict):
            lines.append(line(key, value))
        else:
            lines.append(f"[dim]{key}[/] {format_cell(value)}")
    return Text.from_markup("\n".join(lines))


def _rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def duration(seconds: float) -> str:
    """`h:mm:ss`, hours unbounded: `1:02:03`, `26:00:00`."""
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def progress_times(progress: Progress, now: float) -> str:
    """Elapsed and remaining time of a pass, paused time excluded, `done in` once finished."""
    elapsed = progress.elapsed(now)
    if progress.finished is not None:
        return f"done in {duration(elapsed)}"
    remaining = progress.remaining(now)
    times = f"{duration(elapsed)} elapsed"
    if remaining is not None:
        times += f" · ~{duration(remaining)} left"
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


class ChartPane(Static):
    """Bordered chart of one table group, redrawn only when the tracker changed or the pane was resized."""

    def __init__(self, tracker: Tracker, table: str, group: str, columns: list[str], x_name: str) -> None:
        super().__init__(classes="pane")
        self._tracker, self._table, self._group = tracker, table, group
        self.x_name = x_name
        self._drawn: tuple[Size, int, int] | None = None
        self.x_max = 0
        self.border_title = f"{table} · {group}"
        if len(columns) > 1:
            parts: list[tuple[str, str] | str] = []
            for column, color in zip(columns, cycle(PALETTE), strict=False):
                parts += [("●", color), f" {series_label(group, column)}  "]
            self.border_subtitle = Content.assemble(*parts).rstrip()

    def redraw(self) -> None:
        """Rebuild unless nothing changed since the last draw."""
        key = (self.content_size, self._tracker.version, self.x_max)
        if key != self._drawn and self.content_size.area:
            self._drawn = key
            self.update(render_chart(self._tracker, self._table, self._group, self.content_size, self.x_max))

    def on_resize(self) -> None:
        self.redraw()


class TablePane(DataTable):
    """Scrollable rows of one table, follows new rows unless scrolled up."""

    def __init__(self, tracker: Tracker, table: str) -> None:
        super().__init__(classes="pane table-pane", cursor_type="none")
        self._tracker, self._table = tracker, table
        self._columns: list[str] = []
        self._drawn_rows = 0
        self.border_title = table

    def redraw(self) -> None:
        """Rebuild when columns appeared, else append the new rows."""
        frame = self._tracker.frame(self._table, start=self._drawn_rows)
        is_following = self.scroll_y >= self.max_scroll_y
        if frame.columns != self._columns:  # a column appeared, refetch every row
            frame = self._tracker.frame(self._table)
            self.clear(columns=True)
            self._columns, self._drawn_rows = frame.columns, 0
            for column in self._columns:
                self.add_column(Text(column, style="bold cyan", justify="right"), key=column)
        if frame.height == 0:
            return
        for row in frame.rows():
            self.add_row(*(Text(format_cell(value), justify="right") for value in row))
        self._drawn_rows += frame.height
        if is_following:
            self.call_after_refresh(self.scroll_end, animate=False)  # after the new rows are laid out


class DashboardApp(App[None]):
    """Runs `work` in a daemon thread while showing progress, state, the first table, every chart and log lines (`l`)."""

    BINDINGS = [  # noqa: RUF012
        Binding("l", "toggle_log", "Log"),
        Binding("q", "stop", "Stop"),
    ]
    CSS = f"""
    #progress, #state {{ height: auto; padding: 0 1; }}
    #grid {{ layout: grid; grid-size: 2; grid-rows: {CHART_ROWS}; height: 1fr; overflow-y: auto; }}
    .table-pane {{ column-span: 2; }}
    .pane {{ height: 100%; border: round $primary; padding: 0 1; }}
    .pane:focus {{ border: round $accent; }}
    #log {{ display: none; height: 30%; border: round $primary; }}
    """

    def __init__(self, tracker: Tracker, work: Callable[[], object]) -> None:
        super().__init__()
        self.tracker, self._work = tracker, work
        self.result: object = None
        self.error: BaseException | None = None
        self._ui_thread = threading.get_ident()
        self._panes: dict[tuple[str, str], ChartPane | TablePane] = {}
        self._drawn_version = -1
        self._work_thread: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="progress")
        yield Static(id="state")
        yield Container(id="grid")
        yield RichLog(id="log", markup=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
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
        """Block until `work` ends, Ctrl+C asks it to stop at its next `append` or `set_progress` and waits for that."""
        try:
            while self.is_working:
                self._work_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            self.tracker.request_stop()
            while self.is_working:
                self._work_thread.join(timeout=1.0)
            raise

    def _refresh(self) -> None:
        """Redraw what changed, mount panes for tables and chart groups that appeared."""
        snapshot = self.tracker.snapshot()
        self.title = snapshot.get("title", "Monitor")
        self.query_one("#progress", Static).update(render_progress(self.tracker.progress()))  # times run on
        if self.tracker.version != self._drawn_version:
            self._drawn_version = self.tracker.version
            self.query_one("#state", Static).update(render_state(snapshot))
            self._mount_new_panes(snapshot["tables"])
        x_maxima = last_step(self.tracker)
        self.sub_title = " · ".join(f"{x_name} {x_max}" for x_name, x_max in x_maxima.items())
        for pane in self._panes.values():
            if isinstance(pane, ChartPane):
                pane.x_max = x_maxima.get(pane.x_name, 0)
            pane.redraw()

    def _mount_new_panes(self, tables: dict[str, dict[str, Any]]) -> None:
        """Remount every pane in `charted_tables` order once a table or chart group appeared, rare and cheap."""
        order = charted_tables(tables)
        keys = (
            [(order[0], ""), *((table, group) for table in order for group, _ in chart_groups(tables[table]["columns"]))]
            if order
            else []
        )
        if keys == list(self._panes):
            return
        grid = self.query_one("#grid", Container)
        grid.remove_children()
        self._panes = {}
        for table, group in keys:
            columns, x_name = dict(chart_groups(tables[table]["columns"])), next(iter(tables[table]["columns"]))
            self._panes[(table, group)] = (
                ChartPane(self.tracker, table, group, columns[group], x_name) if group else TablePane(self.tracker, table)
            )
        grid.mount(*self._panes.values())

    def action_toggle_log(self) -> None:
        log = self.query_one("#log", RichLog)
        log.display = not log.display

    def action_stop(self) -> None:
        """First press asks the work to stop at its next `append`, the second quits at once."""
        if self.tracker.stop_requested:
            self.exit()
            return
        self.tracker.request_stop()
        self.notify("Stopping at the next logged row, press q again to quit immediately", timeout=8)

    def write_log(self, text: Text) -> None:
        """Append a line to the log pane from any thread."""
        log = self.query_one("#log", RichLog)
        if threading.get_ident() == self._ui_thread:
            log.write(text)
        else:
            self.call_from_thread(log.write, text)


class _AppLogHandler(logging.Handler):
    """Shows `coolfrog` log records in the app's log pane, as stderr writes would corrupt the screen."""

    def __init__(self, app: DashboardApp) -> None:
        super().__init__()
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        try:
            self._app.write_log(NUMBER_HIGHLIGHTER(message))
        except Exception:  # closed app or full queue, never break the work for a log line
            self.handleError(record)


def run_dashboard[T](tracker: Tracker, work: Callable[[], T], *, enabled: bool = True) -> T:
    """Run `work` while showing `tracker` live, returning its result or re-raising its exception.

    - Without a terminal (pipes, CI) or `enabled=False` -> `work` runs plainly, logging unchanged
    - Console log handlers move into the app's log pane meanwhile, file handlers stay
    - Last rows of the first table stay in the scrollback afterwards
    - `q` stops at the next `tracker.append` or `tracker.set_progress` as `KeyboardInterrupt`
    - The dashboard crashing never ends `work` early: it runs on with plain logging until done
    """
    if not enabled or not (sys.stdout.isatty() and sys.stderr.isatty()):
        return work()
    app = DashboardApp(tracker, work)
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
    tables = charted_tables(tracker.table_infos())
    if tables:
        CONSOLE.print(render_table(tracker, tables[0], max_rows=20))
    if app.error is not None:
        raise app.error
    return app.result
