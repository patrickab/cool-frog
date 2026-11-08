"""Every non-image plot of coolfrog: typed input schemas and functions returning a matplotlib `Figure`.

- Notebooks, experiment PNGs and the web monitor call the same functions, a plot changes in one place
- Figures come from `Figure(...)`, never pyplot, so threads can render them
- Style: `matplotlib.rc_context(STYLE)` in scripts, `plt.rcParams.update(STYLE)` in notebooks, `plot_theme` in the monitor
- Every panel has one y axis, a title and axis labels with a unit or `normalized`
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import io
import math
import threading
from typing import Any, NamedTuple

from cycler import cycler
import matplotlib as mpl
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np
import polars as pl

from coolfrog.lib.config import Grid
from coolfrog.lib.datatypes import CHANNELS, Channel, FlowField
from coolfrog.lib.metrics import aggregate_schema

# Raw tracker tables of a fine-tuning run, one row per sample, and their aggregates, one row per `*_BY` group
METRICS_TRAINING, METRICS_VALIDATION = "metrics_training", "metrics_validation"
METRICS_TRAINING_AGGREGATES, METRICS_VALIDATION_AGGREGATES = f"{METRICS_TRAINING}_aggregates", f"{METRICS_VALIDATION}_aggregates"
CHANNEL_SYMBOLS = tuple(channel.value.partition(" [")[0] for channel in CHANNELS)  # `p [Pa]` -> `p`, model order
CHANNEL_LABELS = dict(zip(CHANNEL_SYMBOLS, (channel.value for channel in CHANNELS), strict=True))  # `p` -> `p [Pa]`
CHANNEL_ENUM = pl.Enum(CHANNEL_SYMBOLS)

# Optional `run` (String) column on the metric frames below: one color per run, e.g. to compare latent scales
SAMPLE_SCHEMA = pl.Schema({"update": pl.Int64, "design_id": pl.Int64, "channel": CHANNEL_ENUM})  # one design and channel
TRAINING_METRICS = ("mse", "spectral_loss", "kl")  # per sample, aggregated over each update's samples
VALIDATION_METRICS = ("nrmse", "mse", "spectral_loss", "kl")
TRAINING_SCHEMA = pl.Schema({**SAMPLE_SCHEMA, **dict.fromkeys(TRAINING_METRICS, pl.Float64)})  # every sample of every update
VALIDATION_SCHEMA = pl.Schema({**SAMPLE_SCHEMA, **dict.fromkeys(VALIDATION_METRICS, pl.Float64)})  # every design per validation
TRAINING_BY, VALIDATION_BY = ("update",), ("update", "channel")  # groups of the aggregates
TRAINING_AGGREGATES_SCHEMA = pl.Schema(
    {
        **aggregate_schema(TRAINING_SCHEMA, TRAINING_BY, TRAINING_METRICS),
        "learning_rate": pl.Float64,  # one value per update, logged once, never aggregated
        "gradient_norm": pl.Float64,  # L2 norm ||g|| over the trainable parameters, averaged over passes and GPUs
        "gradient_to_weight_norm": pl.Float64,  # ||g|| / ||θ||, θ before the step
        "seconds_per_update": pl.Float64,
    }
)
VALIDATION_AGGREGATES_SCHEMA = aggregate_schema(VALIDATION_SCHEMA, VALIDATION_BY, VALIDATION_METRICS)
CHANNEL_AXES_SCHEMA = pl.Schema(  # one row per channel, axis settings of the value-distribution plots
    {
        "channel": CHANNEL_ENUM,
        "offset": pl.Float64,  # subtracted from physical values before plotting, e.g. 293.15 K for T
        "linear_width": pl.Float64,  # asinh axis: linear within it around `offset`, logarithmic beyond
        "rule_limit": pl.Float64,  # upper bound of the outlier rule, null without one
        "is_signed": pl.Boolean,  # the design extreme is max |value| instead of max
        "is_log": pl.Boolean,  # normalized in log10
    }
)
DESIGN_EXTREMES_SCHEMA = pl.Schema(  # one row per design and channel, physical values
    {"channel": CHANNEL_ENUM, "extreme": pl.Float64, "q0.999": pl.Float64, "median": pl.Float64}
)
HISTOGRAM_SCHEMA = pl.Schema(  # one row per channel and bin, physical bin edges, shares of the channel's fluid voxels
    {"channel": CHANNEL_ENUM, "bin_low": pl.Float64, "bin_high": pl.Float64, "share_all": pl.Float64, "share_curated": pl.Float64}
)
QUANTILE_SCHEMA = pl.Schema({"channel": CHANNEL_ENUM, "quantile": pl.Float64, "value": pl.Float64})  # physical values
NORMALIZED_HISTOGRAM_SCHEMA = pl.Schema(  # one row per channel and bin, normalized bin edges
    {"channel": CHANNEL_ENUM, "bin_low": pl.Float64, "bin_high": pl.Float64, "share": pl.Float64}
)

# Okabe-Ito, colorblind-safe, the dark variant lightened for contrast on dark backgrounds
LIGHT_COLORWAY = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#000000")
DARK_COLORWAY = ("#4da6e8", "#f2803d", "#2fc497", "#e397c4", "#f2b84b", "#8fd3f7", "#e6e1d8")
LIGHT_TEXT, LIGHT_GRID = "#2a2620", "#e1dfda"
DARK_TEXT, DARK_GRID = "#e6e1d8", "#2f2c28"
REFERENCE_GRAY = "#a39d94"  # context behind the colored series, e.g. all designs behind the curated ones
TUM_BLUE, TUM_LIGHT_BLUE = "#0065bd", "#98c6ea"  # TUM corporate design: single-run training lines, EMA and its raw curve
DARK_LIGHT_BLUE_ALPHA = 0.45  # light blue outshines TUM blue on dark backgrounds, fading keeps the EMA in front
SOLID_COLOR = "#999999"
LOG_SCALE_RATIO = 10.0  # positive values spanning more than this factor get a log axis


def theme_colors(text: str, grid: str, colorway: Sequence[str]) -> dict[str, Any]:
    """rcParams of text, axes, grid and line colors."""
    return {
        "text.color": text,
        "axes.labelcolor": text,
        "axes.edgecolor": text,
        "xtick.color": text,
        "ytick.color": text,
        "grid.color": grid,
        "axes.prop_cycle": cycler(color=colorway),
    }


STYLE: dict[str, Any] = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.3,
    "lines.markersize": 3,
} | theme_colors(LIGHT_TEXT, LIGHT_GRID, LIGHT_COLORWAY)
TRANSPARENT = {"figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True}
MONITOR_THEMES = {
    False: STYLE | TRANSPARENT,
    True: STYLE | TRANSPARENT | theme_colors(DARK_TEXT, DARK_GRID, DARK_COLORWAY),
}
THEME_LOCK = threading.RLock()  # rcParams are global, one theme at a time


class Panel(NamedTuple):
    """One curve panel: aggregate `column`, title, y label with unit, EMA over the raw curve, `log_y` None = `is_log_scale`."""

    column: str
    title: str
    y_label: str
    is_smoothed: bool = False
    log_y: bool | None = None


TRAIN_PANELS = (
    Panel("mse/mean", "MSE", "MSE [normalized]", is_smoothed=True),
    Panel("spectral_loss/mean", "Spectral loss", "spectral loss [normalized]", is_smoothed=True),
    Panel("kl/mean", "KL divergence", "KL [nats]", is_smoothed=True),
)
OPTIMIZER_PANELS = (
    Panel("learning_rate", "Learning rate", "learning rate [unitless]", log_y=True),  # warmup spans decades
    Panel("gradient_norm", "Gradient norm", "‖g‖ [unitless]", is_smoothed=True),
    Panel("gradient_to_weight_norm", "Gradient to weight norm", "‖g‖ / ‖θ‖ [unitless]", is_smoothed=True),
)
VALIDATION_PANELS = (  # TRAIN_PANELS plus NRMSE
    Panel("mse/median", "MSE", "MSE [normalized]"),
    Panel("spectral_loss/median", "Spectral loss", "spectral loss [normalized]"),
    Panel("kl/median", "KL divergence", "KL [nats]"),
    Panel("nrmse/median", "NRMSE", "NRMSE [%]"),
)
EMA_SMOOTHING = 0.95  # β, weight of the past per update
EMA_WINDOW = round(1 / (1 - EMA_SMOOTHING))  # effective number of updates the EMA averages, 1 / (1 - β) = 20
RAW_ALPHA, RAW_LINEWIDTH = 0.35, 0.7  # the per-update curve behind its EMA
CHANNEL_MARKERS = ("o", "s", "^", "D", "v", "P", "X")  # one per channel, readable in grayscale print
ROW_PANEL_SIZE = (3.3, 2.7)  # inches per column of a `row_panels` figure, e.g. one training or validation panel
ROW_COLUMNS = 4  # columns of every monitor figure, plots fill them from the left
X_TICKS = 4  # x ticks per panel at most, keeps narrow panels readable


@contextmanager
def plot_theme(dark: bool) -> Iterator[None]:
    """matplotlib rc context for the monitor: transparent background, theme text/grid colors, theme colorway."""
    with THEME_LOCK, mpl.rc_context(MONITOR_THEMES[dark]):
        yield


def svg_bytes(figure: Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="svg")
    return buffer.getvalue()


def exponential_average(values: np.ndarray, smoothing: float = EMA_SMOOTHING) -> np.ndarray:
    """Debiased exponential moving average, in log space when all finite values are positive, NaN stays a gap."""
    is_finite = np.isfinite(values)
    is_positive = bool(is_finite.any()) and bool((values[is_finite] > 0).all())
    series = np.log(values, where=is_finite, out=np.full(values.shape, np.nan)) if is_positive else values
    averages = np.full(values.shape, np.nan)
    average = weight = 0.0
    for index in np.flatnonzero(is_finite):
        average = smoothing * average + (1 - smoothing) * series[index]
        weight = smoothing * weight + (1 - smoothing)
        averages[index] = average / weight
    return np.exp(averages) if is_positive else averages


def training_curves(aggregates: pl.DataFrame, samples_per_update: int | None = None) -> Figure:
    """MSE, spectral loss and KL per update: mean over the update's samples plus its EMA.

    Parameters
    ----------
    aggregates : pl.DataFrame
        `TRAINING_AGGREGATES_SCHEMA`, optional `run` column -> one color per run.
    samples_per_update : int | None
        `TrainConfig.designs_per_update`, named in the legend, None leaves the count out.
    """
    samples = f"n={samples_per_update}" if samples_per_update else "its samples"
    return _update_curves(aggregates, TRAIN_PANELS, "Training metrics", f"per update, mean over {samples}")


def optimizer_curves(aggregates: pl.DataFrame) -> Figure:
    """Learning rate, gradient norm and gradient to weight norm per update, the gradient norms plus their EMA.

    Parameters
    ----------
    aggregates : pl.DataFrame
        `TRAINING_AGGREGATES_SCHEMA`, optional `run` column -> one color per run.
    """
    return _update_curves(aggregates, OPTIMIZER_PANELS, "Optimizer", "per update")


def _update_curves(aggregates: pl.DataFrame, update_panels: Sequence[Panel], title: str, raw_label: str) -> Figure:
    """One row of `update_panels` over the training aggregates, titled `title`, `raw_label` names the unsmoothed line."""
    figure, axes = row_panels([1] * len(update_panels))
    rows = typed(aggregates, TRAINING_AGGREGATES_SCHEMA).sort("update", maintain_order=True)
    _draw_training(axes, rows, update_panels, raw_label)
    figure_title(figure, title)
    return figure


def _draw_training(axes: Sequence[Axes], means: pl.DataFrame, update_panels: Sequence[Panel], raw_label: str) -> None:
    """One panel per `update_panels` entry, the line kinds and runs in one legend row below the panels.

    - One run: per-update mean in TUM light blue, its EMA and unsmoothed panels in TUM blue
    - Several runs: one color per run, the per-update mean faint, the EMA solid
    """
    runs = by_run(means)
    is_comparison = runs[0][0] is not None
    ema_label = f"EMA, n={EMA_WINDOW}, β={EMA_SMOOTHING}"
    for axis, panel in zip(axes, update_panels, strict=False):
        for index, (run, rows) in enumerate(runs):
            updates, values = rows["update"].to_numpy(), float_values(rows, panel.column)
            color = cycle_color(index) if is_comparison else TUM_BLUE
            if not panel.is_smoothed:
                axis.plot(updates, values, color=color, label=run)
                continue
            raw_color, raw_alpha = (color, RAW_ALPHA) if is_comparison else (TUM_LIGHT_BLUE, light_blue_alpha())
            axis.plot(
                updates, values, color=raw_color, alpha=raw_alpha, linewidth=RAW_LINEWIDTH, label=None if is_comparison else raw_label
            )
            axis.plot(updates, exponential_average(values), color=color, label=run if is_comparison else ema_label)
        label_axis(axis, panel.title, "update", panel.y_label, float_values(means, panel.column), panel.log_y)
        if means.height:
            integer_x_ticks(axis)
    if not means.height:
        return
    legend_axis = next((axis for axis, panel in zip(axes, update_panels, strict=False) if panel.is_smoothed), axes[0])
    handles, labels = legend_axis.get_legend_handles_labels()  # an unsmoothed panel has no per-update and EMA lines
    if is_comparison:  # line kinds as neutral proxies after the runs: thin faint per update, solid EMA
        neutral = mpl.rcParams["text.color"]
        handles += [Line2D([], [], color=neutral, alpha=RAW_ALPHA, linewidth=RAW_LINEWIDTH), Line2D([], [], color=neutral)]
        labels += [raw_label, ema_label]
    axes[0].figure.legend(handles, labels, loc="outside lower center", ncols=len(labels))


def validation_curves(aggregates: pl.DataFrame) -> Figure:
    """NRMSE, MSE, spectral loss and KL medians over designs per update and channel.

    - Without `run`: one row with a panel per metric, one color and marker per channel, channel legend in one row below
    - With `run`: one panel per metric and channel, one color per run

    Parameters
    ----------
    aggregates : pl.DataFrame
        `VALIDATION_AGGREGATES_SCHEMA`, optional `run` column -> one color per run.
    """
    medians = typed(aggregates, VALIDATION_AGGREGATES_SCHEMA).sort("update", "channel", maintain_order=True)
    if "run" not in medians.columns:
        figure, axes = row_panels([1] * len(VALIDATION_PANELS))
        _draw_validation(figure, axes, medians)
        figure_title(figure, "Validation metrics")
        return figure
    channels, runs = present_channels(medians), by_run(medians)
    figure, axes = panels(len(VALIDATION_PANELS) * len(channels), columns=len(channels), panel_size=(3.4, 2.6))
    for row, panel in enumerate(VALIDATION_PANELS):
        for position, channel in enumerate(channels):
            axis = axes[row * len(channels) + position]
            for index, (run, rows) in enumerate(runs):
                channel_rows = rows.filter(pl.col("channel") == channel)
                updates = channel_rows["update"].to_numpy()
                axis.plot(updates, float_values(channel_rows, panel.column), marker="o", color=cycle_color(index), label=run)
            channel_values = float_values(medians.filter(pl.col("channel") == channel), panel.column)
            label_axis(axis, f"{panel.title} | {channel}", "update", panel.y_label, channel_values, panel.log_y)
            integer_x_ticks(axis)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=len(runs), title="run, median over designs")
    figure_title(figure, "Validation metrics")
    return figure


def _draw_validation(figure: Figure, axes: Sequence[Axes], medians: pl.DataFrame) -> None:
    """One panel per `VALIDATION_PANELS` metric, one color and marker per channel, the channel legend in one row below."""
    channels = present_channels(medians)
    for axis, panel in zip(axes, VALIDATION_PANELS, strict=False):
        for channel in channels:
            rows = medians.filter(pl.col("channel") == channel)
            axis.plot(rows["update"].to_numpy(), float_values(rows, panel.column), label=channel, **channel_style(channel))
        label_axis(axis, panel.title, "update", panel.y_label, float_values(medians, panel.column), panel.log_y)
        if channels:
            integer_x_ticks(axis)
    if channels:
        figure.legend(
            *axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=len(channels), title="channel, median over designs"
        )


def channel_style(channel: str) -> dict[str, str]:
    """Fixed color and marker of a channel symbol, the same in every figure."""
    index = CHANNEL_SYMBOLS.index(channel)
    return {"color": cycle_color(index), "marker": CHANNEL_MARKERS[index]}


def present_channels(frame: pl.DataFrame) -> list[str]:
    """Channel symbols in `frame`, in model order."""
    present = set(frame["channel"].cast(pl.String))
    return [symbol for symbol in CHANNEL_SYMBOLS if symbol in present]


def column_lines(frame: pl.DataFrame) -> Figure:
    """One panel per `column_groups` group of a numeric frame, first column as x, nulls and NaN as line gaps."""
    x_name = frame.columns[0]
    frame = frame.drop_nulls(x_name)
    groups = column_groups(frame.columns)
    figure, axes = panels(len(groups), columns=min(len(groups), 3))
    x = float_values(frame, x_name)
    for axis, (group, columns) in zip(axes, groups, strict=False):
        for column in columns:
            axis.plot(x, float_values(frame, column), label=series_label(group, column))
        values = np.concatenate([float_values(frame, column) for column in columns])
        label_axis(axis, group, x_name, group, values)
        if frame[x_name].dtype.is_integer():
            integer_x_ticks(axis)
        if len(columns) > 1:
            axis.legend()
    return figure


def validation_error_comparison(aggregates: pl.DataFrame, model: str, grid_shape: Sequence[int]) -> Figure | None:
    """Mean and median NRMSE over designs per channel, zero-shot (update 0) vs `model` at its latest validation, two columns each.

    - Dumbbell charts: one row per channel, both errors as dots joined by a line, both values beside the chart
    - None before a validation after update 0

    Parameters
    ----------
    aggregates : pl.DataFrame
        `VALIDATION_AGGREGATES_SCHEMA` rows of one run.
    grid_shape : Sequence[int]
        Voxels along x, y, z, named in the title, e.g. `Grid.shape`.
    """
    aggregates = typed(aggregates, VALIDATION_AGGREGATES_SCHEMA)
    last_update = aggregates["update"].max() if aggregates.height else 0
    if not last_update:
        return None
    channels = present_channels(aggregates)
    figure, axes = row_panels([2, 2])
    for axis, statistic in zip(axes, ("mean", "median"), strict=True):
        zero_shot, tuned = (_channel_values(aggregates, update, channels, f"nrmse/{statistic}") for update in (0, last_update))
        _draw_dumbbell(axis, channels, zero_shot, tuned)
        axis.set(title=f"{statistic.capitalize()} over designs", xlabel="NRMSE [%]")
    figure_title(figure, f"Validation error · zero-shot vs. {model} on {'x'.join(map(str, grid_shape))}")
    handles = [Line2D([], [], color=color, marker="o", linestyle="", markersize=6) for color in (TUM_BLUE, REFERENCE_GRAY)]
    figure.legend(handles, [f"{model}, update {last_update}", "zero-shot"], loc="outside lower center", ncols=2)
    return figure


def _draw_dumbbell(axis: Axes, channels: Sequence[str], zero_shot: np.ndarray, tuned: np.ndarray) -> None:
    """One row per channel, top to bottom: both errors as dots on a line, the values right of the axis, model first.

    - Log x axis when the errors span decades (`is_log_scale`), e.g. a zero-shot mean far above the rest
    """
    rows = np.arange(len(channels))
    axis.hlines(rows, np.minimum(zero_shot, tuned), np.maximum(zero_shot, tuned), color=mpl.rcParams["grid.color"], linewidth=2.5)
    axis.scatter(zero_shot, rows, color=REFERENCE_GRAY, s=36, zorder=2)
    axis.scatter(tuned, rows, color=TUM_BLUE, s=36, zorder=3)
    for row, (before, after) in enumerate(zip(zero_shot, tuned, strict=True)):
        axis.annotate(_short_number(after), (1.01, row), xycoords=("axes fraction", "data"), va="center", color=TUM_BLUE, fontsize=8)
        axis.annotate(
            _short_number(before), (1.08, row), xycoords=("axes fraction", "data"), va="center", color=REFERENCE_GRAY, fontsize=8
        )
    axis.set_yticks(rows, channels)
    axis.invert_yaxis()  # first channel on top
    if is_log_scale([*zero_shot, *tuned]):
        axis.set_xscale("log")
        plain_log_ticks(axis.xaxis)
    else:
        axis.set_xlim(left=0)
    axis.grid(axis="y", visible=False)


def _short_number(value: float) -> str:
    """Three significant digits, whole numbers from 1000 on instead of exponent notation."""
    return f"{value:.0f}" if abs(value) >= 1000 else f"{value:.3g}"  # noqa: PLR2004


def _channel_values(aggregates: pl.DataFrame, update: int, channels: Sequence[str], column: str) -> np.ndarray:
    """`column` of each of `channels` at `update`, in their order."""
    rows = aggregates.filter(pl.col("update") == update)
    return np.array([rows.filter(pl.col("channel") == channel)[column].item() for channel in channels])


def run_model_and_grid(state: Mapping[str, Any]) -> tuple[str, tuple[int, int, int]]:
    """Model label and voxel grid of a fine-tuning run from its tracker state `configs`, e.g. ("LoRA-32", (400, 80, 16))."""
    configs = state["configs"]
    finetune_state = configs["TadpoleConfig"]["encoder_state"]
    model = f"LoRA-{finetune_state}" if isinstance(finetune_state, int) else str(finetune_state)
    return model, Grid(latent_scale=configs["DataConfig"]["grid"]["latent_scale"]).shape


TableFigure = Callable[[pl.DataFrame, Mapping[str, Any]], Figure | None]  # (table rows, tracker state) -> figure, None = nothing yet

# Monitor figures per tracker table in order, () = drawn through its aggregates, unlisted tables get one `column_lines` figure
TABLE_FIGURES: dict[str, tuple[TableFigure, ...]] = {
    METRICS_TRAINING: (),
    METRICS_VALIDATION: (),
    METRICS_TRAINING_AGGREGATES: (
        lambda rows, state: training_curves(rows, state["configs"]["TrainConfig"]["designs_per_update"]),
        lambda rows, _: optimizer_curves(rows),
    ),
    METRICS_VALIDATION_AGGREGATES: (
        lambda rows, _: validation_curves(rows),
        lambda rows, state: validation_error_comparison(rows, *run_model_and_grid(state)),
    ),
}


def figure_count(table: str) -> int:
    """Monitor figures of `table`, 0 for tables drawn through their aggregates."""
    return len(TABLE_FIGURES.get(table, (_numeric_column_lines,)))


def figure_for_table(table: str, frame: pl.DataFrame, state: Mapping[str, Any], index: int = 0) -> Figure | None:
    """Monitor figure `index` of a tracker table, None if empty or nothing to draw yet, see `TABLE_FIGURES`.

    - `state`: the run's tracker snapshot, e.g. its configs name the model and grid
    """
    if frame.height == 0:
        return None
    return TABLE_FIGURES.get(table, (_numeric_column_lines,))[index](frame, state)


def _numeric_column_lines(frame: pl.DataFrame, _state: Mapping[str, Any]) -> Figure | None:
    """`column_lines` of the numeric columns, the first column is x, other columns are skipped."""
    numeric = frame.select(column for column, dtype in frame.schema.items() if dtype.is_numeric() or dtype == pl.Null)
    if numeric.width < 2 or numeric.columns[0] != frame.columns[0]:  # noqa: PLR2004, x plus at least one series
        return None
    return column_lines(numeric)


def design_extremes(designs: pl.DataFrame, channel_axes: pl.DataFrame) -> Figure:
    """Per channel each design statistic sorted descending: at rank r, r designs exceed it, a spike at the left = few designs far off.

    Parameters
    ----------
    designs : pl.DataFrame
        `DESIGN_EXTREMES_SCHEMA`, `extreme` is max |value| for `is_signed` channels, max otherwise.
    channel_axes : pl.DataFrame
        `CHANNEL_AXES_SCHEMA`, a dashed line at `rule_limit` where set.
    """
    designs = typed(designs, DESIGN_EXTREMES_SCHEMA)
    settings = channel_settings(channel_axes)
    figure, axes = panels(len(CHANNEL_SYMBOLS), columns=4, panel_size=(4.0, 3.2))
    statistics = (("extreme", "max", cycle_color(1)), ("q0.999", "q0.999", cycle_color(0)), ("median", "median", REFERENCE_GRAY))
    rule_color = mpl.rcParams["text.color"]
    for axis, channel in zip(axes, CHANNEL_SYMBOLS, strict=False):
        offset = settings[channel]["offset"]
        rows = designs.filter(pl.col("channel") == channel)
        for column, label, color in statistics:
            axis.plot(np.sort(rows[column].drop_nulls().to_numpy())[::-1] - offset, color=color, label=label)
        if settings[channel]["rule_limit"] is not None:
            axis.axhline(settings[channel]["rule_limit"] - offset, color=rule_color, linestyle="--", linewidth=1)
        axis.set_yscale("asinh", linear_width=settings[channel]["linear_width"])
        prefix = "max |·| " if settings[channel]["is_signed"] else ""
        axis.set(title=prefix + CHANNEL_LABELS[channel], xlabel="design rank [designs]", ylabel=offset_label(channel, offset))
    handles, labels = axes[0].get_legend_handles_labels()
    rule = Line2D([], [], color=rule_color, linestyle="--")
    axes[-1].legend([*handles, rule], [*labels, "outlier rule"], loc="center", fontsize=11)
    return figure


def value_histograms(histograms: pl.DataFrame, quantiles: pl.DataFrame, channel_axes: pl.DataFrame) -> Figure:
    """Share of fluid voxels per bin of all and curated designs, x linear near `offset` and logarithmic beyond.

    Parameters
    ----------
    histograms : pl.DataFrame
        `HISTOGRAM_SCHEMA`, ascending bins per channel.
    quantiles : pl.DataFrame
        `QUANTILE_SCHEMA`, curated quantiles drawn as vertical lines.
    channel_axes : pl.DataFrame
        `CHANNEL_AXES_SCHEMA`, `offset` and `linear_width` set the x axis.
    """
    histograms, quantiles = typed(histograms, HISTOGRAM_SCHEMA), typed(quantiles, QUANTILE_SCHEMA)
    settings = channel_settings(channel_axes)
    figure, axes = panels(len(CHANNEL_SYMBOLS), columns=2, panel_size=(7.5, 2.5))
    for axis, channel in zip(axes, CHANNEL_SYMBOLS, strict=False):
        offset = settings[channel]["offset"]
        rows = histograms.filter(pl.col("channel") == channel).sort("bin_low")
        edges = np.append(rows["bin_low"].to_numpy(), rows["bin_high"][-1]) - offset
        for column, color, label in (("share_all", REFERENCE_GRAY, "all designs"), ("share_curated", cycle_color(0), "curated")):
            share = float_values(rows, column)
            axis.stairs(np.where(share > 0, share, np.nan), edges, color=color, label=label, linewidth=1.2)
        occupied = np.flatnonzero(float_values(rows, "share_all") > 0)
        axis.set_xlim(edges[max(occupied[0] - 5, 0)], edges[min(occupied[-1] + 6, len(edges) - 1)])
        for value in quantiles.filter(pl.col("channel") == channel)["value"]:
            axis.axvline(value - offset, color=cycle_color(1), linewidth=0.8)
        axis.set_xscale("asinh", linear_width=settings[channel]["linear_width"])
        axis.set_yscale("log")
        axis.set(title=CHANNEL_LABELS[channel], xlabel=offset_label(channel, offset), ylabel="share of voxels [fraction]")
    quantile_names = ", ".join("median" if q == 0.5 else f"q{q:g}" for q in sorted(set(quantiles["quantile"])))  # noqa: PLR2004
    handles, labels = axes[0].get_legend_handles_labels()
    quantile_line = Line2D([], [], color=cycle_color(1))
    axes[-1].legend([*handles, quantile_line], [*labels, f"curated {quantile_names}"], loc="center", fontsize=11)
    return figure


def normalized_density(histograms: pl.DataFrame, channel_axes: pl.DataFrame) -> Figure:
    """Curated voxel density per normalized unit, ±1 marked, x linear within ±1 and logarithmic beyond.

    Parameters
    ----------
    histograms : pl.DataFrame
        `NORMALIZED_HISTOGRAM_SCHEMA`, ascending bins per channel, `share` of the channel's curated fluid voxels.
    channel_axes : pl.DataFrame
        `CHANNEL_AXES_SCHEMA`, `is_log` marks log10-normalized channels in the titles.
    """
    histograms = typed(histograms, NORMALIZED_HISTOGRAM_SCHEMA)
    settings = channel_settings(channel_axes)
    figure, axes = panels(len(CHANNEL_SYMBOLS), columns=2, panel_size=(7.5, 2.3))
    for axis, channel in zip(axes, CHANNEL_SYMBOLS, strict=False):
        rows = histograms.filter(pl.col("channel") == channel).sort("bin_low")
        occupied = np.flatnonzero(float_values(rows, "share") > 0)
        rows = rows.slice(occupied[0], occupied[-1] - occupied[0] + 1)
        edges = np.append(rows["bin_low"].to_numpy(), rows["bin_high"][-1])
        density = float_values(rows, "share") / np.diff(edges)  # bins differ in width after normalization
        axis.stairs(np.where(density > 0, density, np.nan), edges, color=cycle_color(0), label="curated designs")
        for bound in (-1, 1):
            axis.axvline(bound, color=cycle_color(1), linewidth=0.8)
        axis.set_xscale("symlog", linthresh=1)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_yscale("log")
        title = CHANNEL_LABELS[channel] + (" (log10)" if settings[channel]["is_log"] else "")
        axis.set(title=title, xlabel="value [normalized]", ylabel="density [1/normalized]")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend([*handles, Line2D([], [], color=cycle_color(1))], [*labels, "±1"], loc="center", fontsize=11)
    return figure


def plot_slices(  # noqa: PLR0913
    reference: FlowField,
    prediction: FlowField,
    grid: Grid,
    sample_index: int = 0,
    z_index: int | None = None,
    channels: Sequence[Channel] = CHANNELS,
) -> Figure:
    """Top-view slices of reference, prediction and |error| per channel, `z_index=None` = half height."""
    z_index = grid.shape[2] // 2 if z_index is None else z_index
    extent_mm = (0.0, 1000 * grid.extent[0], 0.0, 1000 * grid.extent[1])
    reference_volumes, prediction_volumes, fluid = _sample_volumes(reference, prediction, sample_index)
    figure = Figure(figsize=(15, 1.3 * len(channels)), layout="constrained")
    axes = figure.subplots(len(channels), 3, squeeze=False)
    for row, channel in enumerate(channels):
        channel_index = CHANNELS.index(channel)
        reference_slice, prediction_slice = reference_volumes[channel_index], prediction_volumes[channel_index]
        color_range = {"vmin": reference_slice[fluid].min(), "vmax": reference_slice[fluid].max()}
        slice_panels = (
            ("reference", reference_slice, "plasma", color_range),
            ("prediction", prediction_slice, "plasma", color_range),
            ("|error|", np.abs(prediction_slice - reference_slice), "magma", {}),
        )
        for axis, (title, volume, colormap_name, limits) in zip(axes[row], slice_panels, strict=True):
            colormap = colormaps[colormap_name].with_extremes(bad=SOLID_COLOR)
            image = axis.imshow(
                np.where(fluid, volume, np.nan)[:, :, z_index].T, origin="lower", extent=extent_mm, cmap=colormap, **limits
            )
            axis.set_title(f"{title} | {channel.value}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.02, pad=0.01)
    figure.suptitle(f"design {int(reference.design_id[sample_index])}, z index {z_index}")
    return figure


def column_groups(names: Sequence[str]) -> list[tuple[str, list[str]]]:
    """(group, its columns) of the columns after the x column `names[0]`, grouped by name up to the last `/`.

    - `mse` -> its own group, `nrmse/median/Vx` and `nrmse/median/Vy` -> one group `nrmse/median`
    """
    groups: dict[str, list[str]] = {}
    for name in names[1:]:
        groups.setdefault(name.rpartition("/")[0] or name, []).append(name)
    return list(groups.items())


def series_label(group: str, column: str) -> str:
    """`nrmse/median/Vx` in group `nrmse/median` -> `Vx`, a single-column group keeps its name."""
    return column.removeprefix(f"{group}/")


def is_log_scale(values: Iterable[float | None]) -> bool:
    """Whether the finite `values` are all positive and span more than `LOG_SCALE_RATIO`."""
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return bool(finite) and min(finite) > 0 and max(finite) / min(finite) > LOG_SCALE_RATIO


def offset_label(channel: str, offset: float) -> str:
    """Axis label of channel values minus `offset`, e.g. `T - 293.15 [K]`."""
    name, _, unit = CHANNEL_LABELS[channel].partition(" [")
    return f"{name} - {offset:g} [{unit}" if offset else CHANNEL_LABELS[channel]


def typed(frame: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """`frame` cast to `schema` plus its optional `run` column, ColumnNotFoundError for a missing column."""
    run = [pl.col("run").cast(pl.String)] if "run" in frame.columns else []
    return frame.select(*(pl.col(name).cast(dtype) for name, dtype in schema.items()), *run)


def by_run(frame: pl.DataFrame) -> list[tuple[str | None, pl.DataFrame]]:
    """(run, its rows) in order of first appearance, a single (None, frame) without a `run` column."""
    if "run" not in frame.columns:
        return [(None, frame)]
    return [(run, rows) for (run,), rows in frame.partition_by("run", maintain_order=True, as_dict=True).items()]


def channel_settings(channel_axes: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Channel symbol -> its `CHANNEL_AXES_SCHEMA` row."""
    return {row["channel"]: row for row in typed(channel_axes, CHANNEL_AXES_SCHEMA).iter_rows(named=True)}


def float_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    """`column` as float numpy, nulls as NaN, so lines break there."""
    return frame[column].cast(pl.Float64).to_numpy()


def row_panels(spans: Sequence[int], columns: int | None = None) -> tuple[Figure, list[Axes]]:
    """Figure of one row of `columns` equal columns of `ROW_PANEL_SIZE`, one axis per span from the left, the rest empty.

    - `columns=ROW_COLUMNS` gives every monitor figure the same width, so the page shows all columns at one scale
    - `columns=None` fits the row to the spans, e.g. for a standalone PNG

    Examples
    --------
    >>> figure, axes = row_panels([1, 1, 1, 1])  # four metric panels
    >>> figure, (comparison,) = row_panels([2], columns=ROW_COLUMNS)  # half a monitor row, room for two more panels
    """
    columns = sum(spans) if columns is None else columns
    figure = Figure(figsize=(columns * ROW_PANEL_SIZE[0], ROW_PANEL_SIZE[1]), layout="constrained")
    grid = figure.add_gridspec(1, columns)
    starts = np.cumsum([0, *spans[:-1]])
    return figure, [figure.add_subplot(grid[0, start : start + span]) for start, span in zip(starts, spans, strict=True)]


def panels(count: int, columns: int, panel_size: tuple[float, float] = (5.0, 3.0)) -> tuple[Figure, list[Axes]]:
    """Figure with `count` panels in rows of `columns`, axes beyond `count` hidden, e.g. for a shared legend."""
    rows = math.ceil(count / columns)
    figure = Figure(figsize=(columns * panel_size[0], rows * panel_size[1]), layout="constrained")
    axes = list(figure.subplots(rows, columns, squeeze=False).flat)
    for axis in axes[count:]:
        axis.axis("off")
    return figure, axes


def light_blue_alpha() -> float:
    """Opacity of `TUM_LIGHT_BLUE` behind `TUM_BLUE` in the current style, faded under the dark monitor theme."""
    return DARK_LIGHT_BLUE_ALPHA if mpl.rcParams["text.color"] == DARK_TEXT else 1.0


def cycle_color(index: int) -> str:
    """Color `index` of the current colorway, resolved now, a `"C<index>"` string would resolve at save time outside the style."""
    return to_hex(f"C{index}")


def figure_title(figure: Figure, title: str) -> None:
    """Left-aligned figure title above the panel titles, one size up and bold, e.g. "Training metrics"."""
    figure.suptitle(title, x=0.0, ha="left", fontsize=mpl.rcParams["axes.titlesize"] + 1, fontweight="bold")


def label_axis(  # noqa: PLR0913, axis, texts, values and the scale override
    axis: Axes, title: str, x_label: str, y_label: str, values: np.ndarray, log_y: bool | None = None
) -> None:
    """Set title and labels, log y when `log_y`, or when `values` span decades for `log_y=None`."""
    axis.set(title=title, xlabel=x_label, ylabel=y_label)
    if is_log_scale(values.tolist()) if log_y is None else log_y:
        axis.set_yscale("log")
        plain_log_ticks(axis.yaxis)
    else:  # plain tick values like 5e-05 or 1500, no offset text above the axis colliding with the title
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _short_number(value)))


def plain_log_ticks(axis: "mpl.axis.Axis") -> None:
    """Log axis labelled at 1, 2 and 5 per decade as plain numbers (0.002, 50, 10000), unlabelled minor ticks."""
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda value, _: _short_number(value)))
    axis.set_minor_formatter(NullFormatter())


def integer_x_ticks(axis: Axes) -> None:
    """At most `X_TICKS` whole-number x ticks for counts like updates, call after plotting, a single update gets one tick."""
    if axis.dataLim.width == 0:
        axis.set_xticks([axis.dataLim.x0])
    else:
        axis.xaxis.set_major_locator(MaxNLocator(X_TICKS, integer=True, steps=[1, 2, 5, 10]))


def _sample_volumes(reference: FlowField, prediction: FlowField, sample_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical (C, X, Y, Z) reference and prediction plus (X, Y, Z) fluid mask of one sample as numpy."""
    reference_sample, prediction_sample = (_physical(flowfield[sample_index]) for flowfield in (reference, prediction))
    return (
        reference_sample.fields.detach().float().cpu().numpy(),
        prediction_sample.fields.detach().float().cpu().numpy(),
        reference_sample.mask.cpu().numpy(),
    )


def _physical(flowfield: FlowField) -> FlowField:
    return flowfield if flowfield.stats is None else flowfield.denormalize()
