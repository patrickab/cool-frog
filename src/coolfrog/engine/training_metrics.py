"""Metrics of one fine-tuning run: windowed training means, validation medians, dashboard layout and table export."""

from collections.abc import Collection, Iterator, Mapping
from itertools import cycle
import time

import polars as pl

from coolfrog.lib.config import TrainConfig
from coolfrog.lib.dashboard import PALETTE, ChartSpec, ColumnSpec, DashboardSpec, SeriesSpec, scientific
from coolfrog.lib.datatypes import CHANNELS
from coolfrog.lib.log import get_logger
from coolfrog.lib.metrics import MetricTracker, WindowMean

logger = get_logger(__name__)

MSE, SPECTRAL, KL = "train/mse", "train/spectral_loss", "train/kl"
LEARNING_RATE, SECONDS_PER_UPDATE = "train/learning_rate", "train/seconds_per_update"
NRMSE, SPECTRAL_LOSS = "nrmse", "spectral_loss"  # keys of `evaluate`
TRAINING_PROGRESS, VALIDATION_PROGRESS = "Training", "Validation"  # progress bar labels


def validation_name(metric: str, channel: str | None = None) -> str:
    """Series name of a validation metric, `val/nrmse` for the channel mean, `val/nrmse/p [Pa]` for one channel."""
    return f"val/{metric}" if channel is None else f"val/{metric}/{channel}"


class TrainingMetrics:
    """Logging, display and export of one `fine_tune_autoencoder` run, keyed by update.

    - Train values are averaged over `log_every` updates, plus the final one
    - Validation logs per-channel medians over the validation designs and their channel mean
    - Progress bars: updates done, designs done of the running or last validation
    - `tracker` holds everything, `spec` tells `run_dashboard` how to show it

    Examples
    --------
    >>> metrics = TrainingMetrics(TrainConfig(updates=100))
    >>> metrics.log_update(1, mse=0.1, spectral=0.4, kl=3.0, learning_rate=5e-5)
    >>> metrics.to_frame().write_parquet(path_run / "metrics.parquet")
    """

    def __init__(self, config: TrainConfig, title: str = "Fine-tuning", log_every: int = 10) -> None:
        self.config = config
        self.log_every = log_every
        self.tracker = MetricTracker()
        self.tracker.set_progress(TRAINING_PROGRESS, 0, config.updates)
        self.spec = self._create_spec(title)
        self._window = WindowMean()
        self._tick = time.perf_counter()

    def log_update(self, update: int, mse: float, spectral: float, kl: float, learning_rate: float) -> None:
        """Record one optimizer update and advance the training bar, KeyboardInterrupt when the dashboard asked to stop."""
        self.tracker.set_progress(TRAINING_PROGRESS, update, self.config.updates)
        now = time.perf_counter()
        self._window.add({MSE: mse, SPECTRAL: spectral, KL: kl, LEARNING_RATE: learning_rate, SECONDS_PER_UPDATE: now - self._tick})
        self._tick = now
        if update % self.log_every == 0 or update == self.config.updates:
            self.tracker.log(update, self._window.flush())

    def track_validation[T](self, designs: Collection[T]) -> Iterator[T]:
        """Yield validation `designs` while advancing the validation bar, KeyboardInterrupt after a stop request."""
        self.tracker.set_progress(VALIDATION_PROGRESS, 0, len(designs))
        for completed, design in enumerate(designs, start=1):
            yield design
            self.tracker.set_progress(VALIDATION_PROGRESS, completed, len(designs))

    def log_validation(self, update: int, per_design_tables: Mapping[str, pl.DataFrame]) -> None:
        """Record the median over designs per channel and its mean over channels of each `evaluate` table.

        - Median, as a few designs dominate the mean, e.g. p NRMSE of designs with nearly constant pressure
        - Resets the update timer, validation time does not count as update time
        """
        values: dict[str, float] = {}
        for metric, per_design in per_design_tables.items():
            channel_medians = {channel.value: per_design[channel.value].median() for channel in CHANNELS}
            values[validation_name(metric)] = sum(channel_medians.values()) / len(channel_medians)
            values |= {validation_name(metric, channel): median for channel, median in channel_medians.items()}
        self.tracker.log(update, values)
        logger.info(
            "update %d validation NRMSE %.3f %% spectral loss %.3e",
            update,
            values[validation_name(NRMSE)],
            values[validation_name(SPECTRAL_LOSS)],
        )
        self._tick = time.perf_counter()

    def to_frame(self) -> pl.DataFrame:
        """All series, one row per update, null where a series skipped it."""
        return self.tracker.to_frame().rename({"step": "update"})

    def _create_spec(self, title: str) -> DashboardSpec:
        config = self.config

        def channel_lines(metric: str) -> tuple[SeriesSpec, ...]:
            return tuple(
                SeriesSpec(validation_name(metric, channel.value), channel.value, color)
                for channel, color in zip(CHANNELS, cycle(PALETTE), strict=False)
            )

        return DashboardSpec(
            title=title,
            charts=(
                ChartSpec(
                    "Train MSE and spectral loss",
                    (SeriesSpec(MSE, "mse", PALETTE[0]), SeriesSpec(SPECTRAL, "spectral", PALETTE[2], side="right")),
                    y_label="mse",
                ),
                ChartSpec("Validation NRMSE [%], median", channel_lines(NRMSE), y_label="nrmse"),
                ChartSpec("Validation spectral loss, median", channel_lines(SPECTRAL_LOSS), y_label="spectral"),
            ),
            columns=(
                ColumnSpec("Train MSE", MSE, "magenta"),
                ColumnSpec("Train spectral", SPECTRAL, "magenta"),
                ColumnSpec("KL", KL, "magenta"),
                ColumnSpec("LR", LEARNING_RATE, "yellow"),
                ColumnSpec("Val NRMSE %", validation_name(NRMSE), "green", "{:.3f}"),
                ColumnSpec("Val spectral", validation_name(SPECTRAL_LOSS), "green"),
                ColumnSpec("s/update", SECONDS_PER_UPDATE, "", "{:.2f}"),
            ),
            step_header="Update",
            progress_bars=(TRAINING_PROGRESS, VALIDATION_PROGRESS),
            summary=(
                f"[dim]batch[/] {config.crops_per_step}x{config.accumulation_steps} crops · "
                f"[dim]lr[/] {scientific(config.learning_rate)} ({config.warmup_updates} warmup) · "
                f"[dim]wd[/] {scientific(config.weight_decay)} · [dim]kl weight[/] {scientific(config.kl_weight)} · "
                f"[dim]validate every[/] {config.validate_every}"
            ),
        )
