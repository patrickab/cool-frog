"""Tables of one fine-tuning run: windowed training means, validation per design and channel, its aggregates."""

from collections.abc import Collection, Iterator, Mapping
import logging
import time

import polars as pl

from coolfrog.engine.model_evaluation import MSE, SPECTRAL_LOSS
from coolfrog.lib.config import TrainConfig
from coolfrog.lib.datatypes import CHANNELS
from coolfrog.lib.metrics import Tracker, WindowMean

logger = logging.getLogger(__name__)

TRAIN, VALIDATION, VALIDATION_AGGREGATES = "train", "validation", "validation_aggregates"  # table names
STATISTICS = ("median", "mean")  # aggregates over designs, median first as a few bad designs dominate the mean
TRAINING_PROGRESS, VALIDATION_PROGRESS = "Training", "Validation"  # progress bar labels


def aggregate_name(metric: str, statistic: str, channel: str) -> str:
    """Column of `validation_aggregates`, e.g. `nrmse/median/p` for channel `p [Pa]`, one chart per `nrmse/median`.

    - Without the unit, its `/` (`m/s`) would split the chart group
    """
    return f"{metric}/{statistic}/{channel.partition(' [')[0]}"


def validation_rows(update: int, per_design_tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """`evaluate` tables as one row per design and channel: `update`, `design_id`, `channel`, one column per metric."""
    channels = [channel.value for channel in CHANNELS]
    long_tables = [
        table.unpivot(on=channels, index="design_id", variable_name="channel", value_name=metric)
        for metric, table in per_design_tables.items()
    ]  # same design and channel order in every table, `evaluate` sorts by design_id
    metrics = pl.concat([long_tables[0], *(table.drop("design_id", "channel") for table in long_tables[1:])], how="horizontal")
    return metrics.select(pl.lit(update).alias("update"), pl.all())


def aggregate_row(update: int, rows: pl.DataFrame, metrics: Collection[str]) -> dict[str, float]:
    """One `validation_aggregates` row: every statistic of every metric per channel over the designs of `rows`."""
    per_channel = rows.group_by("channel", maintain_order=True).agg(
        getattr(pl.col(metric), statistic)().alias(f"{metric}/{statistic}") for metric in metrics for statistic in STATISTICS
    )
    aggregates: dict[str, float] = {"update": update}
    for record in per_channel.iter_rows(named=True):
        channel = record.pop("channel")
        aggregates |= {aggregate_name(*name.split("/"), channel): value for name, value in record.items()}
    return aggregates


class TrainingMetrics:
    """Logging of one `fine_tune_autoencoder` run into `tracker`, keyed by update.

    - `train` table: training values averaged over `log_every` updates, plus the final one
    - `validation` table: every metric per update, design and channel, aggregates are computed from it
    - `validation_aggregates` table: median and mean over the designs per metric and channel, one row per validation,
      computed once when the validation ends
    - Progress bars: updates done, designs done of the running or last validation

    Examples
    --------
    >>> metrics = TrainingMetrics(TrainConfig(updates=100), batch_size=8, tracker=Tracker())
    >>> metrics.log_update(1, mse=0.1, spectral=0.4, kl=3.0, learning_rate=5e-5)
    """

    def __init__(self, config: TrainConfig, batch_size: int, tracker: Tracker, log_every: int = 50) -> None:
        self.config = config
        self.batch_size = batch_size
        self.log_every = log_every
        self.tracker = tracker
        self.tracker.set_progress(TRAINING_PROGRESS, 0, config.updates)
        self._window = WindowMean()
        self._tick = time.perf_counter()

    def log_update(self, update: int, mse: float, spectral: float, kl: float, learning_rate: float) -> None:
        """Record one optimizer update and advance the training bar, KeyboardInterrupt when the dashboard asked to stop."""
        self.tracker.set_progress(TRAINING_PROGRESS, update, self.config.updates)
        now = time.perf_counter()
        self._window.add(
            {"mse": mse, "spectral_loss": spectral, "kl": kl, "learning_rate": learning_rate, "seconds_per_update": now - self._tick}
        )
        self._tick = now
        if update % self.log_every == 0 or update == self.config.updates:
            self.tracker.append(TRAIN, {"update": update, **self._window.flush()})

    def track_validation[T](self, designs: Collection[T]) -> Iterator[T]:
        """Yield validation `designs` while advancing the validation bar, training clock paused, KeyboardInterrupt on stop."""
        self.tracker.pause_progress(TRAINING_PROGRESS)
        try:
            self.tracker.set_progress(VALIDATION_PROGRESS, 0, len(designs))
            for completed, design in enumerate(designs, start=1):
                yield design
                self.tracker.set_progress(VALIDATION_PROGRESS, completed, len(designs))
        finally:
            self.tracker.resume_progress(TRAINING_PROGRESS)

    def log_validation(self, update: int, per_design_tables: Mapping[str, pl.DataFrame]) -> None:
        """Append every design and channel of the `evaluate` tables, then their aggregates.

        - Resets the update timer, validation time does not count as update time
        """
        rows = validation_rows(update, per_design_tables)
        aggregates = aggregate_row(update, rows, per_design_tables)
        self.tracker.extend(VALIDATION, rows)
        self.tracker.append(VALIDATION_AGGREGATES, aggregates)
        channel_mean = {
            metric: sum(aggregates[aggregate_name(metric, "median", channel.value)] for channel in CHANNELS) / len(CHANNELS)
            for metric in (MSE, SPECTRAL_LOSS)
        }
        logger.info("update %d validation median MSE %.3e spectral loss %.3e, mean over channels", update, *channel_mean.values())
        self._tick = time.perf_counter()
