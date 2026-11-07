"""Tables of one fine-tuning run: windowed training means and validation per design and channel."""

from collections.abc import Collection, Iterator, Mapping
import logging
import time

import polars as pl

from coolfrog.engine.model_evaluation import MSE, SPECTRAL_LOSS
from coolfrog.lib.config import TrainConfig
from coolfrog.lib.metrics import Tracker, WindowMean
from coolfrog.lib.plots import CHANNEL_LABELS, TRAIN, TRAIN_SCHEMA, VALIDATION, VALIDATION_SCHEMA

logger = logging.getLogger(__name__)

SCHEMAS = {TRAIN: TRAIN_SCHEMA, VALIDATION: VALIDATION_SCHEMA}  # tracker schemas of the tables, see `Tracker`
TRAINING_PROGRESS, VALIDATION_PROGRESS = "Training", "Validation"  # progress bar labels


def validation_rows(update: int, per_design_tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """`evaluate` tables as `VALIDATION_SCHEMA` rows, one per design and channel, channels as symbols like `p`."""
    symbols = {label: symbol for symbol, label in CHANNEL_LABELS.items()}
    long_tables = [
        table.rename(symbols).unpivot(on=list(symbols.values()), index="design_id", variable_name="channel", value_name=metric)
        for metric, table in per_design_tables.items()
    ]  # same design and channel order in every table, `evaluate` sorts by design_id
    metrics = pl.concat([long_tables[0], *(table.drop("design_id", "channel") for table in long_tables[1:])], how="horizontal")
    metrics = metrics.with_columns(update=pl.lit(update))
    return metrics.select(pl.col(name).cast(dtype) for name, dtype in VALIDATION_SCHEMA.items())


class TrainingMetrics:
    """Logging of one `fine_tune_autoencoder` run into `tracker`, keyed by update.

    - `train` table (`TRAIN_SCHEMA`): training values averaged over `log_every` updates, plus the final one
    - `validation` table (`VALIDATION_SCHEMA`): every metric per update, design and channel, aggregated by its readers
    - Progress bars: updates done, designs done of the running or last validation

    Examples
    --------
    >>> metrics = TrainingMetrics(TrainConfig(updates=100), batch_size=8, tracker=Tracker(schemas=SCHEMAS))
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
        """Append every design and channel of the `evaluate` tables, log the channel mean of their medians over designs.

        - Resets the update timer, validation time does not count as update time
        """
        rows = validation_rows(update, per_design_tables)
        self.tracker.extend(VALIDATION, rows)
        medians = rows.group_by("channel").agg(pl.col(MSE, SPECTRAL_LOSS).median())
        channel_mean = medians.select(pl.col(MSE, SPECTRAL_LOSS).mean()).row(0)
        logger.info("update %d validation median MSE %.3e spectral loss %.3e, mean over channels", update, *channel_mean)
        self._tick = time.perf_counter()
