"""Tables of one fine-tuning run: windowed training means, validation per design and channel, and its medians over designs."""

from collections.abc import Iterator, Mapping
import logging
import time

import polars as pl
from torch.utils.data import DataLoader

from coolfrog.engine.model_evaluation import MSE, SPECTRAL_LOSS
from coolfrog.lib.config import TrainConfig
from coolfrog.lib.dashboard import duration
from coolfrog.lib.datatypes import FlowField
from coolfrog.lib.distributed import all_gather_objects, world_size
from coolfrog.lib.metrics import Tracker, WindowMean
from coolfrog.lib.plots import (
    CHANNEL_LABELS,
    TRAIN,
    TRAIN_SCHEMA,
    VALIDATION,
    VALIDATION_MEDIANS,
    VALIDATION_MEDIANS_SCHEMA,
    VALIDATION_SCHEMA,
    validation_medians,
)

logger = logging.getLogger(__name__)

SCHEMAS = {  # tracker schemas of the tables, see `Tracker`
    TRAIN: TRAIN_SCHEMA,
    VALIDATION: VALIDATION_SCHEMA,
    VALIDATION_MEDIANS: VALIDATION_MEDIANS_SCHEMA,
}
TRAINING_PROGRESS, EPOCHS_PROGRESS, VALIDATION_PROGRESS = "Training", "Epochs", "Validation"  # progress bar labels
EPOCH_STATE = "epoch"  # state key: updates per epoch and the measured duration of one epoch


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
    - `validation` table (`VALIDATION_SCHEMA`): every metric per update, design and channel
    - `validation_medians` table (`VALIDATION_MEDIANS_SCHEMA`): their medians over designs per update and channel
    - Progress bars: updates done, epochs done when `config.epochs` is set, designs done over all GPUs of the last validation
    - State `epoch`: updates per epoch and the duration of one epoch at the training rate so far, validation excluded

    Examples
    --------
    >>> metrics = TrainingMetrics(TrainConfig(epochs=3), batch_size=8, n_train_designs=5538, tracker=Tracker(schemas=SCHEMAS))
    >>> metrics.updates, metrics.updates_per_epoch
    (522, 174)
    >>> metrics.log_update(1, mse=0.1, spectral=0.4, kl=3.0, learning_rate=5e-5)
    """

    def __init__(self, config: TrainConfig, batch_size: int, n_train_designs: int, tracker: Tracker, log_every: int = 10) -> None:
        self.config = config
        self.batch_size = batch_size
        self.updates = config.total_updates(n_train_designs)
        self.updates_per_epoch = config.updates_per_epoch(n_train_designs)
        self.log_every = log_every
        self.tracker = tracker
        if config.epochs is not None:  # bars show in creation order: epochs, training, validation
            self.tracker.set_progress(EPOCHS_PROGRESS, 0, config.epochs)
        self.tracker.set_progress(TRAINING_PROGRESS, 0, self.updates)
        self.tracker.update_state({EPOCH_STATE: {"updates": self.updates_per_epoch}})
        self._window = WindowMean()
        self._tick = time.perf_counter()

    def log_update(self, update: int, mse: float, spectral: float, kl: float, learning_rate: float) -> None:
        """Record one optimizer update and advance the training bar, KeyboardInterrupt when the dashboard asked to stop."""
        self.tracker.set_progress(TRAINING_PROGRESS, update, self.updates)
        if self.config.epochs is not None:
            self.tracker.set_progress(EPOCHS_PROGRESS, update // self.updates_per_epoch, self.config.epochs)
        now = time.perf_counter()
        self._window.add(
            {"mse": mse, "spectral_loss": spectral, "kl": kl, "learning_rate": learning_rate, "seconds_per_update": now - self._tick}
        )
        self._tick = now
        if update % self.log_every == 0 or update == self.updates:
            self.tracker.append(TRAIN, {"update": update, **self._window.flush()})
            self._log_epoch_duration(update)

    def _log_epoch_duration(self, update: int) -> None:
        """Set the epoch state's duration from the training bar's unpaused time per update."""
        training = self.tracker.progress()[TRAINING_PROGRESS]
        seconds_per_epoch = training.elapsed(time.monotonic()) / update * self.updates_per_epoch
        self.tracker.update_state({EPOCH_STATE: {"updates": self.updates_per_epoch, "duration": duration(seconds_per_epoch)}})

    def track_validation(self, loader: DataLoader[FlowField]) -> Iterator[FlowField]:
        """Yield this GPU's validation batches while advancing the validation bar, training clock paused, KeyboardInterrupt on stop.

        - Bar counts designs over all GPUs: the total gathered once, done = this GPU's designs x GPUs, exact at the end
        """
        total = sum(all_gather_objects(len(loader.dataset)))  # every rank calls this, `evaluate` also gathers on all ranks
        self.tracker.pause_progress(TRAINING_PROGRESS)
        self.tracker.pause_progress(EPOCHS_PROGRESS)
        try:
            self.tracker.set_progress(VALIDATION_PROGRESS, 0, total)
            completed = 0
            for batch in loader:
                yield batch
                completed += batch.batch_size[0]
                self.tracker.set_progress(VALIDATION_PROGRESS, min(completed * world_size(), total), total)
            self.tracker.set_progress(VALIDATION_PROGRESS, total, total)
        finally:
            self.tracker.resume_progress(TRAINING_PROGRESS)
            self.tracker.resume_progress(EPOCHS_PROGRESS)

    def log_validation(self, update: int, per_design_tables: Mapping[str, pl.DataFrame]) -> None:
        """Append every design and channel of the `evaluate` tables and their medians over designs, log the channel mean of medians.

        - Resets the update timer, validation time does not count as update time
        """
        rows = validation_rows(update, per_design_tables)
        medians = validation_medians(rows)
        self.tracker.extend(VALIDATION, rows)
        self.tracker.extend(VALIDATION_MEDIANS, medians)
        channel_mean = medians.select(pl.col(MSE, SPECTRAL_LOSS).mean()).row(0)
        logger.info("update %d validation median MSE %.3e spectral loss %.3e, mean over channels", update, *channel_mean)
        self._tick = time.perf_counter()
