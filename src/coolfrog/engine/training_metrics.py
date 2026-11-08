"""The `metrics` table of one fine-tuning run: every train sample and every validation design, per channel."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import logging
import time

import polars as pl
import torch
from torch.utils.data import DataLoader

from coolfrog.engine.model_evaluation import KL, MSE, SPECTRAL_LOSS
from coolfrog.lib.config import TrainConfig
from coolfrog.lib.dashboard import duration
from coolfrog.lib.datatypes import FlowField
from coolfrog.lib.distributed import all_gather_objects, world_size
from coolfrog.lib.metrics import Tracker
from coolfrog.lib.plots import CHANNEL_LABELS, CHANNEL_SYMBOLS, METRICS, METRICS_SCHEMA, TRAIN, VALIDATION

logger = logging.getLogger(__name__)

SCHEMAS = {METRICS: METRICS_SCHEMA}  # tracker schemas, see `Tracker`
TRAINING_PROGRESS, EPOCHS_PROGRESS, VALIDATION_PROGRESS = "Training", "Epochs", "Validation"  # progress bar labels
EPOCH_STATE = "epoch"  # state key: updates per epoch and the duration of one epoch at the training rate so far


@dataclass
class TrainSamples:
    """Per-sample losses of one process in one update, collected pass by pass, a sample is one single-channel crop."""

    design_ids: list[int] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    losses: dict[str, list[float]] = field(default_factory=lambda: {MSE: [], SPECTRAL_LOSS: [], KL: []})

    def add(self, design_id: torch.Tensor, channel_index: torch.Tensor, losses: Mapping[str, torch.Tensor]) -> None:
        """Add one pass: (B,) design ids and channel indices, (B,) per-sample loss tensors keyed like `losses`."""
        self.design_ids += design_id.tolist()
        self.channels += [CHANNEL_SYMBOLS[index] for index in channel_index.tolist()]
        for name, values in losses.items():
            self.losses[name] += values.float().tolist()

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame({"design_id": self.design_ids, "channel": self.channels, **self.losses})


def validation_rows(update: int, per_design_tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """`evaluate` tables as `METRICS_SCHEMA` validation rows, one per design and channel, channels as symbols like `p`."""
    symbols = {label: symbol for symbol, label in CHANNEL_LABELS.items()}
    long_tables = [
        table.rename(symbols).unpivot(on=list(symbols.values()), index="design_id", variable_name="channel", value_name=metric)
        for metric, table in per_design_tables.items()
    ]  # same design and channel order in every table, `evaluate` sorts by design_id
    rows = pl.concat([long_tables[0], *(table.drop("design_id", "channel") for table in long_tables[1:])], how="horizontal")
    rows = rows.with_columns(
        update=pl.lit(update), split=pl.lit(VALIDATION), learning_rate=pl.lit(None), seconds_per_update=pl.lit(None)
    )
    return rows.select(pl.col(name).cast(dtype) for name, dtype in METRICS_SCHEMA.items())


class TrainingMetrics:
    """Logging of one `fine_tune_autoencoder` run into the `metrics` table (`METRICS_SCHEMA`) of `tracker`.

    - Train rows: every sample of every update, a sample is one design and channel
    - Validation rows: every design and channel of every validation
    - Progress bars: updates done, epochs done when `config.epochs` is set, designs done over all GPUs of the last validation
    - State `epoch`: updates per epoch and the duration of one epoch at the training rate so far, validation excluded

    Examples
    --------
    >>> metrics = TrainingMetrics(TrainConfig(epochs=3), batch_size=8, n_train_designs=5538, tracker=Tracker(schemas=SCHEMAS))
    >>> metrics.updates, metrics.updates_per_epoch
    (522, 174)
    """

    def __init__(self, config: TrainConfig, batch_size: int, n_train_designs: int, tracker: Tracker) -> None:
        self.config = config
        self.batch_size = batch_size
        self.updates = config.total_updates(n_train_designs)
        self.updates_per_epoch = config.updates_per_epoch(n_train_designs)
        self.tracker = tracker
        if config.epochs is not None:  # bars show in creation order: epochs, training, validation
            self.tracker.set_progress(EPOCHS_PROGRESS, 0, config.epochs)
        self.tracker.set_progress(TRAINING_PROGRESS, 0, self.updates)
        self.tracker.update_state({EPOCH_STATE: {"updates": self.updates_per_epoch}})
        self._tick = time.perf_counter()

    def log_update(self, update: int, samples: TrainSamples, learning_rate: float) -> None:
        """Append the samples of every process for one optimizer update, KeyboardInterrupt when the dashboard asked to stop.

        - Every process calls it, the samples are gathered over the processes
        """
        now = time.perf_counter()
        rows = pl.concat(all_gather_objects(samples.to_frame()))
        rows = rows.with_columns(
            update=pl.lit(update),
            split=pl.lit(TRAIN),
            nrmse=pl.lit(None),
            learning_rate=pl.lit(learning_rate),
            seconds_per_update=pl.lit(now - self._tick),
        )
        self._tick = now
        self.tracker.extend(METRICS, rows.select(METRICS_SCHEMA.names()))
        self.tracker.set_progress(TRAINING_PROGRESS, update, self.updates)
        if self.config.epochs is not None:
            self.tracker.set_progress(EPOCHS_PROGRESS, update // self.updates_per_epoch, self.config.epochs)
        self._log_epoch_duration(update)
        if update % self.config.validate_every == 0 or update == self.updates:
            logger.info("update %d train MSE %.3e, mean over samples", update, rows[MSE].mean())

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
        """Append every design and channel of the `evaluate` tables, log the channel mean of their medians over designs.

        - Resets the update timer, validation time does not count as update time
        """
        rows = validation_rows(update, per_design_tables)
        self.tracker.extend(METRICS, rows)
        medians = rows.group_by("channel").agg(pl.col(MSE, SPECTRAL_LOSS).median())
        channel_mean = medians.select(pl.col(MSE, SPECTRAL_LOSS).mean()).row(0)
        logger.info("update %d validation median MSE %.3e spectral loss %.3e, mean over channels", update, *channel_mean)
        self._tick = time.perf_counter()
