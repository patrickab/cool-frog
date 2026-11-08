"""Metric tables of one fine-tuning run: every train sample and every validation design per channel, plus their aggregates."""

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
from coolfrog.lib.metrics import Tracker, aggregate
from coolfrog.lib.plots import (
    CHANNEL_LABELS,
    CHANNEL_SYMBOLS,
    METRICS_TRAINING,
    METRICS_TRAINING_AGGREGATES,
    METRICS_VALIDATION,
    METRICS_VALIDATION_AGGREGATES,
    TRAINING_AGGREGATES_SCHEMA,
    TRAINING_BY,
    TRAINING_METRICS,
    TRAINING_SCHEMA,
    VALIDATION_AGGREGATES_SCHEMA,
    VALIDATION_BY,
    VALIDATION_METRICS,
    VALIDATION_SCHEMA,
    typed,
)

logger = logging.getLogger(__name__)

SCHEMAS = {  # tracker schemas, see `Tracker`
    METRICS_TRAINING: TRAINING_SCHEMA,
    METRICS_TRAINING_AGGREGATES: TRAINING_AGGREGATES_SCHEMA,
    METRICS_VALIDATION: VALIDATION_SCHEMA,
    METRICS_VALIDATION_AGGREGATES: VALIDATION_AGGREGATES_SCHEMA,
}
TRAINING_PROGRESS, EPOCHS_PROGRESS, VALIDATION_PROGRESS = "Training", "Epochs", "Validation"  # progress bar labels
CURRENT_EPOCH_PROGRESS = "Current epoch"  # replaces the training bar when `TrainConfig.epochs` is set
TRAINING_CLOCKS = (TRAINING_PROGRESS, EPOCHS_PROGRESS, CURRENT_EPOCH_PROGRESS)  # paused during validation, absent bars skipped
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
    """`evaluate` tables as `VALIDATION_SCHEMA` rows, one per design and channel, channels as symbols like `p`."""
    symbols = {label: symbol for symbol, label in CHANNEL_LABELS.items()}
    long_tables = [
        table.rename(symbols).unpivot(on=list(symbols.values()), index="design_id", variable_name="channel", value_name=metric)
        for metric, table in per_design_tables.items()
    ]  # same design and channel order in every table, `evaluate` sorts by design_id
    rows = pl.concat([long_tables[0], *(table.drop("design_id", "channel") for table in long_tables[1:])], how="horizontal")
    return typed(rows.with_columns(update=pl.lit(update)), VALIDATION_SCHEMA)


class TrainingMetrics:
    """Logging of one `fine_tune_autoencoder` run into the metric tables of `tracker`, see `SCHEMAS`.

    - `metrics_training`: every sample of every update, a sample is one design and channel
    - `metrics_validation`: every design and channel of every validation
    - `*_aggregates`: `aggregate` of the rows per update, for validation also per channel, written with the rows,
      training also holds the `optimizer_values` and `seconds_per_update` once per update
    - Progress bars: updates done, with `config.epochs` instead epochs done (time left for all) and updates done in the
      current epoch (time left in it), designs done over all GPUs of the last validation
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
        if config.epochs is not None:  # bars show in creation order: epochs, current epoch, validation
            self.tracker.set_progress(EPOCHS_PROGRESS, 0, config.epochs)
            self.tracker.set_progress(CURRENT_EPOCH_PROGRESS, 0, self.updates_per_epoch)
        else:
            self.tracker.set_progress(TRAINING_PROGRESS, 0, self.updates)
        self.tracker.update_state({EPOCH_STATE: {"updates": self.updates_per_epoch}})
        self._tick = time.perf_counter()

    def log_checkpointing(self, is_checkpointing: bool) -> None:
        """Log whether activation checkpointing is on and keep it in the tracker state."""
        logger.info("Activation checkpointing %s", "on, an update does not fit without it" if is_checkpointing else "off")
        self.tracker.update_state({"activation_checkpointing": is_checkpointing})

    def log_update(self, update: int, samples: TrainSamples, optimizer_values: Mapping[str, float]) -> None:
        """Append the samples of every process for one optimizer update and their aggregate, KeyboardInterrupt on a dashboard stop.

        - Every process calls it, the samples are gathered over the processes
        - `optimizer_values`: one value per update, e.g. `learning_rate` and `gradient_norm`, the same on every process
        """
        now = time.perf_counter()
        rows = typed(pl.concat(all_gather_objects(samples.to_frame())).with_columns(update=pl.lit(update)), TRAINING_SCHEMA)
        aggregates = aggregate(rows, TRAINING_BY, TRAINING_METRICS).with_columns(
            **{name: pl.lit(value) for name, value in optimizer_values.items()}, seconds_per_update=pl.lit(now - self._tick)
        )
        self._tick = now
        self.tracker.extend(METRICS_TRAINING, rows)
        self.tracker.extend(METRICS_TRAINING_AGGREGATES, typed(aggregates, TRAINING_AGGREGATES_SCHEMA))
        if self.config.epochs is None:
            self.tracker.set_progress(TRAINING_PROGRESS, update, self.updates)
        else:
            self._log_epoch_progress(update)
        self._log_epoch_duration(update)
        if update % self.config.validate_every == 0 or update == self.updates:
            logger.info("update %d train MSE %.3e, mean over samples", update, rows[MSE].mean())

    def _log_epoch_progress(self, update: int) -> None:
        """Advance the epochs and current epoch bars, a finished epoch restarts the current epoch bar and its clock."""
        epochs_done, updates_in_epoch = divmod(update, self.updates_per_epoch)
        self.tracker.set_progress(EPOCHS_PROGRESS, epochs_done, self.config.epochs, done=update / self.updates)
        if updates_in_epoch == 0 and update < self.updates:  # the next epoch's clock starts now, not at its first update
            self.tracker.set_progress(CURRENT_EPOCH_PROGRESS, 0, self.updates_per_epoch)
        else:
            self.tracker.set_progress(CURRENT_EPOCH_PROGRESS, updates_in_epoch or self.updates_per_epoch, self.updates_per_epoch)

    def _log_epoch_duration(self, update: int) -> None:
        """Set the epoch state's duration from the whole run's unpaused time per update."""
        training = self.tracker.progress()[TRAINING_PROGRESS if self.config.epochs is None else EPOCHS_PROGRESS]
        seconds_per_epoch = training.elapsed(time.monotonic()) / update * self.updates_per_epoch
        self.tracker.update_state({EPOCH_STATE: {"updates": self.updates_per_epoch, "duration": duration(seconds_per_epoch)}})

    def track_validation(self, loader: DataLoader[FlowField]) -> Iterator[FlowField]:
        """Yield this GPU's validation batches while advancing the validation bar, training clock paused, KeyboardInterrupt on stop.

        - Bar counts designs over all GPUs: the total gathered once, done = this GPU's designs x GPUs, exact at the end
        """
        total = sum(all_gather_objects(len(loader.dataset)))  # every rank calls this, `evaluate` also gathers on all ranks
        for label in TRAINING_CLOCKS:
            self.tracker.pause_progress(label)
        try:
            self.tracker.set_progress(VALIDATION_PROGRESS, 0, total)
            completed = 0
            for batch in loader:
                yield batch
                completed += batch.batch_size[0]
                self.tracker.set_progress(VALIDATION_PROGRESS, min(completed * world_size(), total), total)
            self.tracker.set_progress(VALIDATION_PROGRESS, total, total)
        finally:
            for label in TRAINING_CLOCKS:
                self.tracker.resume_progress(label)

    def log_validation(self, update: int, per_design_tables: Mapping[str, pl.DataFrame]) -> None:
        """Append every design and channel of the `evaluate` tables and their aggregates, log the channel mean of the medians.

        - Resets the update timer, validation time does not count as update time
        """
        rows = validation_rows(update, per_design_tables)
        aggregates = typed(aggregate(rows, VALIDATION_BY, VALIDATION_METRICS), VALIDATION_AGGREGATES_SCHEMA)
        self.tracker.extend(METRICS_VALIDATION, rows)
        self.tracker.extend(METRICS_VALIDATION_AGGREGATES, aggregates)
        channel_mean = aggregates.select(pl.col(f"{MSE}/median", f"{SPECTRAL_LOSS}/median").mean()).row(0)
        logger.info("update %d validation median MSE %.3e spectral loss %.3e, mean over channels", update, *channel_mean)
        self._tick = time.perf_counter()
