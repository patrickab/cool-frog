"""Fine-tuning of the Tadpole autoencoder on random single-channel crops."""

from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
import itertools
import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from coolfrog.engine.model_evaluation import evaluate, fluid_mse, spectral_distance
from coolfrog.engine.tadpole import Tadpole
from coolfrog.engine.training_metrics import TrainingMetrics
from coolfrog.lib.config import HOST_CONFIG, TrainConfig
from coolfrog.lib.dashboard import run_dashboard
from coolfrog.lib.datatypes import FlowField, NormalizationStats
from coolfrog.lib.distributed import average_gradients, average_values, broadcast_parameters, is_main_process, world_size
from coolfrog.lib.metrics import Tracker
from coolfrog.lib.monitor import serve_monitor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RandomCrops:
    """Collate designs into one random, normalized single-channel crop of edge `crop_size` each, inside a loader worker.

    - Crops before normalizing, so the full designs are never copied
    - `crop_size=None` takes whole single-channel domains
    - The loader's `batch_size` is the number of crops per pass
    """

    stats: NormalizationStats
    crop_size: int | None

    def __call__(self, designs: list[FlowField]) -> FlowField:
        crops, channel_index = random_crops(torch.stack(designs), self.crop_size)
        return crops.normalize(self.stats.select_channels(channel_index))


def random_crops(batch: FlowField, crop_size: int | None) -> tuple[FlowField, torch.Tensor]:
    """One random single-channel crop of edge `crop_size` per design plus its channel index, axes shorter than `crop_size` kept whole.

    - Each crop comes from a random channel and position of its design
    - `crop_size=None` keeps every axis whole
    - Returns (n_designs, 1, X, Y, Z) crops and their (n_designs,) channel indices
    """
    n_designs, n_channels, *grid_shape = batch.fields.shape
    crop_shape = grid_shape if crop_size is None else [min(crop_size, voxels) for voxels in grid_shape]
    channel_index = torch.randint(n_channels, (n_designs,))
    starts = [torch.randint(voxels - crop + 1, (n_designs,)).tolist() for voxels, crop in zip(grid_shape, crop_shape, strict=True)]
    fields, masks = [], []
    for design, channel in enumerate(channel_index.tolist()):
        window = tuple(slice(start[design], start[design] + crop) for start, crop in zip(starts, crop_shape, strict=True))
        fields.append(batch.fields[design, channel : channel + 1][(..., *window)])
        masks.append(batch.mask[design][window])
    crops = FlowField(
        fields=torch.stack(fields),
        mask=torch.stack(masks),
        design_id=batch.design_id,
        params=batch.params,
        batch_size=[n_designs],
    )
    return crops, channel_index


def create_optimizer(parameters: Iterable[nn.Parameter], config: TrainConfig) -> tuple[torch.optim.AdamW, LambdaLR]:
    """AdamW with linear warmup over `warmup_updates`, then constant."""
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    scheduler = LambdaLR(optimizer, lambda update: min(1.0, (update + 1) / config.warmup_updates))
    return optimizer, scheduler


def endless(loader: DataLoader[FlowField]) -> Iterator[FlowField]:
    """Reshuffled passes over `loader` without end, a new `DistributedSampler` shuffle per pass."""
    for epoch in itertools.count():
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        yield from loader


def fine_tune_autoencoder(  # noqa: PLR0913, model, data, stats, config, checkpoint folder and tracker
    model: Tadpole,
    train_loader: DataLoader[FlowField],
    validation_loader: DataLoader[FlowField],
    stats: NormalizationStats,
    config: TrainConfig,
    path_checkpoints: Path,
    tracker: Tracker,
) -> None:
    """Accumulated updates on `train_loader` crops, in `HOST_CONFIG.autocast_dtype`, with periodic full-domain `evaluate`.

    - Update 0 is validated before training, the zero-shot baseline under LoRA
    - Writes `model.adapter()` to `path_checkpoints/adapter_update_<update>.pt` after every validation
    - Logs the spectral loss of the training crops next to the MSE they are trained on
    - Logs into the `tracker` tables of `TrainingMetrics`, e.g. a `Stage.track` tracker persisting them
    - Runs under the live terminal dashboard, plainly when there is no terminal, and serves the tracker to the web
      monitor meanwhile, see `coolfrog.lib.monitor`
    - Stop key `q` in the dashboard raises KeyboardInterrupt
    - Under `torchrun` every process runs it on its GPU: rank 0 alone writes, shows the dashboard and serves the monitor
    """
    metrics = TrainingMetrics(config, train_loader.batch_size, len(train_loader.dataset), tracker)
    path_checkpoints.mkdir(parents=True, exist_ok=True)
    with serve_monitor(tracker) if is_main_process() else nullcontext():
        run_dashboard(
            tracker,
            lambda: _train(model, train_loader, validation_loader, stats, metrics, path_checkpoints),
            enabled=is_main_process(),
        )


def _train(  # noqa: PLR0913, the arguments of fine_tune_autoencoder plus its metrics
    model: Tadpole,
    train_loader: DataLoader[FlowField],
    validation_loader: DataLoader[FlowField],
    stats: NormalizationStats,
    metrics: TrainingMetrics,
    path_checkpoints: Path,
) -> None:
    """Training loop, data-parallel under `torchrun`.

    - `metrics.updates` updates, `TrainConfig.updates` or its `epochs` over the `train_loader` designs
    - `designs_per_update` crops per update in total, `train_loader.batch_size` per pass on each process
    - Passes per process = `designs_per_update / (batch_size * n_gpus)`, so an update is the same on any GPU count
    - Every process starts from rank 0's LoRA initialization, gradients and logged losses are averaged over the processes
    """
    config = metrics.config
    designs_per_pass = metrics.batch_size * world_size()
    if config.designs_per_update % designs_per_pass:
        raise ValueError(
            f"designs_per_update {config.designs_per_update} must split evenly into passes of "
            f"batch size {metrics.batch_size} on {world_size()} GPUs"
        )
    passes = config.designs_per_update // designs_per_pass
    device = next(model.parameters()).device
    autocast_dtype = HOST_CONFIG.autocast_dtype
    logger.info("Training forward passes in %s", "float32" if autocast_dtype is None else f"{autocast_dtype} autocast")
    logger.info("%d passes of %d designs per update on each of %d GPUs", passes, metrics.batch_size, world_size())
    parameters = list(model.trainable_parameters().values())
    broadcast_parameters(parameters)
    optimizer, scheduler = create_optimizer(parameters, config)

    def validate(update: int) -> None:
        metrics.log_validation(update, evaluate(model, metrics.track_validation(validation_loader), stats))
        if is_main_process():
            torch.save(model.adapter(), path_checkpoints / f"adapter_update_{update}.pt")

    validate(0)
    batches = endless(train_loader)
    model.train()
    for update in range(1, metrics.updates + 1):
        loss_sums = torch.zeros(3, device=device)  # MSE, spectral loss, KL
        for _ in range(passes):
            crops = next(batches).to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                prediction, latent = model(crops)
            mse, kl = fluid_mse(prediction, crops), latent.kl_divergence().mean()
            ((mse + config.kl_weight * kl) / passes).backward()  # gradient = mean over this process's passes
            with torch.no_grad():  # monitored only, the loss stays fluid MSE
                spectral = spectral_distance(prediction.detach(), crops).mean()
                loss_sums += torch.stack([mse.detach().float(), spectral.float(), kl.detach().float()])
        average_gradients(parameters)  # mean over all passes of all processes
        learning_rate = optimizer.param_groups[0]["lr"]  # used by this update, the scheduler then sets the next
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        train_mse, train_spectral, train_kl = (average_values(loss_sums) / passes).tolist()
        metrics.log_update(update, train_mse, train_spectral, train_kl, learning_rate)
        if update % config.validate_every == 0 or update == metrics.updates:
            logger.info("update %d train MSE %.3e", update, train_mse)
            validate(update)
