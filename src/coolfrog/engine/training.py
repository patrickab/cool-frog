"""Fine-tuning of the Tadpole autoencoder on random single-channel crops."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import itertools
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
from coolfrog.lib.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RandomCrops:
    """Collate designs into `count` random, normalized single-channel crops of edge `crop_size`, inside a loader worker.

    - Crops before normalizing, so the full designs are never copied
    - `crop_size=None` takes whole single-channel domains, `count=1` then backpropagates one scalar field per pass
    """

    stats: NormalizationStats
    crop_size: int | None
    count: int

    def __call__(self, designs: list[FlowField]) -> FlowField:
        crops, channel_index = random_crops(torch.stack(designs), self.crop_size, self.count)
        return crops.normalize(self.stats.select_channels(channel_index))


def random_crops(batch: FlowField, crop_size: int | None, count: int) -> tuple[FlowField, torch.Tensor]:
    """Random single-channel crops of edge `crop_size` plus each crop's channel index, axes shorter than `crop_size` kept whole.

    - Each crop comes from a random design, channel and position of `batch`
    - `crop_size=None` keeps every axis whole
    - Returns (count, 1, X, Y, Z) crops and their (count,) channel indices
    """
    n_designs, n_channels, *grid_shape = batch.fields.shape
    crop_shape = grid_shape if crop_size is None else [min(crop_size, voxels) for voxels in grid_shape]
    design_index = torch.randint(n_designs, (count,))
    channel_index = torch.randint(n_channels, (count,))
    starts = [torch.randint(voxels - crop + 1, (count,)).tolist() for voxels, crop in zip(grid_shape, crop_shape, strict=True)]
    fields, masks = [], []
    for crop_index, (design, channel) in enumerate(zip(design_index.tolist(), channel_index.tolist(), strict=True)):
        window = tuple(slice(start[crop_index], start[crop_index] + crop) for start, crop in zip(starts, crop_shape, strict=True))
        fields.append(batch.fields[design, channel : channel + 1][(..., *window)])
        masks.append(batch.mask[design][window])
    crops = FlowField(
        fields=torch.stack(fields),
        mask=torch.stack(masks),
        design_id=batch.design_id[design_index],
        params=batch.params[design_index],
        batch_size=[count],
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


def fine_tune_autoencoder(  # noqa: PLR0913, model, data, stats, config, checkpoint folder and the dashboard title
    model: Tadpole,
    train_loader: DataLoader[FlowField],
    validation_loader: DataLoader[FlowField],
    stats: NormalizationStats,
    config: TrainConfig,
    path_checkpoints: Path,
    title: str = "Fine-tuning",
) -> TrainingMetrics:
    """Accumulated updates on `train_loader` crops, in `HOST_CONFIG.autocast_dtype`, with periodic full-domain `evaluate`.

    - Update 0 is validated before training, the zero-shot baseline under LoRA
    - Writes `model.adapter()` to `path_checkpoints/adapter_update_<update>.pt` after every validation
    - Logs the spectral loss of the training crops next to the MSE they are trained on
    - Runs under the live terminal dashboard titled `title`, plainly when there is no terminal
    - Stop key `q` in the dashboard raises KeyboardInterrupt
    - Under `torchrun` every process runs it on its GPU: rank 0 alone writes and shows the dashboard, see `_train`
    """
    metrics = TrainingMetrics(config, title)
    path_checkpoints.mkdir(parents=True, exist_ok=True)
    run_dashboard(
        metrics.tracker,
        metrics.spec,
        lambda: _train(model, train_loader, validation_loader, stats, metrics, path_checkpoints),
        enabled=is_main_process(),
    )
    return metrics


def _train(  # noqa: PLR0913, the arguments of fine_tune_autoencoder plus its metrics
    model: Tadpole,
    train_loader: DataLoader[FlowField],
    validation_loader: DataLoader[FlowField],
    stats: NormalizationStats,
    metrics: TrainingMetrics,
    path_checkpoints: Path,
) -> None:
    """Training loop, data-parallel under `torchrun`.

    - `accumulation_steps` passes per update in total, split evenly over the processes, so updates match a single GPU
    - Every process starts from rank 0's LoRA initialization, gradients and logged losses are averaged over the processes
    """
    config = metrics.config
    if config.accumulation_steps % world_size():
        raise ValueError(f"accumulation_steps {config.accumulation_steps} must split evenly over {world_size()} GPUs")
    passes = config.accumulation_steps // world_size()
    device = next(model.parameters()).device
    autocast_dtype = HOST_CONFIG.autocast_dtype
    logger.info("Training forward passes in %s", "float32" if autocast_dtype is None else f"{autocast_dtype} autocast")
    logger.info("%d passes per update on each of %d GPUs", passes, world_size())
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
    for update in range(1, config.updates + 1):
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
        if update % config.validate_every == 0 or update == config.updates:
            logger.info("update %d train MSE %.3e", update, train_mse)
            validate(update)
