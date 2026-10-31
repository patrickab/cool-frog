"""Fine-tuning of the Tadpole autoencoder on random single-channel crops."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import polars as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from coolfrog.engine.model_evaluation import evaluate, fluid_mse
from coolfrog.engine.tadpole import Tadpole
from coolfrog.lib.config import TrainConfig
from coolfrog.lib.datatypes import CHANNELS, FlowField, NormalizationStats
from coolfrog.lib.log import get_logger, track

logger = get_logger(__name__)


@dataclass(frozen=True)
class RandomCrops:
    """Collate designs into `count` random, normalized single-channel crops of edge `crop_size`, inside a loader worker.

    - Crops before normalizing, so the full designs are never copied
    """

    stats: NormalizationStats
    crop_size: int
    count: int

    def __call__(self, designs: list[FlowField]) -> FlowField:
        crops, channel_index = random_crops(torch.stack(designs), self.crop_size, self.count)
        return crops.normalize(self.stats.select_channels(channel_index))


def random_crops(batch: FlowField, crop_size: int, count: int) -> tuple[FlowField, torch.Tensor]:
    """Random single-channel crops of edge `crop_size` plus each crop's channel index, axes shorter than `crop_size` kept whole.

    - Each crop comes from a random design, channel and position of `batch`
    - Returns (count, 1, X, Y, Z) crops and their (count,) channel indices
    """
    n_designs, n_channels, *grid_shape = batch.fields.shape
    crop_shape = [min(crop_size, voxels) for voxels in grid_shape]
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
    """Reshuffled passes over `loader` without end."""
    while True:
        yield from loader


def fine_tune_autoencoder(
    model: Tadpole,
    train_loader: DataLoader[FlowField],
    validation_loader: DataLoader[FlowField],
    stats: NormalizationStats,
    config: TrainConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Accumulated bf16 updates on `train_loader` crops with periodic full-domain `evaluate`.

    - Update 0 is validated before training, the zero-shot baseline under LoRA
    - Returns (train loss per update, mean validation metric per update, metric and channel)
    """
    device = next(model.parameters()).device
    optimizer, scheduler = create_optimizer(model.trainable_parameters().values(), config)
    loss_rows, validation_tables = [], [_validate(model, validation_loader, stats, update=0)]
    batches = endless(train_loader)
    model.train()
    for update in track(range(1, config.updates + 1), "Fine-tuning"):
        mse_sum, kl_sum = 0.0, 0.0
        for _ in range(config.accumulation_steps):
            crops = next(batches).to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                prediction, latent = model(crops)
            mse, kl = fluid_mse(prediction, crops), latent.kl_divergence().mean()
            ((mse + config.kl_weight * kl) / config.accumulation_steps).backward()  # gradient = mean over all passes
            mse_sum, kl_sum = mse_sum + mse.item(), kl_sum + kl.item()
        learning_rate = optimizer.param_groups[0]["lr"]  # used by this update, the scheduler then sets the next
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        loss_rows.append(
            {
                "update": update,
                "mse": mse_sum / config.accumulation_steps,
                "kl": kl_sum / config.accumulation_steps,
                "learning_rate": learning_rate,
            }
        )
        if update % config.validate_every == 0 or update == config.updates:
            logger.info("update %d train MSE %.3e", update, loss_rows[-1]["mse"])
            validation_tables.append(_validate(model, validation_loader, stats, update))
    return pl.DataFrame(loss_rows), pl.concat(validation_tables)


def _validate(model: Tadpole, loader: DataLoader[FlowField], stats: NormalizationStats, update: int) -> pl.DataFrame:
    """Mean of each `evaluate` metric over the validation designs, one row per metric."""
    channels = pl.col([channel.value for channel in CHANNELS])
    means = [
        per_design.select(pl.lit(update).alias("update"), pl.lit(name).alias("metric"), channels.mean())
        for name, per_design in evaluate(model, loader, stats).items()
    ]
    table = pl.concat(means)
    logger.info("update %d validation means\n%s", update, table)
    return table
