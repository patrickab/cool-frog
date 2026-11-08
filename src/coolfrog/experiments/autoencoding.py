"""Shared setup of the zero-shot and LoRA fine-tuning autoencoding experiments: data, validation split, metric tables.

- Both evaluate the same curated validation designs, the fine-tuning sets its own data, model and training configs
- Imported by the experiment scripts, not an experiment itself
"""

from dataclasses import replace

import polars as pl
import torch
from torch.utils.data import DataLoader, random_split

from coolfrog.engine.cooldata import CoolData, CoolMetadata, create_dataloader
from coolfrog.lib.config import HOST_CONFIG, SEED, DataConfig, FinetuneState, Grid, TadpoleConfig
from coolfrog.lib.datatypes import CHANNELS, FlowField, NormalizationStats
from coolfrog.lib.experiment import PATH_EXPERIMENTS

LATENT_SCALES = (2, 4)  # 800x160x32 with 32³ tiles first, then 1600x320x64 with 64³ tiles
# Normalization and exclusion list of experiment 01 on the full dataset, used on every host
PATH_VALUE_DISTRIBUTION = PATH_EXPERIMENTS / "01_cooldata-value-distribution" / "results"
MEDIAN = 0.5


def load_stats() -> NormalizationStats:
    """Quantile normalization of experiment 01."""
    return NormalizationStats.load(PATH_VALUE_DISTRIBUTION / "normalization_stats.csv")


def curated_metadata() -> CoolMetadata:
    """All designs without the implausible ones excluded in experiment 01."""
    excluded = pl.read_csv(PATH_VALUE_DISTRIBUTION / "excluded_designs.csv")["design_id"]
    designs = CoolMetadata.load(HOST_CONFIG.n_splits, HOST_CONFIG.idx_split).designs
    return CoolMetadata(designs.filter(~pl.col("design_id").is_in(excluded)))


def data_config(latent_scale: int) -> DataConfig:
    """Grid of `latent_scale`, 4 designs per training pass."""
    return DataConfig(grid=Grid(latent_scale=latent_scale), batch_size=4, num_workers=6)


def tadpole_config(latent_scale: int, finetune_state: FinetuneState) -> TadpoleConfig:
    """Tadpole-B with tiles and training crops spanning the full channel height of the grid."""
    return TadpoleConfig(
        size="B",
        encoder_state=finetune_state,
        decoder_state=finetune_state,
        encoder_crop_size=Grid(latent_scale=latent_scale).shape[2],
        max_internal_batchsize=16,  # tiles per encoder/decoder call
    )


def split_metadata(metadata: CoolMetadata, n_validation: int) -> tuple[CoolMetadata, CoolMetadata]:
    """Seeded train / validation split holding out `n_validation` designs, the indices of `CoolData.split`."""
    n_designs = metadata.designs.height
    generator = torch.Generator().manual_seed(SEED)
    subsets = random_split(range(n_designs), [n_designs - n_validation, n_validation], generator)
    train, validation = (CoolMetadata(metadata.designs[sorted(subset.indices)]) for subset in subsets)
    return train, validation


def train_validation_split(metadata: CoolMetadata, config: DataConfig) -> tuple[CoolData, CoolData]:
    """`split_metadata` of `config.n_validation_designs` as datasets, identical at every latent scale."""
    train, validation = split_metadata(metadata, config.n_validation_designs)
    return CoolData(train, config), CoolData(validation, config)


def validation_loader(validation: CoolData, config: DataConfig) -> DataLoader[FlowField]:
    """One design per batch, fixed order."""
    return create_dataloader(validation, replace(config, batch_size=1), shuffle=False)


def per_design_frame(tables: dict[str, pl.DataFrame], latent_scale: int, model: str) -> pl.DataFrame:
    """Long table of `evaluate` output: `latent_scale`, `model`, `metric`, `design_id`, one column per channel."""
    return pl.concat(
        table.select(
            pl.lit(latent_scale).alias("latent_scale"), pl.lit(model).alias("model"), pl.lit(metric).alias("metric"), pl.all()
        )
        for metric, table in tables.items()
    )


def statistic_name(quantile: float) -> str:
    """Row label of a quantile, `median` for 0.5."""
    return "median" if quantile == MEDIAN else f"q{quantile:g}"


def quantile_table(per_design: pl.DataFrame, metric: str, quantiles: tuple[float, ...]) -> pl.DataFrame:
    """Quantiles over designs per channel, one row per latent scale, model and quantile, linear like `Series.median`."""
    channels = [channel.value for channel in CHANNELS]
    rows = [
        per_design.filter(pl.col("metric") == metric)
        .group_by("latent_scale", "model", maintain_order=True)
        .agg(pl.lit(statistic_name(quantile)).alias("statistic"), pl.col(channels).quantile(quantile, interpolation="linear"))
        for quantile in quantiles
    ]
    return pl.concat(rows).sort("latent_scale", "model", maintain_order=True)
