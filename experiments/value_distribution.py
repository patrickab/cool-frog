"""Per-channel fluid value histograms and per-design stats over all CoolData designs, run from the repo root."""

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from coolfrog.engine.cooldata import CoolData, CoolMetadata
from coolfrog.lib.config import HOST_CONFIG, DataConfig
from coolfrog.lib.datatypes import CHANNELS, Channel, FlowField, Grid, NormalizationStats
from coolfrog.lib.experiment import create_experiment
from coolfrog.lib.log import get_logger, setup_logging, track

TITLE = "CoolData value distribution"
GOAL = (
    "Measure the fluid value distribution of every channel over all CoolData designs: "
    "histograms, quantiles and per-design stats, to identify outlier designs and choose a normalization."
)
HYPOTHESIS = None

DATA_CONFIG = DataConfig(grid=Grid(latent_scale=1), batch_size=1, num_workers=8)
N_BINS = 2048
QUANTILES = (1e-4, 1e-3, 1e-2, 0.5, 0.99, 0.999, 0.9999)
# Outlier rule: a design is excluded when any fluid voxel breaks it
MAX_VELOCITY = 100.0  # m/s, per component
TEMPERATURE_RANGE = (250.0, 600.0)  # K
# Normalization of the curated designs: these quantiles map to [-1, 1], log10 first for `LOG_CHANNELS`
NORMALIZATION_QUANTILES = (1e-4, 0.9999)
LOG_CHANNELS = (Channel.TURBULENT_KINETIC_ENERGY, Channel.TURBULENT_DISSIPATION_RATE)
# Bins are uniform in asinh((value - offset) / scale): linear within `scale` of `offset`, logarithmic beyond
# (offset, scale, low, high) per channel, bounds wider than every value seen, values outside land in the edge bins
BINNING: dict[Channel, tuple[float, float, float, float]] = {
    Channel.VX: (0.0, 1e-2, -1e4, 1e4),
    Channel.VY: (0.0, 1e-2, -1e4, 1e4),
    Channel.VZ: (0.0, 1e-2, -1e4, 1e4),
    Channel.PRESSURE: (0.0, 1e-1, -1e9, 1e9),
    Channel.TEMPERATURE: (293.15, 1e-2, 0.0, 1e4),
    Channel.TURBULENT_KINETIC_ENERGY: (0.0, 1e-10, 0.0, 1e7),
    Channel.TURBULENT_DISSIPATION_RATE: (0.0, 1e-10, 0.0, 1e14),
}
OFFSET, SCALE, LOW, HIGH = (
    torch.tensor([BINNING[channel][i] for channel in CHANNELS], dtype=torch.float64).view(-1, 1) for i in range(4)
)


def to_bin_space(values: torch.Tensor) -> torch.Tensor:
    """(C, N) physical values -> asinh space of `BINNING`."""
    return torch.asinh((values - OFFSET) / SCALE)


def from_bin_space(values: torch.Tensor) -> torch.Tensor:
    """(C, N) asinh-space values -> physical values."""
    return torch.sinh(values) * SCALE + OFFSET


BIN_LOW, BIN_HIGH = to_bin_space(LOW), to_bin_space(HIGH)  # (C, 1)
BIN_WIDTH = (BIN_HIGH - BIN_LOW) / N_BINS
BIN_EDGES = from_bin_space(BIN_LOW + torch.arange(N_BINS + 1) * BIN_WIDTH).numpy()  # (C, N_BINS + 1)
BIN_CENTERS = from_bin_space(BIN_LOW + (torch.arange(N_BINS) + 0.5) * BIN_WIDTH).numpy()  # (C, N_BINS)


def design_summary(batch: list[FlowField]) -> dict[str, torch.Tensor]:
    """Histogram (C, N_BINS) and exact min, max, sum, squared sum (C,) of one design's fluid voxels, runs in the workers."""
    (flowfield,) = batch
    fluid = flowfield.fields[:, flowfield.mask].double()  # (C, n_fluid)
    bin_index = ((to_bin_space(fluid) - BIN_LOW) / BIN_WIDTH).long().clamp(0, N_BINS - 1)
    counts = torch.zeros(len(CHANNELS), N_BINS, dtype=torch.int32)
    counts.scatter_add_(1, bin_index, torch.ones_like(bin_index, dtype=torch.int32))
    return {
        "design_id": flowfield.design_id,
        "counts": counts,
        "min": fluid.amin(1),
        "max": fluid.amax(1),
        "sum": fluid.sum(1),
        "sum_squares": fluid.square().sum(1),
    }


def histogram_quantiles(counts: np.ndarray, quantiles: tuple[float, ...] = QUANTILES) -> np.ndarray:
    """(..., C, N_BINS) counts -> (..., C, len(quantiles)) quantiles at bin-center precision."""
    cumulative = np.cumsum(counts, axis=-1, dtype=np.float64) / counts.sum(axis=-1, keepdims=True)
    bin_index = np.stack([(cumulative < q).sum(axis=-1) for q in quantiles], axis=-1).clip(0, N_BINS - 1)
    return np.take_along_axis(np.broadcast_to(BIN_CENTERS, (*counts.shape[:-1], N_BINS)), bin_index, axis=-1)


def channel_table(design_id: np.ndarray, n_fluid: np.ndarray, stats: dict[str, np.ndarray], quantiles: np.ndarray) -> pl.DataFrame:
    """Long table, one row per (design, channel), from (D, C) stats and (D, C, Q) quantiles."""
    n_designs = len(design_id)
    columns = {
        "design_id": np.repeat(design_id, len(CHANNELS)),
        "channel": np.tile([channel.value for channel in CHANNELS], n_designs),
        "n_fluid": np.repeat(n_fluid, len(CHANNELS)),
        **{name: values.reshape(-1) for name, values in stats.items()},
        **{f"q{q:g}": quantiles[..., i].reshape(-1) for i, q in enumerate(QUANTILES)},
    }
    return pl.DataFrame(columns)


def excluded_designs(design_id: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> pl.DataFrame:
    """`design_id` and `reason` of every design breaking the outlier rule, from its exact (D, C) fluid min and max."""
    velocity_extreme = np.maximum(np.abs(minimum[:, :3]), np.abs(maximum[:, :3])).max(axis=1)  # (D,), any component
    temperature = CHANNELS.index(Channel.TEMPERATURE)
    reasons = []
    for velocity, temperature_min, temperature_max in zip(
        velocity_extreme, minimum[:, temperature], maximum[:, temperature], strict=True
    ):
        broken = []
        if velocity > MAX_VELOCITY:
            broken.append(f"|V| {velocity:.0f} > {MAX_VELOCITY:g} m/s")
        if temperature_min < TEMPERATURE_RANGE[0]:
            broken.append(f"T min {temperature_min:.0f} < {TEMPERATURE_RANGE[0]:g} K")
        if temperature_max > TEMPERATURE_RANGE[1]:
            broken.append(f"T max {temperature_max:.0f} > {TEMPERATURE_RANGE[1]:g} K")
        reasons.append(", ".join(broken))
    return pl.DataFrame({"design_id": design_id, "reason": reasons}).filter(pl.col("reason") != "")


def normalization_stats(curated_counts: np.ndarray) -> NormalizationStats:
    """`NORMALIZATION_QUANTILES` of (C, N_BINS) curated counts as [low, high], log10 for `LOG_CHANNELS`."""
    low, high = torch.from_numpy(histogram_quantiles(curated_counts, NORMALIZATION_QUANTILES)).float().T.reshape(2, -1, 1, 1, 1)
    is_log = torch.tensor([channel in LOG_CHANNELS for channel in CHANNELS]).view(-1, 1, 1, 1)
    linear = NormalizationStats(low=low, high=high, is_log=is_log, batch_size=[])
    return linear.replace(low=linear.transform(low), high=linear.transform(high))


def normalized(stats: NormalizationStats, values: np.ndarray) -> np.ndarray:
    """(C,) physical values -> normalized values under `stats`."""
    transformed = stats.transform(torch.from_numpy(values).float().view(-1, 1, 1, 1))
    return (2 * (transformed - stats.low) / (stats.high - stats.low) - 1).flatten().numpy()


if __name__ == "__main__":
    path_results = create_experiment(TITLE, GOAL, HYPOTHESIS, [DATA_CONFIG])
    setup_logging(log_file=path_results / "run.log")
    log = get_logger("coolfrog.experiment")

    dataset = CoolData(CoolMetadata.load(HOST_CONFIG.n_splits, HOST_CONFIG.idx_split), DATA_CONFIG)
    log.info("%d designs on grid %s", len(dataset), DATA_CONFIG.grid.shape)
    loader = DataLoader(dataset, batch_size=1, num_workers=DATA_CONFIG.num_workers, collate_fn=design_summary)
    summaries = list(track(loader, f"Value distribution {'x'.join(map(str, DATA_CONFIG.grid.shape))}"))
    merged = {key: torch.stack([summary[key] for summary in summaries]).numpy() for key in summaries[0]}

    design_id, counts = merged["design_id"], merged["counts"]  # (D,), (D, C, N_BINS)
    n_fluid = counts[:, 0].sum(axis=-1)  # (D,)
    mean = merged["sum"] / n_fluid[:, None]
    std = np.sqrt(np.maximum(merged["sum_squares"] / n_fluid[:, None] - mean**2, 0))
    np.savez_compressed(path_results / "histograms.npz", design_id=design_id, counts=counts, bin_edges=BIN_EDGES)
    design_stats = {"min": merged["min"], "max": merged["max"], "mean": mean, "std": std}
    channel_table(design_id, n_fluid, design_stats, histogram_quantiles(counts)).write_parquet(path_results / "design_stats.parquet")

    total_fluid = n_fluid.sum()
    total_mean = merged["sum"].sum(0) / total_fluid
    total_stats = {
        "min": merged["min"].min(0, keepdims=True),
        "max": merged["max"].max(0, keepdims=True),
        "mean": total_mean[None],
        "std": np.sqrt(np.maximum(merged["sum_squares"].sum(0) / total_fluid - total_mean**2, 0))[None],
    }
    total_quantiles = histogram_quantiles(counts.sum(0, dtype=np.int64))[None]
    channel_stats = channel_table(np.array([-1]), np.array([total_fluid]), total_stats, total_quantiles)
    channel_stats.drop("design_id").write_csv(path_results / "channel_stats.csv")

    excluded = excluded_designs(design_id, merged["min"], merged["max"])
    excluded.write_csv(path_results / "excluded_designs.csv")
    is_curated = ~np.isin(design_id, excluded["design_id"].to_numpy())
    stats = normalization_stats(counts[is_curated].sum(0, dtype=np.int64))
    stats.save(path_results / "normalization_stats.csv")
    curated_range = normalized(stats, merged["min"][is_curated].min(0)), normalized(stats, merged["max"][is_curated].max(0))

    with pl.Config(tbl_cols=-1, tbl_width_chars=250, tbl_hide_dataframe_shape=True, tbl_rows=-1, fmt_str_lengths=100):
        log.info("channel stats over %d designs\n%s", len(design_id), channel_stats.drop("design_id", "n_fluid"))
        log.info("%d designs excluded\n%s", excluded.height, excluded)
        log.info(
            "normalization stats over %d curated designs, curated min and max normalized\n%s",
            is_curated.sum(),
            pl.read_csv(path_results / "normalization_stats.csv").with_columns(
                normalized_min=curated_range[0], normalized_max=curated_range[1]
            ),
        )
