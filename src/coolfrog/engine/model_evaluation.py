"""Reconstruction metrics as polars tables, one row per design and one column per channel, plus the fluid training loss."""

from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl
import torch

from coolfrog.engine.tadpole import Tadpole
from coolfrog.lib.datatypes import CHANNELS, FlowField, NormalizationStats

SPATIAL_DIMS = (-3, -2, -1)


def nrmse(prediction: FlowField, reference: FlowField) -> pl.DataFrame:
    """Fluid-voxel RMSE divided by the reference fluid std, in percent."""
    fluid = reference.mask.unsqueeze(-4)
    fluid_count = fluid.sum(dim=SPATIAL_DIMS, keepdim=True)
    mean_squared_error = _fluid_sum((prediction.fields - reference.fields).square(), fluid) / fluid_count
    reference_mean = _fluid_sum(reference.fields, fluid) / fluid_count
    reference_variance = _fluid_sum((reference.fields - reference_mean).square(), fluid) / (fluid_count - 1)
    return _per_channel_table(100 * (mean_squared_error / reference_variance).sqrt().squeeze(SPATIAL_DIMS), reference)


def spectral_loss(prediction: FlowField, reference: FlowField) -> pl.DataFrame:
    """Mean L1 distance of log1p FFT amplitudes over the full grid, solids zeroed in both."""
    fluid = reference.mask.unsqueeze(-4)
    prediction_amplitude = torch.fft.rfftn(prediction.fields * fluid, dim=SPATIAL_DIMS).abs().log1p()
    reference_amplitude = torch.fft.rfftn(reference.fields * fluid, dim=SPATIAL_DIMS).abs().log1p()
    return _per_channel_table((prediction_amplitude - reference_amplitude).abs().mean(dim=SPATIAL_DIMS), reference)


def fluid_mse(prediction: FlowField, reference: FlowField) -> torch.Tensor:
    """Scalar MSE over fluid voxels, the differentiable training loss."""
    fluid = reference.mask.unsqueeze(-4).expand_as(reference.fields)
    squared_error = (prediction.fields.float() - reference.fields).square()
    return torch.where(fluid, squared_error, 0.0).sum() / fluid.sum()


def evaluate(model: Tadpole, flowfields: Iterable[FlowField], stats: NormalizationStats) -> dict[str, pl.DataFrame]:
    """NRMSE and spectral loss per design and channel on the full domain, float32, latent mean decoded."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    nrmse_tables, spectral_tables = [], []
    with torch.no_grad():
        for flowfield in flowfields:
            reference = flowfield.to(device).normalize(stats)
            prediction = model.decode(model.encode(reference))
            nrmse_tables.append(nrmse(prediction, reference))  # scale-invariant, equals physical-unit NRMSE
            spectral_tables.append(spectral_loss(prediction, reference))  # in normalized units
    model.train(was_training)
    return {"nrmse": pl.concat(nrmse_tables), "spectral_loss": pl.concat(spectral_tables)}


def summarize(per_design: pl.DataFrame) -> pl.DataFrame:
    """Mean, median and max per channel, one row per statistic."""
    channels = pl.col([channel.value for channel in CHANNELS])
    statistics = {"mean": channels.mean(), "median": channels.median(), "max": channels.max()}
    return pl.concat([per_design.select(pl.lit(name).alias("statistic"), values) for name, values in statistics.items()])


def write_metric_tables(tables: Mapping[str, pl.DataFrame], path_results: Path) -> None:
    """Write `<name>_per_design.csv` and `<name>_summary.csv` per metric."""
    for name, per_design in tables.items():
        per_design.write_csv(path_results / f"{name}_per_design.csv")
        summarize(per_design).write_csv(path_results / f"{name}_summary.csv")


def _fluid_sum(values: torch.Tensor, fluid: torch.Tensor) -> torch.Tensor:
    return torch.where(fluid, values, 0.0).sum(dim=SPATIAL_DIMS, keepdim=True)


def _per_channel_table(values: torch.Tensor, reference: FlowField) -> pl.DataFrame:
    """(B, C) values -> `design_id` plus one column per channel label."""
    columns = values.detach().float().cpu().T.tolist()
    return pl.DataFrame({"design_id": reference.design_id.cpu().tolist(), **dict(zip(CHANNELS, columns, strict=True))})
