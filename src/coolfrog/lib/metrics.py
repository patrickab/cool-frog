"""Reconstruction metrics as polars tables, one row per design and one column per channel."""

import polars as pl
import torch

from coolfrog.lib.datatypes import CHANNELS, FlowField

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


def _fluid_sum(values: torch.Tensor, fluid: torch.Tensor) -> torch.Tensor:
    return torch.where(fluid, values, 0.0).sum(dim=SPATIAL_DIMS, keepdim=True)


def _per_channel_table(values: torch.Tensor, reference: FlowField) -> pl.DataFrame:
    """(B, C) values -> `design_id` plus one column per channel label."""
    columns = values.detach().float().cpu().T.tolist()
    return pl.DataFrame({"design_id": reference.design_id.cpu().tolist(), **dict(zip(CHANNELS, columns, strict=True))})
