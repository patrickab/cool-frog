"""Reconstruction metrics as polars tables, one row per design and one column per channel, plus the fluid training loss."""

from collections.abc import Iterable

import polars as pl
import torch

from coolfrog.engine.tadpole import Tadpole
from coolfrog.lib.datatypes import CHANNELS, FlowField, LatentFlowField, NormalizationStats
from coolfrog.lib.distributed import all_gather_objects

SPATIAL_DIMS = (-3, -2, -1)
NRMSE, MSE, SPECTRAL_LOSS, KL = "nrmse", "mse", "spectral_loss", "kl"  # keys of `evaluate`


def nrmse(prediction: FlowField, reference: FlowField) -> pl.DataFrame:
    """Fluid-voxel RMSE divided by the reference fluid std, in percent."""
    fluid = reference.mask.unsqueeze(-4)
    fluid_count = fluid.sum(dim=SPATIAL_DIMS, keepdim=True)
    reference_mean = _fluid_sum(reference.fields, fluid) / fluid_count
    reference_variance = (_fluid_sum((reference.fields - reference_mean).square(), fluid) / (fluid_count - 1)).squeeze(SPATIAL_DIMS)
    return _per_channel_table(100 * (field_mse(prediction, reference) / reference_variance).sqrt(), reference)


def mse(prediction: FlowField, reference: FlowField) -> pl.DataFrame:
    """Fluid-voxel MSE per design and channel, in normalized units like the training loss."""
    return _per_channel_table(field_mse(prediction, reference), reference)


def field_mse(prediction: FlowField, reference: FlowField) -> torch.Tensor:
    """(B, C) fluid-voxel MSE per field."""
    fluid = reference.mask.unsqueeze(-4)
    fluid_count = fluid.sum(dim=SPATIAL_DIMS, keepdim=True)
    squared_error = (prediction.fields.float() - reference.fields).square()
    return (_fluid_sum(squared_error, fluid) / fluid_count).squeeze(SPATIAL_DIMS)


def spectral_distance(prediction: FlowField, reference: FlowField) -> torch.Tensor:
    """(B, C) mean L1 distance of log1p FFT amplitudes over the full grid, solids zeroed in both, float32."""
    fluid = reference.mask.unsqueeze(-4)
    prediction_amplitude = torch.fft.rfftn(prediction.fields.float() * fluid, dim=SPATIAL_DIMS).abs().log1p()
    reference_amplitude = torch.fft.rfftn(reference.fields.float() * fluid, dim=SPATIAL_DIMS).abs().log1p()
    return (prediction_amplitude - reference_amplitude).abs().mean(dim=SPATIAL_DIMS)


def spectral_loss(prediction: FlowField, reference: FlowField) -> pl.DataFrame:
    """`spectral_distance` per design and channel."""
    return _per_channel_table(spectral_distance(prediction, reference), reference)


def fluid_mse(prediction: FlowField, reference: FlowField) -> torch.Tensor:
    """Scalar MSE over fluid voxels, the differentiable training loss."""
    fluid = reference.mask.unsqueeze(-4).expand_as(reference.fields)
    squared_error = (prediction.fields.float() - reference.fields).square()
    return torch.where(fluid, squared_error, 0.0).sum() / fluid.sum()


def field_kl(latent: LatentFlowField) -> torch.Tensor:
    """(B, C) mean KL divergence to the prior per field, over all latent cells."""
    return latent.kl_divergence().flatten(2).mean(dim=2)


def evaluate(model: Tadpole, flowfields: Iterable[FlowField], stats: NormalizationStats) -> dict[str, pl.DataFrame]:
    """NRMSE, MSE, spectral loss and KL divergence per design and channel on the full domain, float32, latent mean decoded.

    - Under `torchrun` each process evaluates its share of the designs, every process gets all rows sorted by `design_id`
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    local_tables: dict[str, list[pl.DataFrame]] = {NRMSE: [], MSE: [], SPECTRAL_LOSS: [], KL: []}
    with torch.no_grad():
        for flowfield in flowfields:
            reference = flowfield.to(device).normalize(stats)
            latent = model.encode(reference)
            prediction = model.decode(latent)
            local_tables[NRMSE].append(nrmse(prediction, reference))  # scale-invariant: physical units, log10 space for log channels
            local_tables[MSE].append(mse(prediction, reference))  # in normalized units, like the training loss
            local_tables[SPECTRAL_LOSS].append(spectral_loss(prediction, reference))  # in normalized units
            local_tables[KL].append(_per_channel_table(field_kl(latent), reference))
    model.train(was_training)
    gathered = all_gather_objects({metric: pl.concat(tables) for metric, tables in local_tables.items()})
    return {metric: pl.concat(tables[metric] for tables in gathered).sort("design_id") for metric in local_tables}


def _fluid_sum(values: torch.Tensor, fluid: torch.Tensor) -> torch.Tensor:
    return torch.where(fluid, values, 0.0).sum(dim=SPATIAL_DIMS, keepdim=True)


def _per_channel_table(values: torch.Tensor, reference: FlowField) -> pl.DataFrame:
    """(B, C) values -> `design_id` plus one column per channel label."""
    columns = values.detach().float().cpu().T.tolist()
    return pl.DataFrame({"design_id": reference.design_id.cpu().tolist(), **dict(zip(CHANNELS, columns, strict=True))})
