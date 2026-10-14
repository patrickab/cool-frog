"""Reconstruction metrics as polars tables, one row per design and one column per channel."""

import polars as pl

from coolfrog.lib.datatypes import FlowField


def nrmse(prediction: FlowField, reference: FlowField) -> pl.DataFrame:
    """Fluid-voxel RMSE divided by the reference fluid std, in percent."""
    raise NotImplementedError


def spectral_loss(prediction: FlowField, reference: FlowField) -> pl.DataFrame:
    """Mean L1 distance of log1p FFT amplitudes over the full grid, filled solids included."""
    raise NotImplementedError
