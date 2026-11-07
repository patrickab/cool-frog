"""Reference vs prediction plots, solids in gray."""

from collections.abc import Sequence

from matplotlib import colormaps
from matplotlib.figure import Figure
import numpy as np

from coolfrog.lib.config import Grid
from coolfrog.lib.datatypes import CHANNELS, Channel, FlowField

SOLID_COLOR = "#999999"


def plot_slices(  # noqa: PLR0913
    reference: FlowField,
    prediction: FlowField,
    grid: Grid,
    sample_index: int = 0,
    z_index: int | None = None,
    channels: Sequence[Channel] = CHANNELS,
) -> "Figure":
    """Top-view slices of reference, prediction and |error| per channel, `z_index=None` = half height."""
    z_index = grid.shape[2] // 2 if z_index is None else z_index
    extent_mm = (0.0, 1000 * grid.extent[0], 0.0, 1000 * grid.extent[1])
    reference_volumes, prediction_volumes, fluid = _sample_volumes(reference, prediction, sample_index)
    figure = Figure(figsize=(15, 1.3 * len(channels)), layout="constrained")
    axes = figure.subplots(len(channels), 3, squeeze=False)
    for row, channel in enumerate(channels):
        channel_index = CHANNELS.index(channel)
        reference_slice, prediction_slice = reference_volumes[channel_index], prediction_volumes[channel_index]
        color_range = {"vmin": reference_slice[fluid].min(), "vmax": reference_slice[fluid].max()}
        panels = (
            ("reference", reference_slice, "plasma", color_range),
            ("prediction", prediction_slice, "plasma", color_range),
            ("|error|", np.abs(prediction_slice - reference_slice), "magma", {}),
        )
        for axis, (title, volume, colormap_name, limits) in zip(axes[row], panels, strict=True):
            colormap = colormaps[colormap_name].with_extremes(bad=SOLID_COLOR)
            image = axis.imshow(
                np.where(fluid, volume, np.nan)[:, :, z_index].T, origin="lower", extent=extent_mm, cmap=colormap, **limits
            )
            axis.set_title(f"{title} | {channel.value}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.02, pad=0.01)
    figure.suptitle(f"design {int(reference.design_id[sample_index])}, z index {z_index}")
    return figure


def _sample_volumes(reference: FlowField, prediction: FlowField, sample_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical (C, X, Y, Z) reference and prediction plus (X, Y, Z) fluid mask of one sample as numpy."""
    reference_sample, prediction_sample = (_physical(flowfield[sample_index]) for flowfield in (reference, prediction))
    return (
        reference_sample.fields.detach().float().cpu().numpy(),
        prediction_sample.fields.detach().float().cpu().numpy(),
        reference_sample.mask.cpu().numpy(),
    )


def _physical(flowfield: FlowField) -> FlowField:
    return flowfield if flowfield.stats is None else flowfield.denormalize()
