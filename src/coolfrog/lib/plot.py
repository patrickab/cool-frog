"""Reference vs prediction plots, solids in gray."""

from collections.abc import Sequence

from matplotlib.figure import Figure
import pyvista as pv

from coolfrog.lib.datatypes import CHANNELS, Channel, FlowField, Grid


def plot_slices(  # noqa: PLR0913, PLR0917
    reference: FlowField,
    prediction: FlowField,
    grid: Grid,
    sample_index: int = 0,
    z_index: int | None = None,
    channels: Sequence[Channel] = CHANNELS,
) -> "Figure":
    """Top-view slices of reference, prediction and |error| per channel, `z_index=None` = half height."""
    raise NotImplementedError


def plot_volume(
    reference: FlowField,
    prediction: FlowField,
    grid: Grid,
    channel: Channel = Channel.PRESSURE,
    sample_index: int = 0,
) -> "pv.Plotter":
    """Linked 3D orthogonal slices of reference, prediction and |error|, returned unshown."""
    raise NotImplementedError
