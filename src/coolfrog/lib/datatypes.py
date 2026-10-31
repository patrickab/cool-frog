"""Core datatypes shared across coolfrog."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from tensordict import tensorclass
import torch

from coolfrog.lib.log import track

if TYPE_CHECKING:
    from coolfrog.engine.cooldata import CoolMetadata
    from coolfrog.lib.config import DataConfig


class Channel(StrEnum):
    """Volume field channel in model order, value is the display label with unit."""

    VX = "Vx [m/s]"
    VY = "Vy [m/s]"
    VZ = "Vz [m/s]"
    PRESSURE = "p [Pa]"
    TEMPERATURE = "T [K]"
    TURBULENT_KINETIC_ENERGY = "k [m²/s²]"
    TURBULENT_DISSIPATION_RATE = "ε [m²/s³]"


class SurfaceChannel(StrEnum):
    """Per-face boundary quantity, value is the display label with unit, NaN where a boundary doesn't store it."""

    HEAT_TRANSFER_COEFFICIENT = "h [W/(m²·K)]"
    WALL_SHEAR_STRESS_X = "τx [Pa]"
    WALL_SHEAR_STRESS_Y = "τy [Pa]"
    WALL_SHEAR_STRESS_Z = "τz [Pa]"
    PRESSURE = "p [Pa]"
    TEMPERATURE = "T [K]"


# Order of the C axis in `FlowField.fields`
CHANNELS: tuple[Channel, ...] = tuple(Channel)
SURFACE_CHANNELS: tuple[SurfaceChannel, ...] = tuple(SurfaceChannel)

# Column names of the P axis in `FlowField.params`, in cooldata v2 metadata order, bodies 1-4 are quads, 5-6 cylinders
# Defines simulation input.
# https://github.com/peteole/flow_field_dataset/blob/a1498c97e5dc19349dd9670793bc84b033f8f1d7/cooldata/metadata.py#L126-L128
PARAM_COLUMNS: tuple[str, ...] = (
    *(f"T{body}" for body in range(1, 7)),  # body temperature
    *(f"x{body}" for body in range(1, 7)),  # body position along the flow
    *(f"y{body}" for body in range(1, 7)),  # body position across the flow, 1.0 = inactive body
    *(f"xs{quad}" for quad in range(1, 5)),  # quad size along x
    *(f"ys{quad}" for quad in range(1, 5)),  # quad size along y
    *(f"zs{body}" for body in range(1, 7)),  # body height
    "r5",  # cylinder 5 radius
    "r6",  # cylinder 6 radius
    "V",  # inlet velocity
)

# Units of `PARAM_COLUMNS` where known, others are unitless or undocumented upstream
PARAM_UNITS: dict[str, str] = {"V": "m/s"}


@dataclass(frozen=True)
class Grid:
    """Voxel grid of the channel domain, shared by all samples.

    - `latent_shape = aspect_ratio * latent_scale`, Tadpole latent cells along x, y, z
    - `shape = latent_shape * downsampling_factor`, voxels along x, y, z
    - Every axis is a whole number of latent cells, so reconstructions keep the input shape

    Attributes
    ----------
    latent_scale : int
        Multiplier of the smallest valid grid, 1 -> latent (25, 5, 1), voxels (400, 80, 16).
    aspect_ratio : tuple[int, int, int]
        Domain proportions x:y:z, matches `extent`.
    downsampling_factor : int
        Voxels per Tadpole latent cell along each axis, 2^4 from its four stride-2 layers.
    extent : tuple[float, float, float]
        Domain size in m along x (flow), y (width), z (height).
    """

    latent_scale: int
    aspect_ratio: tuple[int, int, int] = (25, 5, 1)
    downsampling_factor: int = 16
    extent: tuple[float, float, float] = (0.5, 0.1, 0.02)

    def __post_init__(self) -> None:
        """Raise ValueError unless `latent_scale` is positive."""
        if self.latent_scale < 1:
            raise ValueError(f"latent_scale must be a positive integer, got {self.latent_scale}")

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        """Tadpole latent cells along x, y, z."""
        cells_x, cells_y, cells_z = (ratio * self.latent_scale for ratio in self.aspect_ratio)
        return cells_x, cells_y, cells_z

    @property
    def shape(self) -> tuple[int, int, int]:
        """Voxels along x, y, z."""
        voxels_x, voxels_y, voxels_z = (cells * self.downsampling_factor for cells in self.latent_shape)
        return voxels_x, voxels_y, voxels_z

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Voxel spacing in m, `extent / (shape - 1)` like cooldata."""
        spacing_x, spacing_y, spacing_z = (length / (voxels - 1) for length, voxels in zip(self.extent, self.shape, strict=True))
        return spacing_x, spacing_y, spacing_z


@tensorclass
class NormalizationStats:
    """Global per-channel fluid min and max over a design selection.

    Attributes
    ----------
    channel_min : torch.Tensor
        (..., C, 1, 1, 1)
    channel_max : torch.Tensor
        (..., C, 1, 1, 1)
    """

    channel_min: torch.Tensor
    channel_max: torch.Tensor

    @classmethod
    def compute(cls, metadata: "CoolMetadata", config: "DataConfig") -> "NormalizationStats":
        """Fluid min and max of every `metadata` design on `config.grid`, read with `config` batching and workers.

        - Voxelizes uncached designs first, like `CoolData`
        - Same stats at another resolution need the same `metadata` on that grid
        """
        from coolfrog.engine.cooldata import CoolData, create_dataloader  # noqa: PLC0415, engine imports this module

        loader = create_dataloader(CoolData(metadata, config), config, shuffle=False)
        return cls.from_flowfields(track(loader, f"Normalization stats {'x'.join(map(str, config.grid.shape))}"))

    @classmethod
    def from_flowfields(cls, flowfields: Iterable["FlowField"]) -> "NormalizationStats":
        """Per-channel fluid min and max over physical flow fields, reduced batch by batch, ValueError for a constant channel."""
        channel_min, channel_max = None, None
        for flowfield in flowfields:
            fluid = flowfield.mask.unsqueeze(-4)
            batch_min = _reduce_to_channels(torch.where(fluid, flowfield.fields, torch.inf), torch.amin)
            batch_max = _reduce_to_channels(torch.where(fluid, flowfield.fields, -torch.inf), torch.amax)
            channel_min = batch_min if channel_min is None else torch.minimum(channel_min, batch_min)
            channel_max = batch_max if channel_max is None else torch.maximum(channel_max, batch_max)
        if channel_min is None or channel_max is None:
            raise ValueError("no flow fields to reduce")
        if (channel_max <= channel_min).any():
            raise ValueError(f"constant or fluid-free channel, min {channel_min.tolist()}, max {channel_max.tolist()}")
        return cls(channel_min=channel_min.view(-1, 1, 1, 1), channel_max=channel_max.view(-1, 1, 1, 1), batch_size=[])

    def save(self, path: Path) -> None:
        """Write `channel`, `min`, `max` as CSV, one row per channel."""
        table = pl.DataFrame(
            {
                "channel": [channel.value for channel in CHANNELS],
                "min": self.channel_min.flatten().tolist(),
                "max": self.channel_max.flatten().tolist(),
            }
        )
        table.write_csv(path)

    @classmethod
    def load(cls, path: Path) -> "NormalizationStats":
        """Read a `save` CSV, ValueError unless its channels match `CHANNELS`."""
        table = pl.read_csv(path)
        if table["channel"].to_list() != [channel.value for channel in CHANNELS]:
            raise ValueError(f"channels in {path} differ from CHANNELS: {table['channel'].to_list()}")
        channel_min, channel_max = (table[column].to_torch().float().view(-1, 1, 1, 1) for column in ("min", "max"))
        return cls(channel_min=channel_min, channel_max=channel_max, batch_size=[])

    def select_channels(self, channel_index: torch.Tensor) -> "NormalizationStats":
        """Stats of single-channel fields batched along `channel_index`, e.g. crops: (N,) -> batch [N], (N, 1, 1, 1, 1)."""
        return NormalizationStats(
            channel_min=self.channel_min[channel_index].unsqueeze(1),
            channel_max=self.channel_max[channel_index].unsqueeze(1),
            batch_size=[len(channel_index)],
        )


def _reduce_to_channels(fields: torch.Tensor, reduction: Callable[..., torch.Tensor]) -> torch.Tensor:
    """Reduce (..., C, X, Y, Z) over every axis but C."""
    return reduction(fields.movedim(-4, 0).flatten(1), dim=1)


@tensorclass
class FlowField:
    """Voxelized flow field in Tadpole layout, physical units until normalized.

    Attributes
    ----------
    fields : torch.Tensor
        (..., C, X, Y, Z) float32, `CHANNELS` order.
    mask : torch.Tensor
        (..., X, Y, Z) bool, True = fluid.
    design_id : torch.Tensor
        (...,) int64
    params : torch.Tensor
        (..., P) float32, `PARAM_COLUMNS` order.
    stats : NormalizationStats | None
        Set while normalized.
    """

    fields: torch.Tensor
    mask: torch.Tensor
    design_id: torch.Tensor
    params: torch.Tensor
    stats: NormalizationStats | None = None

    def normalize(self, stats: NormalizationStats) -> "FlowField":
        """Scale fields to [-1, 1] and attach `stats`, one broadcast over the whole batch."""
        if self.stats is not None:
            raise ValueError("flow field is already normalized")
        stats = stats.to(self.fields.device).expand(self.batch_size)
        fields = 2 * (self.fields - stats.channel_min) / (stats.channel_max - stats.channel_min) - 1
        return self.replace(fields=fields, stats=stats)

    def denormalize(self) -> "FlowField":
        """Restore physical units and drop `stats`."""
        if self.stats is None:
            raise ValueError("flow field is not normalized")
        fields = (self.fields + 1) * (self.stats.channel_max - self.stats.channel_min) / 2 + self.stats.channel_min
        return self.replace(fields=fields, stats=None)


@tensorclass
class LatentFlowField:
    """Tadpole latent of a FlowField, one scalar-field latent per channel.

    Attributes
    ----------
    latent_mean, latent_logvar : torch.Tensor
        (..., C, L, X/16, Y/16, Z/16)
    mask : torch.Tensor
        (..., X, Y, Z) bool, full-resolution fluid mask restored on decode.
    design_id : torch.Tensor
        (...,) int64
    params : torch.Tensor
        (..., P) float32, `PARAM_COLUMNS` order.
    stats : NormalizationStats | None
        Carried through so decode can restore physical units.
    """

    latent_mean: torch.Tensor
    latent_logvar: torch.Tensor
    mask: torch.Tensor
    design_id: torch.Tensor
    params: torch.Tensor
    stats: NormalizationStats | None = None

    def kl_divergence(self) -> torch.Tensor:
        """Elementwise KL divergence to the standard normal prior."""
        latent_mean, latent_logvar = self.latent_mean.float(), self.latent_logvar.float()
        return 0.5 * (latent_mean.square() + latent_logvar.exp() - 1 - latent_logvar)
