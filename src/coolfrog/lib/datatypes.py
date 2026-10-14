"""Core datatypes shared across coolfrog."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from tensordict import tensorclass
import torch


class Channel(StrEnum):
    """Physical channel in model order, value is the display label with unit."""

    VX = "Vx [m/s]"
    VY = "Vy [m/s]"
    VZ = "Vz [m/s]"
    PRESSURE = "p [Pa]"
    TEMPERATURE = "T [K]"


# Order of the C axis in `FlowField.fields`
CHANNELS: tuple[Channel, ...] = tuple(Channel)

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


@dataclass(frozen=True)
class Grid:
    """Voxel grid of the channel domain, shared by all samples.

    Attributes
    ----------
    resolution : tuple[int, int, int]
        Voxels along x, y, z, each a multiple of 16.
    extent : tuple[float, float, float]
        Domain size in m along x (flow), y (width), z (height).
    """

    resolution: tuple[int, int, int]
    extent: tuple[float, float, float] = (0.5, 0.1, 0.02)

    def __post_init__(self) -> None:
        """Raise ValueError unless every axis is a multiple of 16."""
        raise NotImplementedError

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Voxel spacing in m, `extent / (resolution - 1)` like cooldata."""
        raise NotImplementedError


@tensorclass
class NormalizationStats:
    """Global per-channel fluid min and max of the training split.

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
    def from_flowfields(cls, flowfields: Iterable["FlowField"]) -> "NormalizationStats":
        """Per-channel fluid min and max over physical flow fields, reduced batch by batch, ValueError for a constant channel."""
        raise NotImplementedError


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
        raise NotImplementedError

    def denormalize(self) -> "FlowField":
        """Restore physical units and drop `stats`."""
        raise NotImplementedError


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
        raise NotImplementedError
