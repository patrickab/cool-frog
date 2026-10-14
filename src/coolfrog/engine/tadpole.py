"""Tadpole autoencoder wrapper for FlowFields."""

from torch import nn

from coolfrog.lib.config import TadpoleConfig
from coolfrog.lib.datatypes import FlowField, LatentFlowField


class Tadpole(nn.Module):
    """Pretrained Tadpole autoencoder, one scalar-field encoding per channel."""

    def __init__(self, config: TadpoleConfig) -> None:
        raise NotImplementedError

    def encode(self, flowfield: FlowField) -> LatentFlowField:
        """Encode a batched, normalized flow field per channel, solids filled first."""
        raise NotImplementedError

    def decode(self, flowfield_latent: LatentFlowField, sample: bool = False) -> FlowField:
        """Decode to a normalized flow field, `sample=False` uses the latent mean."""
        raise NotImplementedError

    def forward(self, flowfield: FlowField, sample: bool = True) -> tuple[FlowField, LatentFlowField]:
        """Encode, then decode, the training entry point."""
        raise NotImplementedError


def _fill_solids(flowfield: FlowField) -> FlowField:
    """Fill solid voxels by repeated 3x3x3 fluid-neighbour averaging.

    Notes
    -----
    - Tadpole has no mask input, so solids need smooth values
    - `mask` stays unchanged, metrics still use fluid voxels only
    """
    raise NotImplementedError
