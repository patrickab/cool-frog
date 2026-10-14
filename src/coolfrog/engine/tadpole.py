"""Tadpole autoencoder wrapper for FlowFields."""

from einops import rearrange
from huggingface_hub import hf_hub_download
from tadpole.model.autoencoder import TadpoleAutoencoder
import torch
from torch import nn
import torch.nn.functional as F

from coolfrog.lib.config import TadpoleConfig
from coolfrog.lib.datatypes import FlowField, LatentFlowField

TADPOLE_REPO = "thuerey-group/Tadpole"


class Tadpole(nn.Module):
    """Pretrained Tadpole autoencoder, one scalar-field encoding per channel."""

    def __init__(self, config: TadpoleConfig) -> None:
        super().__init__()
        self.config = config
        autoencoder = TadpoleAutoencoder(
            size=config.size,
            weight_encoder=_weight_path(config, "encoder"),
            weight_decoder=_weight_path(config, "decoder"),
            encoder_ft_state=config.encoder_state,
            decoder_ft_state=config.decoder_state,
        )
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder

    def encode(self, flowfield: FlowField) -> LatentFlowField:
        """Encode a batched, normalized flow field per channel, solids filled first."""
        tiles = self._tiles_per_axis(flowfield.fields.shape[-3:])
        scalar_fields = _to_tiles(_fill_solids(flowfield).fields.unsqueeze(2), tiles)
        distributions = [self.encoder(chunk, "distribution") for chunk in self._chunks(scalar_fields)]
        batch, channels = flowfield.fields.shape[:2]
        latent_mean = _from_tiles(torch.cat([distribution.mean for distribution in distributions]), batch, channels, tiles)
        latent_logvar = _from_tiles(torch.cat([distribution.logvar for distribution in distributions]), batch, channels, tiles)
        return LatentFlowField(
            latent_mean=latent_mean,
            latent_logvar=latent_logvar,
            mask=flowfield.mask,
            design_id=flowfield.design_id,
            params=flowfield.params,
            stats=flowfield.stats,
            batch_size=flowfield.batch_size,
        )

    def decode(self, flowfield_latent: LatentFlowField, sample: bool = False) -> FlowField:
        """Decode to a normalized flow field, `sample=False` uses the latent mean."""
        latent = flowfield_latent.latent_mean
        if sample:
            latent = latent + torch.randn_like(latent) * (0.5 * flowfield_latent.latent_logvar).exp()
        tiles = self._tiles_per_axis(flowfield_latent.mask.shape[-3:])
        decoded = torch.cat([self.decoder(chunk) for chunk in self._chunks(_to_tiles(latent, tiles))])
        batch, channels = latent.shape[:2]
        return FlowField(
            fields=_from_tiles(decoded, batch, channels, tiles).squeeze(2),
            mask=flowfield_latent.mask,
            design_id=flowfield_latent.design_id,
            params=flowfield_latent.params,
            stats=flowfield_latent.stats,
            batch_size=flowfield_latent.batch_size,
        )

    def forward(self, flowfield: FlowField, sample: bool = True) -> tuple[FlowField, LatentFlowField]:
        """Encode, then decode, the training entry point."""
        flowfield_latent = self.encode(flowfield)
        return self.decode(flowfield_latent, sample), flowfield_latent

    def _tiles_per_axis(self, resolution: torch.Size) -> tuple[int, int, int]:
        """Crop tiles along x, y, z, one tile per axis without `encoder_crop_size` or below it."""
        crop_size = self.config.encoder_crop_size
        if crop_size is None:
            return 1, 1, 1
        if any(voxels > crop_size and voxels % crop_size for voxels in resolution):
            raise ValueError(f"encoder_crop_size {crop_size} must divide every axis above it, got {tuple(resolution)}")
        tiles_x, tiles_y, tiles_z = (max(voxels // crop_size, 1) for voxels in resolution)
        return tiles_x, tiles_y, tiles_z

    def _chunks(self, scalar_fields: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split along the scalar-field axis into `max_internal_batchsize` chunks."""
        if self.config.max_internal_batchsize is None:
            return (scalar_fields,)
        return torch.split(scalar_fields, self.config.max_internal_batchsize)


def _weight_path(config: TadpoleConfig, part: str) -> str:
    """Local or Hugging Face path of `tadpole_<size>_<part>.safetensors`."""
    filename = f"tadpole_{config.size.lower()}_{part}.safetensors"
    if config.path_weights is not None:
        return str(config.path_weights / filename)
    return hf_hub_download(TADPOLE_REPO, filename)


def _to_tiles(volumes: torch.Tensor, tiles: tuple[int, int, int]) -> torch.Tensor:
    """(B, C, N, X, Y, Z) -> (B * C * tiles, N, X/U, Y/V, Z/W) scalar-field crops."""
    tiles_x, tiles_y, tiles_z = tiles
    return rearrange(volumes, "B C N (U x) (V y) (W z) -> (B C U V W) N x y z", U=tiles_x, V=tiles_y, W=tiles_z)


def _from_tiles(crops: torch.Tensor, batch: int, channels: int, tiles: tuple[int, int, int]) -> torch.Tensor:
    """Inverse of `_to_tiles`."""
    tiles_x, tiles_y, tiles_z = tiles
    return rearrange(crops, "(B C U V W) N x y z -> B C N (U x) (V y) (W z)", B=batch, C=channels, U=tiles_x, V=tiles_y, W=tiles_z)


def _fill_solids(flowfield: FlowField) -> FlowField:
    """Fill solid voxels by repeated 3x3x3 fluid-neighbour averaging.

    Notes
    -----
    - Tadpole has no mask input, so solids need smooth values
    - `mask` stays unchanged, metrics still use fluid voxels only
    """
    known = flowfield.mask.unsqueeze(-4).expand_as(flowfield.fields).to(flowfield.fields.dtype)
    values = flowfield.fields * known
    while True:
        neighbour_count = F.avg_pool3d(known, 3, 1, 1)
        newly_filled = (known == 0) & (neighbour_count > 0)
        if not newly_filled.any():
            return flowfield.replace(fields=values)
        values = torch.where(newly_filled, F.avg_pool3d(values, 3, 1, 1) / neighbour_count.clamp_min(1e-6), values)
        known = torch.where(newly_filled, 1.0, known)
