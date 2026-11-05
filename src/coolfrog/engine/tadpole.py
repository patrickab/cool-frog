"""Tadpole autoencoder wrapper for FlowFields."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

from einops import rearrange
from GIFt.modules.lora import LoRALinearLike
import httpx
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from tadpole.architecture.p3d.conv import ConditionedDecoder3D, ConditionedEncoder3D, ConditionedEncoder3DBlock
from tadpole.architecture.p3d.transformer import PDEDiTBlock
from tadpole.model.autoencoder import TadpoleAutoencoder
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from coolfrog.lib.config import TadpoleConfig
from coolfrog.lib.datatypes import FlowField, LatentFlowField
from coolfrog.lib.log import get_logger

logger = get_logger(__name__)

TADPOLE_REPO = "thuerey-group/Tadpole"
# must match the `tadpole` code rev in pyproject.toml, weight revisions change state dict keys
TADPOLE_REVISION = "9119e886f50ff8c941955dfad0b65c18c2694d1d"
# full-resolution voxels per stem/head slab, bounds the widest feature maps, 2^21 = 0.5 GiB of 64 float32 channels
SLAB_VOXELS = 2**21


class Tadpole(nn.Module):
    """Pretrained Tadpole autoencoder, one scalar-field encoding per channel."""

    def __init__(self, config: TadpoleConfig) -> None:
        super().__init__()
        self.config = config
        if config.path_weights is None:
            _warn_if_newer_weights()
        autoencoder = TadpoleAutoencoder(
            size=config.size,
            weight_encoder=_weight_path(config, "encoder"),
            weight_decoder=_weight_path(config, "decoder"),
            encoder_ft_state=config.encoder_state,
            decoder_ft_state=config.decoder_state,
            encoder_crop_size=None,
        )
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        _slab_full_resolution(self.encoder.conv_encoder, self.decoder.conv_decoder, config.activation_checkpointing)
        if config.activation_checkpointing:
            _checkpoint_blocks(self)

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

    def train(self, mode: bool = True) -> "Tadpole":
        """Set train mode, LoRA weights merged by an eval forward are unmerged here in float32.

        - GIFt otherwise unmerges lazily in the next training forward, under bf16 autocast and inside checkpointed blocks
        - That leaves bf16 rounding in the frozen weights and saves different tensors in forward and recomputation
        """
        super().train(mode)
        if mode:
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                for module in self.modules():
                    if isinstance(module, LoRALinearLike) and module.merged:
                        module.unmerge_weight()
        return self

    def trainable_parameters(self) -> dict[str, nn.Parameter]:
        """Parameters with requires_grad: LoRA matrices plus the biases GIFt leaves trainable, all under FPFT."""
        return {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}

    def adapter(self) -> dict[str, torch.Tensor]:
        """`trainable_parameters` detached on the CPU, about 11 MB at LoRA-32, written as `adapter.pt`."""
        return {name: parameter.detach().cpu() for name, parameter in self.trainable_parameters().items()}

    def load_adapter(self, path: Path) -> None:
        """Load an `adapter.pt` into a model built with the same `TadpoleConfig`, ValueError on mismatching names or shapes."""
        adapter = torch.load(path, map_location="cpu")
        expected_shapes = {name: parameter.shape for name, parameter in self.trainable_parameters().items()}
        if {name: tensor.shape for name, tensor in adapter.items()} != expected_shapes:
            raise ValueError(f"adapter {path} does not match the trainable parameters of this TadpoleConfig, e.g. its LoRA rank")
        self.load_state_dict(adapter, strict=False)

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


def _warn_if_newer_weights() -> None:
    """Warn when the Hugging Face weights moved past `TADPOLE_REVISION`, silent when offline."""
    try:
        latest_revision = HfApi().model_info(TADPOLE_REPO).sha
    except (HfHubHTTPError, httpx.HTTPError, OSError):
        return
    if latest_revision != TADPOLE_REVISION:
        logger.warning(
            "Newer Tadpole weights on Hugging Face: %s (pinned %s). "
            "Bump TADPOLE_REVISION together with the `tadpole` rev in pyproject.toml",
            latest_revision,
            TADPOLE_REVISION,
        )


def _checkpoint_blocks(model: nn.Module) -> None:
    """Recompute each conv and attention block in backward, parameter names unchanged."""
    for module in model.modules():
        if isinstance(module, ConditionedEncoder3DBlock):
            module.forward = partial(checkpoint, partial(_conv_block_forward, module), use_reentrant=False)
        elif isinstance(module, PDEDiTBlock):
            module.forward = partial(checkpoint, module.forward, use_reentrant=False)


def _conv_block_forward(block: ConditionedEncoder3DBlock, features: torch.Tensor) -> torch.Tensor:
    """`ConditionedEncoder3DBlock.forward` with nested checkpointed norm-activations, the convs keep only their bf16 inputs."""
    hidden = block.conv_1(checkpoint(_norm_activation, block.gn_1, block.activation_1, features, use_reentrant=False))
    hidden = block.conv_2(checkpoint(_norm_activation, block.gn_2, block.activation_2, hidden, use_reentrant=False))
    return hidden + features


def _norm_activation(norm: nn.Module, activation: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """GroupNorm then GELU, cast to the autocast dtype here, so the float32 result never outlives the recomputation."""
    hidden = activation(norm(features))
    return hidden.to(torch.get_autocast_dtype("cuda")) if torch.is_autocast_enabled("cuda") else hidden


def _slab_full_resolution(conv_encoder: ConditionedEncoder3D, conv_decoder: ConditionedDecoder3D, checkpointed: bool) -> None:
    """Run the full-resolution encoder stem and decoder head in x-slabs, each slab recomputed in backward when `checkpointed`.

    - Stem `feature_embed -> downsampling_layers[0]` and head `upsampling_layers[-1] -> decompress` hold the widest feature maps
    - Exact: both are zero-padded convs without GroupNorm, a one-cell halo covers their reach
    """
    conv_encoder.forward = partial(_conv_encoder_forward, conv_encoder, checkpointed)
    conv_decoder.forward = partial(_conv_decoder_forward, conv_decoder, checkpointed)


def _conv_encoder_forward(conv_encoder: ConditionedEncoder3D, checkpointed: bool, field: torch.Tensor) -> torch.Tensor:
    """`ConditionedEncoder3D.forward` with the stem in x-slabs."""
    features = _encoder_stem_in_slabs(conv_encoder, field, checkpointed)
    for level in range(conv_encoder.num_downsampling_layers - 1):
        for repetition in range(conv_encoder.repetitions):
            features = conv_encoder.blocks[level * conv_encoder.repetitions + repetition](features)
        features = conv_encoder.downsampling_layers[level + 1](features)
    return features


def _conv_decoder_forward(conv_decoder: ConditionedDecoder3D, checkpointed: bool, features: torch.Tensor) -> torch.Tensor:
    """`ConditionedDecoder3D.forward` with the head in x-slabs."""
    features = conv_decoder.upsampling_layers[0](features)
    for level in range(conv_decoder.num_upsampling_layers - 1):
        for repetition in range(conv_decoder.repetitions):
            features = conv_decoder.blocks[level * conv_decoder.repetitions + repetition](features)
        if level < conv_decoder.num_upsampling_layers - 2:
            features = conv_decoder.upsampling_layers[level + 1](features)
    return _decoder_head_in_slabs(conv_decoder, features, checkpointed)


def _encoder_stem_in_slabs(conv_encoder: ConditionedEncoder3D, field: torch.Tensor, checkpointed: bool) -> torch.Tensor:
    """(N, 1, X, Y, Z) field -> (N, C, X/2, Y/2, Z/2) stem features, output cell i reads field planes 2i-2 to 2i+2."""
    voxels_x = field.shape[-3]
    slabs = []
    for start, stop in _slab_ranges(voxels_x // 2, field.shape[-2] * field.shape[-1]):
        input_start, input_stop = max(2 * start - 2, 0), min(2 * stop + 1, voxels_x)
        features = _run(_encoder_stem, checkpointed, conv_encoder, field[..., input_start:input_stop, :, :])
        slabs.append(features[..., start - input_start // 2 : stop - input_start // 2, :, :])
    return torch.cat(slabs, dim=-3)


def _decoder_head_in_slabs(conv_decoder: ConditionedDecoder3D, features: torch.Tensor, checkpointed: bool) -> torch.Tensor:
    """(N, C, X/2, Y/2, Z/2) features -> (N, 1, X, Y, Z) field, output planes 2i and 2i+1 read feature cells i-1 to i+1."""
    cells_x = features.shape[-3]
    slabs = []
    for start, stop in _slab_ranges(cells_x, 4 * features.shape[-2] * features.shape[-1]):
        input_start, input_stop = max(start - 1, 0), min(stop + 1, cells_x)
        field = _run(_decoder_head, checkpointed, conv_decoder, features[..., input_start:input_stop, :, :])
        slabs.append(field[..., 2 * (start - input_start) : 2 * (stop - input_start), :, :])
    return torch.cat(slabs, dim=-3)


def _encoder_stem(conv_encoder: ConditionedEncoder3D, field: torch.Tensor) -> torch.Tensor:
    return conv_encoder.downsampling_layers[0](conv_encoder.feature_embed(field))


def _decoder_head(conv_decoder: ConditionedDecoder3D, features: torch.Tensor) -> torch.Tensor:
    return conv_decoder.decompress(conv_decoder.upsampling_layers[-1](features))


def _slab_ranges(cells_x: int, voxels_per_plane: int) -> list[tuple[int, int]]:
    """Half-resolution [start, stop) x-ranges of about `SLAB_VOXELS` full-resolution voxels, two planes per cell."""
    cells_per_slab = max(1, SLAB_VOXELS // (2 * voxels_per_plane))
    return [(start, min(start + cells_per_slab, cells_x)) for start in range(0, cells_x, cells_per_slab)]


def _run(function: Callable[..., torch.Tensor], checkpointed: bool, *args: object) -> torch.Tensor:
    """Call one slab, without the autocast weight cache, so slab gradients of a parameter sum in float32, not its bf16 copy."""
    with torch.autocast("cuda", enabled=torch.is_autocast_enabled("cuda"), cache_enabled=False):
        return checkpoint(function, *args, use_reentrant=False) if checkpointed else function(*args)


def _weight_path(config: TadpoleConfig, part: str) -> str:
    """Local or Hugging Face path of `tadpole_<size>_<part>.safetensors`."""
    filename = f"tadpole_{config.size.lower()}_{part}.safetensors"
    if config.path_weights is not None:
        return str(config.path_weights / filename)
    return hf_hub_download(TADPOLE_REPO, filename, revision=TADPOLE_REVISION)


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
    - The fill front is shared by all channels and computed once, channels are filled one at a time in place:
      peak memory ~2 full-domain fields instead of ~6, 1600x320x64 needs it
    """
    known = flowfield.mask.unsqueeze(-4).to(flowfield.fields.dtype)  # (..., 1, X, Y, Z)
    values = flowfield.fields * known
    while True:
        neighbour_count = F.avg_pool3d(known, 3, 1, 1)
        newly_filled = (known == 0) & (neighbour_count > 0)
        if not newly_filled.any():
            return flowfield.replace(fields=values)
        neighbour_count = neighbour_count.clamp_min(1e-6)
        for channel in range(values.shape[-4]):
            channel_values = values[..., channel : channel + 1, :, :, :]
            channel_values.copy_(torch.where(newly_filled, F.avg_pool3d(channel_values, 3, 1, 1) / neighbour_count, channel_values))
        known = torch.where(newly_filled, 1.0, known)
