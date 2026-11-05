"""Frozen run configuration."""

from dataclasses import dataclass
from pathlib import Path
import random
import socket
from typing import ClassVar, Literal

import numpy as np
import torch

Range = tuple[float | None, float | None]  # inclusive (min, max), None = open side
FinetuneState = Literal["frozen", "FPFT"] | int  # frozen, full-parameter fine-tuning, or LoRA rank
PATH_REPO = Path(__file__).parents[3]

# Voxels per Tadpole latent cell per axis, 2 conv + 2 transformer stride-2 layers, identical for every Tadpole size
TADPOLE_DOWNSAMPLING_FACTOR = 16

# Single seed of the application: python, numpy and torch RNGs, dataset splits and samples
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Seed python, numpy and torch on CPU and every GPU, loader workers derive their seeds from torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class Grid:
    """Voxel grid of the channel domain, shared by all samples.

    - `latent_shape = aspect_ratio * latent_scale`, Tadpole latent cells along x, y, z
    - `shape = latent_shape * TADPOLE_DOWNSAMPLING_FACTOR`, voxels along x, y, z
    - Every axis is a whole number of latent cells, so reconstructions keep the input shape
    - `aspect_ratio` and `extent` are fixed by the CoolData domain, class constants kept out of saved configs

    Attributes
    ----------
    latent_scale : int
        Multiplier of the smallest valid grid, 1 -> latent (25, 5, 1), voxels (400, 80, 16).
    """

    aspect_ratio: ClassVar[tuple[int, int, int]] = (25, 5, 1)  # domain proportions x:y:z, matches `extent`
    extent: ClassVar[tuple[float, float, float]] = (0.5, 0.1, 0.02)  # domain size in m along x (flow), y (width), z (height)

    latent_scale: int

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
        voxels_x, voxels_y, voxels_z = (cells * TADPOLE_DOWNSAMPLING_FACTOR for cells in self.latent_shape)
        return voxels_x, voxels_y, voxels_z

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Voxel spacing in m, `extent / (shape - 1)` like cooldata."""
        spacing_x, spacing_y, spacing_z = (length / (voxels - 1) for length, voxels in zip(self.extent, self.shape, strict=True))
        return spacing_x, spacing_y, spacing_z


@dataclass(frozen=True)
class HostConfig:
    """Per-machine cache location and share of the dataset.

    Attributes
    ----------
    cache_dir : Path
        Root of every cache: metadata, voxel shards, surfaces.
    n_splits, idx_split : int
        This machine uses split `idx_split` of `n_splits`, passed to `CoolMetadata.load`, 1 split = the full dataset.
    """

    cache_dir: Path
    n_splits: int
    idx_split: int

    @property
    def is_partial(self) -> bool:
        """Whether this machine uses only part of the dataset, its experiments get the `_partial` folder suffix."""
        return self.n_splits > 1


DEFAULT_HOST_CONFIG = HostConfig(cache_dir=PATH_REPO / ".cache", n_splits=10, idx_split=0)
CUBE1_HOST_CONFIG = HostConfig(cache_dir=Path("/mnt/data/bitzer/.cache"), n_splits=1, idx_split=0)
HOST_CONFIG = CUBE1_HOST_CONFIG if socket.gethostname() == "cube1" else DEFAULT_HOST_CONFIG


@dataclass(frozen=True)
class DataConfig:
    """Data source, caching and batching, the design selection comes as `CoolMetadata`.

    Attributes
    ----------
    grid : Grid
        Voxel grid every design is sampled onto.
    source : Path | None
        Path -> Local CoolData root holding `volume/volume_design_<design_id:07d>_p.cgns`.
        None -> Hugging Face, each design extracted from its `runs/<run>/<batch>.zip`.
    cache : bool
        True -> voxelize uncached designs on construction into HDF5 shards in
        `HOST_CONFIG.cache_dir / "voxels_<X>x<Y>x<Z>"`, then read from them.
        False -> voxelize from the local `source` on every access, nothing written.
    batch_size : int
        Designs per batch.
    num_workers : int
        Loader worker processes.

    Examples
    --------
    >>> config = DataConfig(grid=Grid(latent_scale=2))
    >>> metadata = CoolMetadata.load(HOST_CONFIG.n_splits, HOST_CONFIG.idx_split)
    >>> metadata = metadata.select(ranges={"V": (2.0, 5.0), "T1": (None, 60.0)})
    >>> train, val = CoolData(metadata, config).split([0.9, 0.1])  # voxelizes uncached designs first
    >>> train_loader = create_dataloader(train, config, shuffle=True)
    """

    grid: Grid
    source: Path | None = None
    cache: bool = True
    batch_size: int = 32
    num_workers: int = 0


@dataclass(frozen=True)
class TadpoleConfig:
    """Pretrained Tadpole autoencoder setup.

    Attributes
    ----------
    size : Literal["S", "B", "L"]
        Model size.
    path_weights : Path | None
        Holds `tadpole_<size>_{encoder,decoder}.safetensors`, None = Hugging Face.
    encoder_state, decoder_state : FinetuneState
        "frozen", "FPFT" (full-parameter fine-tuning) or a LoRA rank.
    encoder_crop_size: int | None
        int -> sets crop size for voxels
        None -> sets crop size to 1e6
    max_internal_batchsize : int | None
        Scalar fields per encoder/decoder call, None = all at once, 1 for whole-domain evaluation from latent scale 3 on.
    activation_checkpointing : bool
        Recompute conv blocks, norm-activations, attention blocks and full-resolution slabs in backward, ~25% longer steps.
    """

    size: Literal["S", "B", "L"] = "B"
    path_weights: Path | None = None
    encoder_state: FinetuneState = "frozen"
    decoder_state: FinetuneState = "frozen"
    encoder_crop_size: int | None = None
    max_internal_batchsize: int | None = None
    activation_checkpointing: bool = False


@dataclass(frozen=True)
class TrainConfig:
    """Autoencoding fine-tuning on random single-channel crops, defaults from Tadpole Appendix C.3.

    - Crop edge is `TadpoleConfig.encoder_crop_size`, so training crops equal validation tiles, None = whole domain

    Attributes
    ----------
    updates : int
        Optimizer steps, each over `crops_per_step * accumulation_steps` crops.
    crops_per_step : int
        Single-channel crops per forward/backward pass, bounded by VRAM (8 at 64³ on 12 GB), 1 for whole domains.
    accumulation_steps : int
        Passes summed into one update, effective batch = `crops_per_step * accumulation_steps`.
    learning_rate, weight_decay : float
        AdamW, linear warmup over `warmup_updates`, constant afterwards.
    kl_weight : float
        Weight of the mean elementwise KL term next to the fluid-voxel MSE.
    validate_every : int
        Updates between validation runs, update 0 = zero-shot.
    n_validation_designs : int
        Held-out designs, never cropped for training.
    """

    updates: int = 14_000
    crops_per_step: int = 8
    accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_updates: int = 1000
    weight_decay: float = 1e-15
    kl_weight: float = 1e-6
    validate_every: int = 1000
    n_validation_designs: int = 128
