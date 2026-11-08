"""Frozen run configuration."""

from dataclasses import dataclass
import math
from pathlib import Path
import random
import socket
import sys
from typing import ClassVar, Literal

import numpy as np
import torch

Range = tuple[float | None, float | None]  # inclusive (min, max), None = open side
FinetuneState = Literal["frozen", "FPFT"] | int  # frozen, full-parameter fine-tuning, or LoRA rank
PATH_REPO = Path(__file__).parents[3]

# Voxels per Tadpole latent cell per axis, 2 conv + 2 transformer stride-2 layers, identical for every Tadpole size
TADPOLE_DOWNSAMPLING_FACTOR = 16

SEED = 42

# Quick end-to-end run of an experiment script: `TEST_OVERRIDES` configs, `<NN>_<name>_test/` folder, nothing pushed
IS_TEST = "--test" in sys.argv


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
    """Per-machine cache location, share of the dataset and training precision.

    Attributes
    ----------
    cache_dir : Path
        Root of every cache: metadata, voxel shards, surfaces.
    n_splits, idx_split : int
        This machine uses split `idx_split` of `n_splits`, passed to `CoolMetadata.load`, 1 split = the full dataset.
    autocast_dtype : torch.dtype | None
        torch.dtype -> training forward passes run under autocast in this dtype, e.g. bfloat16 to halve activation memory.
        None -> training runs in float32. Evaluation is float32 on every host.
    """

    cache_dir: Path
    n_splits: int
    idx_split: int
    autocast_dtype: torch.dtype | None

    @property
    def is_partial(self) -> bool:
        """Whether this machine uses only part of the dataset, its experiments get the `_partial` folder suffix."""
        return self.n_splits > 1


DEFAULT_HOST_CONFIG = HostConfig(cache_dir=PATH_REPO / ".cache", n_splits=10, idx_split=0, autocast_dtype=torch.bfloat16)
CUBE1_HOST_CONFIG = HostConfig(cache_dir=Path("/mnt/data/bitzer/.cache"), n_splits=1, idx_split=0, autocast_dtype=None)
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
    n_validation_designs : int
        Designs held out for validation by experiments that split their designs, e.g. the autoencoding experiments.

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
    n_validation_designs: int = 128

    def __post_init__(self) -> None:
        apply_test_overrides(self)


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
    encoder_crop_size : int | None
        int -> encode and decode in tiles of this edge, it must divide every longer axis
        None -> whole domain in one tile
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
    updates : int | None
        Optimizer steps, each over `designs_per_update` crops. Exactly one of `updates` and `epochs` is set.
    epochs : int | None
        Passes over the training designs, `updates` = `epochs` x `updates_per_epoch`.
    designs_per_update : int
        Crops summed into one update over all GPUs, one crop per loaded design.
        Each GPU runs `designs_per_update / (DataConfig.batch_size * n_gpus)` passes, which must divide evenly.
    learning_rate, weight_decay : float
        AdamW, linear warmup over `warmup_updates`, constant afterwards.
    kl_weight : float
        Weight of the mean elementwise KL term next to the fluid-voxel MSE.
    validate_every : int
        Updates between validation runs, update 0 = zero-shot.
    """

    updates: int | None = None
    epochs: int | None = None
    designs_per_update: int = 32
    learning_rate: float = 5e-5
    warmup_updates: int = 1000
    weight_decay: float = 1e-15
    kl_weight: float = 1e-6
    validate_every: int = 1000

    def __post_init__(self) -> None:
        apply_test_overrides(self)
        if (self.updates is None) == (self.epochs is None):
            raise ValueError(f"Set exactly one of updates ({self.updates}) and epochs ({self.epochs})")

    def updates_per_epoch(self, n_train_designs: int) -> int:
        """Updates to see every training design once, the last one may be partial."""
        return math.ceil(n_train_designs / self.designs_per_update)

    def total_updates(self, n_train_designs: int) -> int:
        """`updates`, or `epochs` x `updates_per_epoch` when epochs are set."""
        return self.updates if self.updates is not None else self.epochs * self.updates_per_epoch(n_train_designs)


TEST_OVERRIDES: dict[type, dict[str, object]] = {  # config fields under `--test`: a whole experiment in minutes
    DataConfig: {"batch_size": 2, "n_validation_designs": 4},  # 2 x 4 GPUs = 8 designs per update, also on cube1
    TrainConfig: {"updates": 32, "epochs": None, "designs_per_update": 8, "warmup_updates": 4, "validate_every": 8},
}


def apply_test_overrides(config: object) -> None:
    """Under `--test`, set the `TEST_OVERRIDES` fields of a config being constructed, they win over the given values.

    - Called by the `__post_init__` of every overridden config class, so every script gets them without own code
    """
    if IS_TEST:
        for name, value in TEST_OVERRIDES.get(type(config), {}).items():
            object.__setattr__(config, name, value)  # frozen dataclass, still in construction
