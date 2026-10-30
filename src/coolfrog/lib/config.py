"""Frozen run configuration."""

from dataclasses import dataclass
from pathlib import Path
import socket
from typing import Literal

from coolfrog.lib.datatypes import Grid

Range = tuple[float | None, float | None]  # inclusive (min, max), None = open side
FinetuneState = Literal["frozen", "FPFT"] | int  # frozen, full-parameter fine-tuning, or LoRA rank


@dataclass(frozen=True)
class HostConfig:
    """Per-machine cache location, voxel grid and dataset split.

    Attributes
    ----------
    cache_dir : Path
        Root of every cache: metadata, voxel shards, surfaces.
    grid : Grid
        Default voxel grid of `DataConfig`.
    n_splits, idx_split : int
        This machine voxelizes split `idx_split` of `n_splits`, passed to `CoolMetadata.load`.
    """

    cache_dir: Path
    grid: Grid
    n_splits: int
    idx_split: int


DEFAULT_HOST_CONFIG = HostConfig(cache_dir=Path.cwd() / ".cache", grid=Grid(resolution=(400, 80, 16)), n_splits=10, idx_split=0)
CUBE1_HOST_CONFIG = HostConfig(
    cache_dir=Path("/mnt/data/bitzer/.cache"), grid=Grid(resolution=(800, 160, 32)), n_splits=1, idx_split=0
)
HOST_CONFIG = CUBE1_HOST_CONFIG if socket.gethostname() == "cube1" else DEFAULT_HOST_CONFIG


@dataclass(frozen=True)
class DataConfig:
    """Data source, caching and batching, the design selection comes as `CoolMetadata`.

    Attributes
    ----------
    grid : Grid
        Voxel grid every design is sampled onto, defaults to this machine's `HOST_CONFIG.grid`.
    source : Path | None
        Path -> Local CoolData root with `volume/volume_design_<design_id:07d>_p.cgns`
        None -> Hugging Face.
    cache : bool
        True -> voxelize uncached designs into HDF5 shards in `CACHE_DIR` on construction, then read from disk.
        False -> voxelize from `source` on every access, nothing written.
    seed : int
        Seeds `CoolData.split`.
    batch_size : int
        Designs per batch.
    num_workers : int
        Loader worker processes.

    Examples
    --------
    >>> config = DataConfig()  # grid of `HOST_CONFIG`
    >>> metadata = CoolMetadata.load(HOST_CONFIG.n_splits, HOST_CONFIG.idx_split).select(ranges={"V": (2.0, 5.0), "T1": (None, 60.0)})
    >>> train, val = CoolData(metadata, config).split([0.9, 0.1])  # voxelizes uncached designs first
    >>> train_loader = create_dataloader(train, config, shuffle=True)
    """

    grid: Grid = HOST_CONFIG.grid
    source: Path | None = None
    cache: bool = True
    seed: int = 42
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
        Scalar fields per encoder/decoder call, None = all at once.
    """

    size: Literal["S", "B", "L"] = "B"
    path_weights: Path | None = None
    encoder_state: FinetuneState = "frozen"
    decoder_state: FinetuneState = "frozen"
    encoder_crop_size: int | None = None
    max_internal_batchsize: int | None = None
