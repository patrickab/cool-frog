"""Frozen run configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from coolfrog.lib.datatypes import Grid

Range = tuple[float | None, float | None]  # inclusive (min, max), None = open side
FinetuneState = Literal["frozen", "FPFT"] | int  # frozen, full-parameter fine-tuning, or LoRA rank


@dataclass(frozen=True)
class DataConfig:
    """Design selection, data source and batching.

    Attributes
    ----------
    grid : Grid
        Voxel grid every design is sampled onto.
    source : Path | None
        Path -> Local CoolData root with `volume/volume_design_<design_id:07d>_p.cgns`
        None -> Hugging Face.
    n_samples : int | None
        Designs drawn at random from the range matches, None = all matches.
    ranges : dict[str, Range] | None
        dict[str, Range] -> Defines `Range=(min, max)` filtering on metadata column str
        None -> No restrictions.
    cache : bool
        True -> voxelize uncached designs into `CACHE_DIR` on construction, then load from disk.
        False -> voxelize on every access, nothing written.
    seed : int
        Seeds the `n_samples` draw and `CoolDataset.split`.
    batch_size : int
        Designs per batch.
    num_workers : int
        Loader worker processes.

    Examples
    --------
    >>> config = DataConfig(
    ...     grid=Grid(resolution=(400, 80, 16)),
    ...     n_samples=50,
    ...     ranges={"V": (2.0, 5.0), "T1": (None, 60.0)},
    ... )
    >>> train, val = CoolDataset(config).split([0.9, 0.1])
    >>> train_loader = create_dataloader(train, config, shuffle=True)
    """

    grid: Grid
    source: Path | None = None
    n_samples: int | None = None
    ranges: dict[str, Range] = field(default_factory=dict)
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
