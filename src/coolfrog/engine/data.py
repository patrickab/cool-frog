"""CoolData dataset and loader, voxels cached in `CACHE_DIR/voxels_<X>x<Y>x<Z>/<design_id>.pt`."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl
from torch.utils.data import DataLoader, Dataset

from coolfrog.lib.config import DataConfig
from coolfrog.lib.datatypes import FlowField

CACHE_DIR = Path.cwd() / ".cache"


class CoolDataset(Dataset[FlowField]):
    """Owns metadata, filestorage and provides access to single FlowFields

    - Can load from cache.
    - Represents selection of CoolData designs
    - If `config.cache==True` voxelizes uncached designs into voxel cache on construction, logs the count first
    - Selects designs inside `config.ranges`, sorted by `design_id`, then takes seeded sample of size `config.n_samples`
    """

    def __init__(self, config: DataConfig) -> None:
        raise NotImplementedError

    @property
    def design_ids(self) -> list[int]:
        """Design ids in the dataset in index order."""
        raise NotImplementedError

    @property
    def metadata(self) -> pl.DataFrame:
        """Metadata rows of the selected designs, in index order."""
        raise NotImplementedError

    def split(self, fractions: Sequence[float]) -> list["CoolDataset"]:
        """Disjoint random subsets by design, seeded by `config.seed`, sharing the cache: `train, val = dataset.split([0.9, 0.1])`."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> FlowField:
        raise NotImplementedError

    def _prepare_voxel_cache(self, design_ids: Sequence[int]) -> None:
        """Download and voxelize `design_ids` into the voxel cache, called by `__init__` for uncached designs.

        Notes
        -----
        - progress bar per design, raw zips downloaded once per `(run, batch)` from `metadata`
        - writes via temp file and rename, so an interrupted run leaves no partial files and resumes on the next construction
        """
        raise NotImplementedError


def create_dataloader(dataset: CoolDataset, config: DataConfig, shuffle: bool) -> DataLoader[FlowField]:
    """DataLoader of stacked FlowField batches, batch size and workers from `config`."""
    raise NotImplementedError
