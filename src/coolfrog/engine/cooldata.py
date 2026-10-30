"""CoolData design catalogue, voxel dataset and UI surface meshes, all cached in `CACHE_DIR`."""

from collections.abc import Iterable, Mapping, Sequence
import copy
from dataclasses import dataclass
import fcntl
import functools
import itertools
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import tempfile
from typing import Any, ClassVar
import zipfile

import h5py
import hdf5plugin
from huggingface_hub import HfFileSystem
import numpy as np
import polars as pl
import pyvista as pv
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from coolfrog.lib.config import HOST_CONFIG, DataConfig, Range
from coolfrog.lib.datatypes import (
    CHANNELS,
    PARAM_COLUMNS,
    SURFACE_CHANNELS,
    Channel,
    FlowField,
    Grid,
    SurfaceChannel,
)
from coolfrog.lib.log import get_logger, progress_bar

CACHE_DIR = HOST_CONFIG.cache_dir
SHARD_SIZE = 64
PROCESSES = max(1, len(os.sched_getaffinity(0)) - 2)  # shards built in parallel by `build_voxel_cache`, 2 CPUs left for the system

HF_REVISION = "49d714f48160e3ecc788aa3b5a89c5c0c68a40e2"
HF_DATASET = f"datasets/bgce/cooldata-v2@{HF_REVISION}"

logger = get_logger(__name__)
DesignRow = tuple[int, str, str]  # (design_id, run, batch)
VoxelizedDesign = tuple[int, np.ndarray, np.ndarray]  # (design_id, (C, X, Y, Z) fields, (X, Y, Z) fluid mask)


class UnknownDesignError(LookupError):
    """Design id missing from the CoolData metadata."""


@dataclass(frozen=True)
class CoolMetadata:
    """Design catalogue, one row per simulated design: `design_id`, `run`, `batch` and its official parameters.

    - Never touches simulation files, `CoolData` materializes a selection
    """

    designs: pl.DataFrame

    # Official `metadata.parquet` row per simulation file, as the files are named by design id only
    DESIGN_ROWS_PATH: ClassVar[Path] = Path(__file__).with_name("design_rows.csv")

    @classmethod
    @functools.cache
    def load(cls, n_splits: int = 1, idx_split: int = 0) -> "CoolMetadata":
        """Deduplicated catalogue, every `n_splits`-th design by `design_id` from `idx_split` on, built once per process and split.

        - Parameter sets and design ids repeat across files, the earlier run is kept
        - The `n_splits` splits are disjoint, cover all designs and differ in size by at most one, the defaults load all
        """
        if not 0 <= idx_split < n_splits:
            raise ValueError(f"idx_split {idx_split} outside [0, {n_splits})")
        official = cls._official_metadata().with_row_index("row").with_columns(pl.col("row").cast(pl.Int64))
        designs = (
            pl.read_csv(cls.DESIGN_ROWS_PATH)
            .join(official, on="row")
            .sort("run", "design_id")
            .unique(PARAM_COLUMNS, keep="first", maintain_order=True)
            .unique("design_id", keep="first", maintain_order=True)
            .select("design_id", "run", "batch", *PARAM_COLUMNS)
            .sort("design_id")
            .gather_every(n_splits, offset=idx_split)
        )
        return cls(designs)

    def select(self, *, ranges: Mapping[str, Range], n_samples: int | None = None, seed: int = 0) -> "CoolMetadata":
        """Designs inside all `ranges`, sorted by `design_id`, then a seeded sample of `n_samples`, None = all."""
        selected = self.designs.filter(
            pl.lit(True),
            *(
                pl.col(column).is_between(-math.inf if low is None else low, math.inf if high is None else high)
                for column, (low, high) in ranges.items()
            ),
        ).sort("design_id")
        if n_samples is not None:
            selected = selected.sample(n=n_samples, seed=seed).sort("design_id")
        return CoolMetadata(selected)

    def design(self, design_id: int) -> dict[str, Any]:
        """Row of one design by column name."""
        rows = self.designs.filter(pl.col("design_id") == design_id)
        if rows.is_empty():
            raise UnknownDesignError(f"Unknown design: {design_id}")
        return rows.row(0, named=True)

    @staticmethod
    def _official_metadata() -> pl.DataFrame:
        """Official HF `metadata.parquet` at `HF_REVISION`, downloaded once into `CACHE_DIR`."""
        metadata_path = CACHE_DIR / "official_metadata.parquet"
        if not metadata_path.exists():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            temp_path = metadata_path.with_suffix(".tmp")
            HfFileSystem().get_file(f"{HF_DATASET}/metadata.parquet", temp_path)
            temp_path.replace(metadata_path)
        return pl.read_parquet(metadata_path).drop("__index_level_0__")


class CoolDataShard:
    """One HDF5 file of up to `SHARD_SIZE` voxelized designs, written once by `write`, read row by row.

    - Holds `design_id (n,)`, `fields (n, C, X, Y, Z)` and `mask (n, X, Y, Z)`, one blosc2-zstd chunk per design
    - Keeps only `path` and `design_ids` in memory
    """

    # CGNS (CFD General Notation System) volume array per channel
    CGNS_ARRAYS: ClassVar[dict[Channel, str]] = {
        Channel.VX: "Velocity_0",
        Channel.VY: "Velocity_1",
        Channel.VZ: "Velocity_2",
        Channel.PRESSURE: "Pressure",
        Channel.TEMPERATURE: "Temperature",
        Channel.TURBULENT_KINETIC_ENERGY: "TurbulentKineticEnergy",
        Channel.TURBULENT_DISSIPATION_RATE: "TurbulentDissipationRate",
    }
    # Lossless, about 4x smaller than raw float32 and decoded in ~12 ms per design at 400x80x16, h5py detects it when reading
    COMPRESSION: ClassVar[hdf5plugin.Blosc2] = hdf5plugin.Blosc2(cname="zstd", clevel=1, filters=hdf5plugin.Blosc2.SHUFFLE)

    def __init__(self, path: Path) -> None:
        self.path = path
        with h5py.File(path, "r") as file:
            self.design_ids: list[int] = file["design_id"][:].tolist()

    @staticmethod
    def cache_dir(grid: Grid) -> Path:
        """Directory of the shards on `grid`."""
        voxels_x, voxels_y, voxels_z = grid.resolution
        return CACHE_DIR / f"voxels_{voxels_x}x{voxels_y}x{voxels_z}"

    @staticmethod
    def local_volume_path(source: Path, design_id: int) -> Path:
        """CGNS volume of one design inside a local CoolData root."""
        return source / "volume" / f"volume_design_{design_id:07d}_p.cgns"

    @classmethod
    def voxelize_design(cls, row: DesignRow, grid: Grid, source: Path | None = None) -> VoxelizedDesign:
        """Download and voxelize one design, polars-free as polars deadlocks in the forked workers of `build_voxel_cache`.

        - Reads the volume from `source` when given, else extracts it from the design's `(run, batch)` HF zip
        """
        design_id, run, batch = row
        if source is not None:
            fields, mask = cls.voxelize(cls.local_volume_path(source, design_id), grid)
        else:
            with (
                tempfile.TemporaryDirectory() as temp_dir,
                HfFileSystem().open(f"{HF_DATASET}/runs/{run}/{batch}.zip", "rb") as remote_zip,
                zipfile.ZipFile(remote_zip) as archive,
            ):
                fields, mask = cls.voxelize(Path(archive.extract(cls.local_volume_path(Path(), design_id).name, temp_dir)), grid)
        return design_id, fields.numpy(), mask.numpy()

    @classmethod
    def write(cls, voxelized: Iterable[VoxelizedDesign], count: int, grid: Grid) -> "CoolDataShard":
        """Write the next `count` designs of `voxelized` into `<first design_id>.h5` in `cache_dir(grid)`.

        - Writes via temp file and rename, so an interrupted build leaves no partial shard
        """
        cache_dir = cls.cache_dir(grid)
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_name = tempfile.mkstemp(suffix=".tmp", dir=cache_dir)
        os.close(file_descriptor)
        temp_path = Path(temp_name)
        with h5py.File(temp_path, "w") as file:
            design_id_dataset = file.create_dataset("design_id", (count,), dtype="int64")
            design_shape, mask_shape = (len(CHANNELS), *grid.resolution), grid.resolution
            fields_dataset = file.create_dataset(
                "fields", (count, *design_shape), "float32", chunks=(1, *design_shape), **cls.COMPRESSION
            )
            mask_dataset = file.create_dataset("mask", (count, *mask_shape), "bool", chunks=(1, *mask_shape), **cls.COMPRESSION)
            written = 0
            for row, (design_id, fields, mask) in enumerate(itertools.islice(voxelized, count)):
                design_id_dataset[row], fields_dataset[row], mask_dataset[row] = design_id, fields, mask
                written += 1
            if written != count:
                raise ValueError(f"expected {count} designs, got {written}")
            path = cache_dir / f"{design_id_dataset[0]}.h5"
        temp_path.replace(path)
        return cls(path)

    def read(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(C, X, Y, Z) fields and (X, Y, Z) fluid mask of one row."""
        with h5py.File(self.path, "r") as file:  # opened per access, handles break across DataLoader workers
            return torch.from_numpy(file["fields"][row]), torch.from_numpy(file["mask"][row])

    @classmethod
    def voxelize(cls, volume_path: Path, grid: Grid) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a CGNS volume onto `grid` as (C, X, Y, Z) fields and (X, Y, Z) fluid mask, like cooldata `from_pyvista`."""
        axes = [np.linspace(0.0, length, voxels) for length, voxels in zip(grid.extent, grid.resolution, strict=True)]
        with pv.vtk_verbosity("error"):  # silences "Skipping BC_t node" warnings, only cell and face data are read
            volume = pv.read(volume_path)[0][0][0]
        sampled = pv.StructuredGrid(*np.meshgrid(*axes, indexing="ij")).sample(volume)
        fields = np.stack([sampled[cls.CGNS_ARRAYS[channel]].reshape(grid.resolution, order="F") for channel in CHANNELS])
        mask = sampled["vtkValidPointMask"].reshape(grid.resolution, order="F")
        return torch.tensor(fields, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


def cached_shards(grid: Grid) -> dict[int, tuple[CoolDataShard, int]]:
    """Shard and row of every voxelized design on `grid`, read from the shards' `design_id` datasets."""
    locations: dict[int, tuple[CoolDataShard, int]] = {}
    for path in sorted(CoolDataShard.cache_dir(grid).glob("*.h5")):
        shard = CoolDataShard(path)
        locations.update({design_id: (shard, row) for row, design_id in enumerate(shard.design_ids)})
    return locations


class CoolData(Dataset[FlowField]):
    """Voxel FlowFields of the `metadata` designs on `config.grid`.

    - `config.cache=True` voxelizes uncached designs into HDF5 shards on construction, see `build_voxel_cache`, then reads from disk
    - `config.cache=False` voxelizes from the local `config.source` on every access
    """

    def __init__(self, metadata: CoolMetadata, config: DataConfig) -> None:
        if not config.cache and config.source is None:
            raise ValueError("cache=False voxelizes on every access and needs a local `source`")
        self.config = config
        self._designs = metadata.designs
        self._params = self._designs.select(PARAM_COLUMNS).to_torch(dtype=pl.Float32)
        self._locations: dict[int, tuple[CoolDataShard, int]] = {}
        if config.cache:
            self.build_voxel_cache(metadata, config.grid, config.source)
            self._locations = cached_shards(config.grid)

    @staticmethod
    def is_cached(design_id: int, grid: Grid) -> bool:
        """Whether `design_id` is already voxelized on `grid`."""
        return design_id in cached_shards(grid)

    @classmethod
    def load_flow_field(cls, design_id: int, grid: Grid, source: Path | None = None) -> FlowField:
        """FlowField of one design, voxelized into its own shard first if uncached.

        - Reads the volume from `source` when it holds it, else from Hugging Face
        - Not thread-safe while voxelizing, VTK readers share state
        """
        metadata = CoolMetadata.load().select(ranges={"design_id": (design_id, design_id)})
        if metadata.designs.is_empty():
            raise UnknownDesignError(f"Unknown design: {design_id}")
        if not cls.is_cached(design_id, grid):
            if source is not None and not CoolDataShard.local_volume_path(source, design_id).is_file():
                source = None
            row = metadata.designs.select("design_id", "run", "batch").row(0)
            CoolDataShard.write([CoolDataShard.voxelize_design(row, grid, source)], 1, grid)
        return cls(metadata, DataConfig(grid=grid))[0]

    @staticmethod
    def build_voxel_cache(metadata: CoolMetadata, grid: Grid, source: Path | None = None) -> None:
        """Voxelize every uncached `metadata` design, `PROCESSES` designs at a time, into new shards of `SHARD_SIZE`.

        - Holds `build.lock` of the grid, so concurrent builds (e.g. one per GPU process) wait, then find their designs cached
        - Deletes leftover `.tmp` shards of interrupted builds
        - Raises before downloading when the uncached designs do not fit on the cache disk
        - A failing shard or Ctrl+C stops the build at once, finished shards stay and the next call resumes
        """
        cache_dir = CoolDataShard.cache_dir(grid)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with (cache_dir / "build.lock").open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            for temp_path in cache_dir.glob("*.tmp"):
                logger.info("Deleting leftover %s", temp_path.name)
                temp_path.unlink()
            uncached = metadata.designs.filter(~pl.col("design_id").is_in(list(cached_shards(grid)))).sort("run", "batch", "design_id")
            required_bytes = uncached.height * (len(CHANNELS) * 4 + 1) * math.prod(grid.resolution)  # uncompressed, upper bound
            available_bytes = shutil.disk_usage(cache_dir).free
            logger.info(
                "%d of %d designs uncached, %.1f GB needed, %.1f GB free",
                uncached.height,
                metadata.designs.height,
                required_bytes / 1e9,
                available_bytes / 1e9,
            )
            if required_bytes > available_bytes:
                raise OSError(f"{required_bytes / 1e9:.1f} GB needed, only {available_bytes / 1e9:.1f} GB free in {cache_dir}")
            if uncached.is_empty():
                return
            # fork, as spawn would rerun the calling script's top level in every process
            # workers voxelize single designs and ignore Ctrl+C, the pool's exit kills them at once
            # this process writes them into shards, an interrupt loses only the shard being filled
            voxelize_design = functools.partial(CoolDataShard.voxelize_design, grid=grid, source=source)
            rows = uncached.select("design_id", "run", "batch").rows()
            fork = multiprocessing.get_context("fork")
            with (
                fork.Pool(PROCESSES, initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN)) as pool,
                progress_bar() as progress,
            ):
                voxelized = progress.track(pool.imap_unordered(voxelize_design, rows), len(rows), description="Voxelizing designs")
                for shard_start in range(0, len(rows), SHARD_SIZE):
                    CoolDataShard.write(voxelized, min(SHARD_SIZE, len(rows) - shard_start), grid)

    @property
    def design_ids(self) -> list[int]:
        """Design ids in the dataset in index order."""
        return self._designs["design_id"].to_list()

    def split(self, fractions: Sequence[float]) -> list["CoolData"]:
        """Disjoint random subsets by design, seeded by `config.seed`, sharing the cache: `train, val = dataset.split([0.9, 0.1])`."""
        generator = torch.Generator().manual_seed(self.config.seed)
        return [self._subset(sorted(subset.indices)) for subset in random_split(range(len(self)), fractions, generator)]

    def __len__(self) -> int:
        return self._designs.height

    def __getitem__(self, index: int) -> FlowField:
        design_id = self._designs.item(index, "design_id")
        if self.config.cache:
            shard, row = self._locations[design_id]
            fields, mask = shard.read(row)
        else:
            assert self.config.source is not None  # checked in __init__
            fields, mask = CoolDataShard.voxelize(CoolDataShard.local_volume_path(self.config.source, design_id), self.config.grid)
        return FlowField(fields=fields, mask=mask, design_id=torch.tensor(design_id), params=self._params[index], batch_size=[])

    def _subset(self, indices: Sequence[int]) -> "CoolData":
        subset = copy.copy(self)
        subset._designs = self._designs[list(indices)]
        subset._params = self._params[list(indices)]
        return subset


def create_dataloader(dataset: CoolData, config: DataConfig, shuffle: bool) -> DataLoader[FlowField]:
    """DataLoader of stacked FlowField batches, batch size and workers from `config`."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=torch.stack,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


@dataclass(frozen=True)
class SurfaceMesh:
    """Boundary faces of one design for the UI viewer, see `load`.

    Attributes
    ----------
    points : np.ndarray
        (N, 3) in m.
    polygons : np.ndarray
        Legacy VTK faces [n, i0, …, n, i0, …].
    values : np.ndarray
        (channels, faces) in `SURFACE_CHANNELS` order, NaN where a patch lacks the array.
    """

    points: np.ndarray
    polygons: np.ndarray
    values: np.ndarray

    # CGNS (CFD General Notation System) face array per surface channel
    CGNS_ARRAYS: ClassVar[dict[SurfaceChannel, str]] = {
        SurfaceChannel.HEAT_TRANSFER_COEFFICIENT: "HeatTransferCoefficient",
        SurfaceChannel.WALL_SHEAR_STRESS_X: "WallShearStress_0",
        SurfaceChannel.WALL_SHEAR_STRESS_Y: "WallShearStress_1",
        SurfaceChannel.WALL_SHEAR_STRESS_Z: "WallShearStress_2",
        SurfaceChannel.PRESSURE: "Pressure",
        SurfaceChannel.TEMPERATURE: "Temperature",
    }
    # Vertical domain walls that can cut a body open, (normal axis, coordinate in m): inlet x = 0, sides y = 0 and y = 0.1
    CUT_WALLS: ClassVar[tuple[tuple[int, float], ...]] = ((0, 0.0), (1, 0.0), (1, 0.1))

    @staticmethod
    def cache_path(design_id: int) -> Path:
        """Cache file of one design, existing once loaded."""
        return CACHE_DIR / "surfaces" / f"{design_id}.npz"

    @classmethod
    def load(cls, design_id: int, source: Path | None = None) -> "SurfaceMesh":
        """Surface of one design, from `cache_path` or read from its CGNS and cached.

        - Reads the CGNS from `source` when it holds it, else from the design's HF batch zip
        - Not thread-safe on cache misses, VTK readers share state
        """
        cache_path = cls.cache_path(design_id)
        if cache_path.exists():
            with np.load(cache_path) as cached:
                return cls(**cached)
        design = CoolMetadata.load().design(design_id)
        filename = f"surface_design_{design_id:07d}_p.cgns"
        with tempfile.TemporaryDirectory() as temp_dir:
            if source is not None and (source / "surface" / filename).is_file():
                mesh = cls._read_cgns(source / "surface" / filename)
            else:
                batch_path = f"{HF_DATASET}/runs/{design['run']}/{design['batch']}.zip"
                with HfFileSystem().open(batch_path, "rb", block_size=2**18) as remote_zip, zipfile.ZipFile(remote_zip) as archive:
                    mesh = cls._read_cgns(Path(archive.extract(filename, temp_dir)))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")
        with temp_path.open("wb") as temp_file:
            np.savez(temp_file, **vars(mesh))
        temp_path.replace(cache_path)
        return mesh

    @classmethod
    def _read_cgns(cls, cgns_path: Path) -> "SurfaceMesh":
        """Merge every boundary patch into one mesh, plus caps where walls cut bodies open, see `_build_caps`."""
        with pv.vtk_verbosity("error"):  # silences "Skipping BC_t node" warnings, only cell and face data are read
            patches = pv.read(cgns_path)[0]
        points, polygons, values = [], [], []
        point_offset = 0
        for name, patch in zip(patches.keys(), patches, strict=True):
            face_sizes = np.diff(patch.cell_offsets)
            polygons.append(np.insert(patch.cell_connectivity + point_offset, patch.cell_offsets[:-1], face_sizes))
            points.append(patch.points)
            point_offset += patch.n_points
            missing = np.full(patch.n_cells, np.nan)
            values.append([np.asarray(patch.cell_data.get(cls.CGNS_ARRAYS[channel], missing)) for channel in SURFACE_CHANNELS])
            if name.startswith("Pipe Body"):
                for cap in cls._build_caps(np.asarray(patch.points)):
                    corner_indices = point_offset + np.arange(4)
                    # Both windings, so the cap survives the viewer's front-face culling from either side
                    polygons.append(np.concatenate([[4], corner_indices, [4], corner_indices[::-1]]))
                    points.append(cap)
                    values.append(np.full((len(SURFACE_CHANNELS), 2), np.nan))
                    point_offset += 4
        return cls(
            points=np.concatenate(points).astype(np.float32),
            polygons=np.concatenate(polygons).astype(np.uint32),
            values=np.concatenate(values, axis=1).astype(np.float32),
        )

    @classmethod
    def _build_caps(cls, body_points: np.ndarray) -> list[np.ndarray]:
        """(4, 3) corners of one rectangle per wall cutting the body, closing the hole the cut leaves.

        - UI rendering only, the caps touch no airflow and are not part of the simulation, so `CoolDataset` never sees them
        - Walls and bodies are vertical, so a cut spans the body's in-wall points horizontally and vertically
        """
        caps = []
        for axis, coordinate in cls.CUT_WALLS:
            points_in_wall = body_points[np.isclose(body_points[:, axis], coordinate, rtol=0, atol=1e-6)]
            if len(points_in_wall) > 0:
                (x_low, y_low, z_low), (x_high, y_high, z_high) = points_in_wall.min(axis=0), points_in_wall.max(axis=0)
                corners = [(x_low, y_low, z_low), (x_high, y_high, z_low), (x_high, y_high, z_high), (x_low, y_low, z_high)]
                caps.append(np.array(corners))
        return caps
