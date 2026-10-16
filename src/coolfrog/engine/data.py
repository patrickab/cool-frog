"""CoolData dataset and loader, voxels cached in `CACHE_DIR/voxels_<X>x<Y>x<Z>/<design_id>.pt`."""

from collections.abc import Iterator, Mapping, Sequence
import copy
from dataclasses import dataclass
import functools
import logging
import math
from pathlib import Path
import tempfile
import zipfile

from huggingface_hub import HfFileSystem
import numpy as np
import polars as pl
import pyvista as pv
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from coolfrog.lib.config import DataConfig, Range
from coolfrog.lib.datatypes import CHANNELS, PARAM_COLUMNS, SURFACE_CHANNELS, Channel, FlowField, Grid, SurfaceChannel

CACHE_DIR = Path.cwd() / ".cache"

HF_REVISION = "49d714f48160e3ecc788aa3b5a89c5c0c68a40e2"
HF_DATASET = f"datasets/bgce/cooldata-v2@{HF_REVISION}"
# Official `metadata.parquet` row per simulation file, as the files are named by design id only
DESIGN_ROWS_PATH = Path(__file__).with_name("design_rows.csv")

# CGNS volume array per channel
MESH_ARRAYS: dict[Channel, str] = {
    Channel.VX: "Velocity_0",
    Channel.VY: "Velocity_1",
    Channel.VZ: "Velocity_2",
    Channel.PRESSURE: "Pressure",
    Channel.TEMPERATURE: "Temperature",
    Channel.TURBULENT_KINETIC_ENERGY: "TurbulentKineticEnergy",
    Channel.TURBULENT_DISSIPATION_RATE: "TurbulentDissipationRate",
}
# CGNS surface face array per surface channel
SURFACE_ARRAYS: dict[SurfaceChannel, str] = {
    SurfaceChannel.HEAT_TRANSFER_COEFFICIENT: "HeatTransferCoefficient",
    SurfaceChannel.WALL_SHEAR_STRESS_X: "WallShearStress_0",
    SurfaceChannel.WALL_SHEAR_STRESS_Y: "WallShearStress_1",
    SurfaceChannel.WALL_SHEAR_STRESS_Z: "WallShearStress_2",
    SurfaceChannel.PRESSURE: "Pressure",
    SurfaceChannel.TEMPERATURE: "Temperature",
}

logger = logging.getLogger(__name__)


class UnknownDesignError(LookupError):
    """Design id missing from the CoolData metadata."""


def _official_metadata() -> pl.DataFrame:
    """Official HF `metadata.parquet` at `HF_REVISION`, downloaded once into `CACHE_DIR`."""
    metadata_path = CACHE_DIR / "official_metadata.parquet"
    if not metadata_path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = metadata_path.with_suffix(".tmp")
        HfFileSystem().get_file(f"{HF_DATASET}/metadata.parquet", temp_path)
        temp_path.replace(metadata_path)
    return pl.read_parquet(metadata_path).drop("__index_level_0__")


@functools.cache
def load_metadata() -> pl.DataFrame:
    """One row per simulated design: `design_id`, `run`, `batch` and its official parameters.

    - Parameter sets and design ids repeat across files, the earlier run is kept
    """
    official = _official_metadata().with_row_index("row").with_columns(pl.col("row").cast(pl.Int64))
    return (
        pl.read_csv(DESIGN_ROWS_PATH)
        .join(official, on="row")
        .sort("run", "design_id")
        .unique(PARAM_COLUMNS, keep="first", maintain_order=True)
        .unique("design_id", keep="first", maintain_order=True)
        .select("design_id", "run", "batch", *PARAM_COLUMNS)
        .sort("design_id")
    )


def select_designs(
    metadata: pl.DataFrame, *, ranges: Mapping[str, Range], n_samples: int | None = None, seed: int = 0
) -> pl.DataFrame:
    """Rows inside all `ranges`, sorted by `design_id`, then a seeded sample of `n_samples`, None = all."""
    selected = metadata.filter(
        pl.lit(True),
        *(
            pl.col(column).is_between(-math.inf if low is None else low, math.inf if high is None else high)
            for column, (low, high) in ranges.items()
        ),
    ).sort("design_id")
    if n_samples is not None:
        selected = selected.sample(n=n_samples, seed=seed).sort("design_id")
    return selected


def cache_path(design_id: int, grid: Grid) -> Path:
    """Voxel cache file of one design on `grid`, existing once voxelized."""
    voxels_x, voxels_y, voxels_z = grid.resolution
    return CACHE_DIR / f"voxels_{voxels_x}x{voxels_y}x{voxels_z}" / f"{design_id}.pt"


def load_flow_field(design_id: int, grid: Grid, source: Path | None = None) -> FlowField:
    """FlowField of one design, voxelized into the cache first if needed.

    - Reads the volume from `source` when it holds it, else from Hugging Face
    - Not thread-safe while voxelizing, VTK readers share state
    """
    if source is not None and not (source / "volume" / _volume_filename(design_id)).is_file():
        source = None
    dataset = CoolDataset(DataConfig(grid=grid, source=source, ranges={"design_id": (design_id, design_id)}))
    if len(dataset) == 0:
        raise UnknownDesignError(f"Unknown design: {design_id}")
    return dataset[0]


def _volume_filename(design_id: int) -> str:
    return f"volume_design_{design_id:07d}_p.cgns"


@dataclass(frozen=True)
class SurfaceMesh:
    """Boundary faces of one design: `points` (N, 3) in m, legacy VTK `polygons` [n, i0, …], `values` (channels, faces)."""

    points: np.ndarray
    polygons: np.ndarray
    values: np.ndarray


def surface_cache_path(design_id: int) -> Path:
    """Surface cache file of one design, existing once read."""
    return CACHE_DIR / "surfaces" / f"{design_id}.npz"


def load_surface(design_id: int, source: Path | None = None) -> SurfaceMesh:
    """Boundary surface of one design in `SURFACE_CHANNELS` order, cached in `surface_cache_path` on first read.

    - Reads the surface from `source` when it holds it, else from the design's HF batch zip
    - Not thread-safe on cache misses, VTK readers share state
    """
    design_cache_path = surface_cache_path(design_id)
    if not design_cache_path.exists():
        rows = load_metadata().filter(pl.col("design_id") == design_id)
        if rows.is_empty():
            raise UnknownDesignError(f"Unknown design: {design_id}")
        surface_filename = f"surface_design_{design_id:07d}_p.cgns"
        with tempfile.TemporaryDirectory() as temp_dir:
            if source is not None and (source / "surface" / surface_filename).is_file():
                surface = _read_surface(source / "surface" / surface_filename)
            else:
                batch_path = f"{HF_DATASET}/runs/{rows.item(0, 'run')}/{rows.item(0, 'batch')}.zip"
                with HfFileSystem().open(batch_path, "rb", block_size=2**18) as remote_zip, zipfile.ZipFile(remote_zip) as archive:
                    surface = _read_surface(Path(archive.extract(surface_filename, temp_dir)))
        design_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = design_cache_path.with_suffix(".tmp")
        with temp_path.open("wb") as temp_file:
            np.savez(temp_file, points=surface.points, polygons=surface.polygons, values=surface.values)
        temp_path.replace(design_cache_path)
    with np.load(design_cache_path) as cached:
        return SurfaceMesh(points=cached["points"], polygons=cached["polygons"], values=cached["values"])


def _read_surface(surface_path: Path) -> SurfaceMesh:
    """Merge every boundary patch of a surface CGNS into one mesh, NaN for arrays a patch lacks.

    - Adds NaN caps where walls cut bodies open, see `_build_caps`
    """
    surface = pv.read(surface_path)[0]
    points, polygons, values = [], [], []
    point_offset = 0
    for name, patch in zip(surface.keys(), surface, strict=True):
        face_sizes = np.diff(patch.cell_offsets)
        polygons.append(np.insert(patch.cell_connectivity + point_offset, patch.cell_offsets[:-1], face_sizes))
        points.append(patch.points)
        point_offset += patch.n_points
        missing = np.full(patch.n_cells, np.nan)
        values.append([np.asarray(patch.cell_data.get(SURFACE_ARRAYS[channel], missing)) for channel in SURFACE_CHANNELS])
        if name.startswith("Pipe Body"):
            for cap in _build_caps(np.asarray(patch.points)):
                corner_indices = point_offset + np.arange(4)
                # Both windings, so the cap survives the viewer's front-face culling from either side
                polygons.append(np.concatenate([[4], corner_indices, [4], corner_indices[::-1]]))
                points.append(cap)
                values.append(np.full((len(SURFACE_CHANNELS), 2), np.nan))
                point_offset += 4
    return SurfaceMesh(
        points=np.concatenate(points).astype(np.float32),
        polygons=np.concatenate(polygons).astype(np.uint32),
        values=np.concatenate(values, axis=1).astype(np.float32),
    )


# Vertical domain walls that can cut a body open, (normal axis, coordinate in m): inlet x = 0, sides y = 0 and y = 0.1
CUT_WALLS = ((0, 0.0), (1, 0.0), (1, 0.1))


def _build_caps(body_points: np.ndarray) -> list[np.ndarray]:
    """(4, 3) corners of one rectangle per wall cutting the body, closing the hole the cut leaves.

    - UI rendering only, the caps touch no airflow and are not part of the simulation, so `CoolDataset` never sees them
    - Walls and bodies are vertical, so a cut spans the body's in-wall points horizontally and vertically
    """
    caps = []
    for axis, coordinate in CUT_WALLS:
        points_in_wall = body_points[np.isclose(body_points[:, axis], coordinate, rtol=0, atol=1e-6)]
        if len(points_in_wall) > 0:
            (x_low, y_low, z_low), (x_high, y_high, z_high) = points_in_wall.min(axis=0), points_in_wall.max(axis=0)
            caps.append(np.array([(x_low, y_low, z_low), (x_high, y_high, z_low), (x_high, y_high, z_high), (x_low, y_low, z_high)]))
    return caps


class CoolDataset(Dataset[FlowField]):
    """Owns metadata, filestorage and provides access to single FlowFields

    - Can load from cache.
    - Represents selection of CoolData designs
    - If `config.cache==True` voxelizes uncached designs into voxel cache on construction, logs the count first
    - Selects designs inside `config.ranges`, sorted by `design_id`, then takes seeded sample of size `config.n_samples`
    """

    def __init__(self, config: DataConfig) -> None:
        if not config.cache and config.source is None:
            raise ValueError("cache=False voxelizes on every access and needs a local `source`")
        self.config = config
        self._metadata = select_designs(load_metadata(), ranges=config.ranges, n_samples=config.n_samples, seed=config.seed)
        self._params = self._metadata.select(PARAM_COLUMNS).to_torch(dtype=pl.Float32)
        if config.cache:
            uncached_design_ids = [design_id for design_id in self.design_ids if not cache_path(design_id, config.grid).exists()]
            logger.info("%d of %d selected designs uncached", len(uncached_design_ids), len(self))
            if uncached_design_ids:
                self._prepare_voxel_cache(uncached_design_ids)

    @property
    def design_ids(self) -> list[int]:
        """Design ids in the dataset in index order."""
        return self._metadata["design_id"].to_list()

    def split(self, fractions: Sequence[float]) -> list["CoolDataset"]:
        """Disjoint random subsets by design, seeded by `config.seed`, sharing the cache: `train, val = dataset.split([0.9, 0.1])`."""
        generator = torch.Generator().manual_seed(self.config.seed)
        return [self._subset(sorted(subset.indices)) for subset in random_split(range(len(self)), fractions, generator)]

    def __len__(self) -> int:
        return self._metadata.height

    def __getitem__(self, index: int) -> FlowField:
        design_id = self._metadata.item(index, "design_id")
        if self.config.cache:
            voxels = torch.load(cache_path(design_id, self.config.grid), weights_only=True)
            fields, mask = voxels["fields"], voxels["mask"]
        else:
            fields, mask = self._voxelize(self._local_volume_path(design_id), self.config.grid)
        return FlowField(fields=fields, mask=mask, design_id=torch.tensor(design_id), params=self._params[index], batch_size=[])

    def _prepare_voxel_cache(self, design_ids: Sequence[int]) -> None:
        """Download and voxelize `design_ids` into the voxel cache, called by `__init__` for uncached designs.

        Notes
        -----
        - progress bar per design, each `(run, batch)` zip on HF opened once and only the needed volumes streamed
        - writes via temp file and rename, so an interrupted run leaves no partial files and resumes on the next construction
        """
        rows = self._metadata.filter(pl.col("design_id").is_in(design_ids))
        with tempfile.TemporaryDirectory() as temp_dir, tqdm(total=rows.height, desc="Voxelizing designs", unit="design") as progress:
            for (run, batch), group in rows.group_by("run", "batch", maintain_order=True):
                for design_id, volume_path in self._volume_paths(str(run), str(batch), group["design_id"].to_list(), Path(temp_dir)):
                    fields, mask = self._voxelize(volume_path, self.config.grid)
                    design_cache_path = cache_path(design_id, self.config.grid)
                    design_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temp_path = design_cache_path.with_suffix(".tmp")
                    torch.save({"fields": fields, "mask": mask}, temp_path)
                    temp_path.replace(design_cache_path)
                    progress.update()

    def _volume_paths(self, run: str, batch: str, design_ids: Sequence[int], temp_dir: Path) -> Iterator[tuple[int, Path]]:
        """Yield the CGNS volume per design, local or extracted from its HF zip and deleted after use."""
        if self.config.source is not None:
            for design_id in design_ids:
                yield design_id, self._local_volume_path(design_id)
            return
        with HfFileSystem().open(f"{HF_DATASET}/runs/{run}/{batch}.zip", "rb") as remote_zip, zipfile.ZipFile(remote_zip) as archive:
            for design_id in design_ids:
                volume_path = Path(archive.extract(_volume_filename(design_id), temp_dir))
                yield design_id, volume_path
                volume_path.unlink()

    def _local_volume_path(self, design_id: int) -> Path:
        if self.config.source is None:
            raise ValueError("no local `source` configured")
        return self.config.source / "volume" / _volume_filename(design_id)

    def _subset(self, indices: Sequence[int]) -> "CoolDataset":
        subset = copy.copy(self)
        subset._metadata = self._metadata[list(indices)]
        subset._params = self._params[list(indices)]
        return subset

    @staticmethod
    def _voxelize(volume_path: Path, grid: Grid) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a CGNS volume onto `grid` as (C, X, Y, Z) fields and (X, Y, Z) fluid mask, like cooldata `from_pyvista`."""
        axes = [np.linspace(0.0, length, voxels) for length, voxels in zip(grid.extent, grid.resolution, strict=True)]
        sampled = pv.StructuredGrid(*np.meshgrid(*axes, indexing="ij")).sample(pv.read(volume_path)[0][0][0])
        fields = np.stack([sampled[MESH_ARRAYS[channel]].reshape(grid.resolution, order="F") for channel in CHANNELS])
        mask = sampled["vtkValidPointMask"].reshape(grid.resolution, order="F")
        return torch.tensor(fields, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


def create_dataloader(dataset: CoolDataset, config: DataConfig, shuffle: bool) -> DataLoader[FlowField]:
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
