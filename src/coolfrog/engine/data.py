"""CoolData dataset and loader, voxels cached in `CACHE_DIR/voxels_<X>x<Y>x<Z>/<design_id>.pt`."""

from collections.abc import Iterator, Mapping, Sequence
import contextlib
import copy
from dataclasses import dataclass
import functools
import hashlib
import logging
import math
import multiprocessing
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
OFFICIAL_METADATA_MD5 = "aef935ea0ff0580561911bb7d50828f5"
DESIGN_ROWS_PATH = Path(__file__).with_name("design_rows.csv")
"""
Official metadata row of every simulation file, measured by `scan_design_rows`

- Official `metadata.parquet` has 270,876 rows and no design id, files are named by design id only
- Row = design id - 1 in runs 1, 3, 4, 6, per batch -1 or 0 in run_7
- 70,947 files but 60,848 ids (the paper's count): 10,099 ids exist in run_4 and run_7, mostly with different rows
- 19 run_7 files match no row
- Official rows repeat parameter sets, the files cover 56,936 distinct ones
- `load_metadata` keeps one file per parameter set and per id, earlier run first: 56,916 designs
"""
# Candidate offsets row - design_id, preferred first, duplicate rows match several
ROW_OFFSETS = (-1, 0, -2, 1, -3, 2, 3)
BODY_COUNT = 6
TEMPERATURE_TOLERANCE_K = 1e-3

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


def official_metadata() -> pl.DataFrame:
    """Official HF `metadata.parquet` at `HF_REVISION`, downloaded once into `CACHE_DIR` and md5-checked."""
    metadata_path = CACHE_DIR / "official_metadata.parquet"
    if not metadata_path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = metadata_path.with_suffix(".tmp")
        HfFileSystem().get_file(f"{HF_DATASET}/metadata.parquet", temp_path)
        temp_path.replace(metadata_path)
    if hashlib.md5(metadata_path.read_bytes()).hexdigest() != OFFICIAL_METADATA_MD5:
        raise RuntimeError(f"{metadata_path} differs from the official metadata the design rows were measured against")
    return pl.read_parquet(metadata_path).drop("__index_level_0__")


def scan_design_rows(workers: int = 4) -> None:
    """Measure the official metadata row of every simulated design into `DESIGN_ROWS_PATH`, resumable per batch.

    - Surface CGNS stores each body's constant wall temperature, matched against official `T1..T6`
    - Probes first and last design per batch, every design if their offsets disagree, unmatched designs get no row
    """
    done_batches = set()
    if DESIGN_ROWS_PATH.exists():
        done_batches = set(pl.read_csv(DESIGN_ROWS_PATH).select(pl.format("{}/{}", "run", "batch")).to_series())
    else:
        DESIGN_ROWS_PATH.write_text("design_id,run,batch,row\n")
    batch_paths = HfFileSystem().glob(f"{HF_DATASET}/runs/run_*/batch_*.zip")
    batch_paths = [path for path in batch_paths if _batch_key(path) not in done_batches]
    with multiprocessing.get_context("spawn").Pool(workers) as pool, DESIGN_ROWS_PATH.open("a") as rows_file:
        batches = pool.imap_unordered(_scan_batch, batch_paths)
        for batch_rows in tqdm(batches, total=len(batch_paths), desc="Scanning batches", unit="batch"):
            for design_id, run, batch, row in batch_rows:
                rows_file.write(f"{design_id},{run},{batch},{'' if row is None else row}\n")
            rows_file.flush()


def _batch_key(batch_path: str) -> str:
    run, batch_file = batch_path.split("/")[-2:]
    return f"{run}/{batch_file.removesuffix('.zip')}"


@functools.cache
def _official_temperatures() -> np.ndarray:
    """(rows, bodies) official wall temperatures in K, NaN for inactive bodies."""
    official = official_metadata()
    temperatures = official.select(f"T{body}" for body in range(1, BODY_COUNT + 1)).to_numpy() + 273.15
    inactive = official.select(pl.col(f"y{body}") == 1.0 for body in range(1, BODY_COUNT + 1)).to_numpy()
    return np.where(inactive, np.nan, temperatures)


def _scan_batch(batch_path: str) -> list[tuple[int, str, str, int | None]]:
    """Rows of all designs in one batch zip, see `scan_design_rows`."""
    run, batch = _batch_key(batch_path).split("/")
    with HfFileSystem().open(batch_path, "rb", block_size=2**18) as remote_zip, zipfile.ZipFile(remote_zip) as archive:
        surface_names = sorted(name for name in archive.namelist() if name.startswith("surface_"))
        design_ids = [int(name.split("_")[2]) for name in surface_names]
        first_offsets, last_offsets = (_matching_offsets(design_ids[index], archive, surface_names[index]) for index in (0, -1))
        shared_offsets = [offset for offset in first_offsets if offset in last_offsets]
        if shared_offsets:
            return [(design_id, run, batch, design_id + shared_offsets[0]) for design_id in design_ids]
        rows = []
        for design_id, surface_name in zip(design_ids, surface_names, strict=True):
            offsets = _matching_offsets(design_id, archive, surface_name)
            rows.append((design_id, run, batch, design_id + offsets[0] if offsets else None))
        return rows


def _matching_offsets(design_id: int, archive: zipfile.ZipFile, surface_name: str) -> list[int]:
    """Offsets in `ROW_OFFSETS` order whose official row matches the wall temperature of every simulated body.

    - Some active metadata bodies have no surface patch, so only bodies present in the surface are compared
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        surface = pv.read(archive.extract(surface_name, temp_dir))[0]
    surface_temperatures = np.full(BODY_COUNT, np.nan)
    for body in range(1, BODY_COUNT + 1):
        with contextlib.suppress(KeyError):
            surface_temperatures[body - 1] = np.asarray(surface[f"Pipe Body {body}"]["Temperature"]).mean()
    official_temperatures = _official_temperatures()
    simulated_bodies = ~np.isnan(surface_temperatures)
    return [
        offset
        for offset in ROW_OFFSETS
        if 0 <= design_id + offset < len(official_temperatures)
        and np.allclose(
            official_temperatures[design_id + offset, simulated_bodies],
            surface_temperatures[simulated_bodies],
            rtol=0,
            atol=TEMPERATURE_TOLERANCE_K,
        )
    ]


@functools.cache
def load_metadata() -> pl.DataFrame:
    """One row per distinct simulated design: `design_id`, `run`, `batch` and its official parameters, see `DESIGN_ROWS_PATH`."""
    official = official_metadata().with_row_index("row").with_columns(pl.col("row").cast(pl.Int64))
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

    - Bodies cut by the inlet or a side plane have no face there, so a NaN cap closes each cut
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
            for cap in _cut_caps(np.asarray(patch.points)):
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


# Domain planes a body can be cut by, (axis, coordinate in m): inlet x = 0, side planes y = 0 and y = 0.1
CUT_PLANES = ((0, 0.0), (1, 0.0), (1, 0.1))
PLANE_TOLERANCE_M = 1e-6


def _cut_caps(body_points: np.ndarray) -> list[np.ndarray]:
    """(4, 3) rectangle per domain plane the body touches, spanning the body's points in that plane.

    - Cuboid and cylinder cuts are both rectangles, as bodies are vertical extrusions
    """
    caps = []
    for axis, coordinate in CUT_PLANES:
        in_plane = body_points[np.abs(body_points[:, axis] - coordinate) < PLANE_TOLERANCE_M]
        if in_plane.size == 0:
            continue
        first_axis, second_axis = (other for other in range(3) if other != axis)
        low, high = in_plane.min(axis=0), in_plane.max(axis=0)
        if high[first_axis] > low[first_axis] and high[second_axis] > low[second_axis]:
            corners = np.full((4, 3), coordinate)
            corners[:, first_axis] = [low[first_axis], high[first_axis], high[first_axis], low[first_axis]]
            corners[:, second_axis] = [low[second_axis], low[second_axis], high[second_axis], high[second_axis]]
            caps.append(corners)
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

    @property
    def metadata(self) -> pl.DataFrame:
        """Metadata rows of the selected designs, in index order."""
        return self._metadata

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
        cache_path(0, self.config.grid).parent.mkdir(parents=True, exist_ok=True)
        rows = self._metadata.filter(pl.col("design_id").is_in(design_ids))
        with tempfile.TemporaryDirectory() as temp_dir, tqdm(total=rows.height, desc="Voxelizing designs", unit="design") as progress:
            for (run, batch), group in rows.group_by("run", "batch", maintain_order=True):
                for design_id, volume_path in self._volume_paths(str(run), str(batch), group["design_id"].to_list(), Path(temp_dir)):
                    fields, mask = self._voxelize(volume_path, self.config.grid)
                    design_cache_path = cache_path(design_id, self.config.grid)
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
