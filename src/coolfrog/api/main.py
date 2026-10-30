"""CoolData browser API. Run with `uvicorn coolfrog.api.main:app --port 8010`."""

import asyncio
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
import numpy as np
import polars as pl
from pydantic import BaseModel, Field, FiniteFloat, field_validator
import torch

from coolfrog.engine.cooldata import CACHE_DIR, CoolData, CoolMetadata, SurfaceMesh, UnknownDesignError
from coolfrog.engine.tadpole import Tadpole
from coolfrog.lib.config import TadpoleConfig
from coolfrog.lib.datatypes import CHANNELS, PARAM_COLUMNS, PARAM_UNITS, SURFACE_CHANNELS, FlowField, Grid, NormalizationStats
from coolfrog.lib.log import get_logger, setup_logging
from coolfrog.lib.metrics import nrmse, spectral_loss

app = FastAPI(title="Coolfrog")
GRID = Grid((400, 80, 16))
LOCAL_SOURCE = Path("data/cooldata")
# Numeric columns the dataset explorer filters and sorts, `run` is filtered separately
DATASET_COLUMNS = ("design_id", *PARAM_COLUMNS)
HISTOGRAM_BINS = 40
setup_logging()
logger = get_logger(__name__)
# VTK readers share state, so uncached designs queue here and abandoned requests are skipped
VTK_QUEUE = asyncio.Lock()


class Parameter(BaseModel):
    name: str
    unit: str | None
    min: float
    max: float


ScaleKind = Literal["linear", "log"]
# k and ε span decades and are colored on a log scale, every other channel linearly
LOG_SCALE_CHANNELS = {"k", "ε"}


class Channel(BaseModel):
    name: str
    unit: str
    scale: ScaleKind

    @classmethod
    def from_label(cls, label: str) -> "Channel":
        """`Vx [m/s]` → name `Vx`, unit `m/s`, log color scale for `LOG_SCALE_CHANNELS`."""
        name, _, unit = label.partition(" [")
        return cls(name=name, unit=unit.removesuffix("]"), scale="log" if name in LOG_SCALE_CHANNELS else "linear")


class Config(BaseModel):
    """Sampling inputs and the grid every field is returned on, `dimensions` and `spacing` in x, y, z."""

    parameters: list[Parameter]
    channels: list[Channel]
    surface_channels: list[Channel]
    dimensions: tuple[int, int, int]
    spacing: tuple[float, float, float]


def check_ranges(ranges: dict[str, tuple[float, float]], columns: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    """Reject unknown columns and inverted bounds."""
    for name, (low, high) in ranges.items():
        if name not in columns:
            raise ValueError(f"Unknown column: {name}")
        if low > high:
            raise ValueError(f"{name}: minimum must not exceed maximum")
    return ranges


class SampleRequest(BaseModel):
    """Inclusive parameter bounds and a reproducible metadata-only sample."""

    count: Annotated[int, Field(strict=True, ge=1, le=12)]
    seed: Annotated[int, Field(strict=True, ge=0)]
    ranges: dict[str, tuple[FiniteFloat, FiniteFloat]] = Field(default_factory=dict)

    @field_validator("ranges")
    @classmethod
    def validate_ranges(cls, ranges: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        return check_ranges(ranges, PARAM_COLUMNS)


class Design(BaseModel):
    id: int
    params: dict[str, float]


class Sample(BaseModel):
    designs: list[Design]
    matched: int


class DatasetColumn(BaseModel):
    """Numeric column with its bounds and the histogram of all rows in `HISTOGRAM_BINS` equal bins."""

    name: str
    unit: str | None
    min: float
    max: float
    counts: list[int]


class Dataset(BaseModel):
    """Static overview, `header` orders `/api/dataset/rows` rows."""

    total: int
    header: list[str]
    columns: list[DatasetColumn]
    runs: dict[str, int]


class DatasetFilter(BaseModel):
    """Inclusive column ranges, design ids starting with `search` ("" = all) and the runs to keep, None = every run."""

    ranges: dict[str, tuple[FiniteFloat, FiniteFloat]] = Field(default_factory=dict)
    search: Annotated[str, Field(pattern=r"^\d*$")] = ""
    runs: list[str] | None = None

    @field_validator("ranges")
    @classmethod
    def validate_ranges(cls, ranges: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        return check_ranges(ranges, DATASET_COLUMNS)

    def apply(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Rows inside the ranges and search, then those also inside the run filter."""
        in_ranges = CoolMetadata.load().select(ranges=self.ranges).designs
        if self.search:
            in_ranges = in_ranges.filter(pl.col("design_id").cast(pl.String).str.starts_with(self.search))
        return in_ranges, in_ranges if self.runs is None else in_ranges.filter(pl.col("run").is_in(self.runs))


class RowsQuery(DatasetFilter):
    """One sorted slice of the filtered rows."""

    sort: str = "design_id"
    descending: bool = False
    offset: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1, le=500)] = 100

    @field_validator("sort")
    @classmethod
    def validate_sort(cls, name: str) -> str:
        if name not in DATASET_COLUMNS:
            raise ValueError(f"Unknown column: {name}")
        return name


class DatasetSummary(BaseModel):
    """Filtered row count, per-column histograms on the overview bins, run counts ignoring the run filter."""

    matched: int
    histograms: dict[str, list[int]]
    runs: dict[str, int]


class ReconstructionMetrics(BaseModel):
    """Per channel in `CHANNELS` order, `nrmse` in percent, `spectral_loss` on the per-design [-1, 1] scale Tadpole sees."""

    nrmse: list[float]
    spectral_loss: list[float]


@app.exception_handler(Exception)
async def server_error(_request: Request, exc: Exception) -> JSONResponse:
    """Show download, voxelization and other server failures in the browser."""
    logger.error("Request failed", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/api/config")
def config() -> Config:
    metadata = CoolMetadata.load().designs
    parameters = []
    for name in PARAM_COLUMNS:
        finite = metadata[name].filter(metadata[name].is_finite())
        if finite.is_empty():
            raise HTTPException(503, f"No finite metadata bounds for {name}")
        parameters.append(Parameter(name=name, unit=PARAM_UNITS.get(name), min=finite.min(), max=finite.max()))
    channels = [Channel.from_label(channel) for channel in CHANNELS]
    surface_channels = [Channel.from_label(channel) for channel in SURFACE_CHANNELS]
    return Config(
        parameters=parameters, channels=channels, surface_channels=surface_channels, dimensions=GRID.resolution, spacing=GRID.spacing
    )


@app.post("/api/sample")
def sample(request: SampleRequest) -> Sample:
    matches = CoolMetadata.load().select(ranges=request.ranges)
    matched = matches.designs.height
    if matched < request.count:
        raise HTTPException(
            400,
            f"Only {matched} designs match these ranges; requested {request.count}. Widen the ranges or reduce the count.",
        )
    selected = matches.select(ranges={}, n_samples=request.count, seed=request.seed).designs
    designs = [design_from_row(row) for row in selected.iter_rows(named=True)]
    return Sample(designs=designs, matched=matched)


def design_from_row(row: dict) -> Design:
    return Design(id=row["design_id"], params={name: row[name] for name in PARAM_COLUMNS})


@cache
def dataset_overview() -> Dataset:
    metadata = CoolMetadata.load().designs
    columns = []
    for name in DATASET_COLUMNS:
        values = metadata[name].to_numpy()
        bounds = (float(values.min()), float(values.max()))
        counts = np.histogram(values, bins=HISTOGRAM_BINS, range=bounds)[0]
        columns.append(DatasetColumn(name=name, unit=PARAM_UNITS.get(name), min=bounds[0], max=bounds[1], counts=counts.tolist()))
    runs = dict(metadata["run"].value_counts().sort("run").iter_rows())
    return Dataset(total=metadata.height, header=metadata.columns, columns=columns, runs=runs)


@app.get("/api/dataset")
def dataset() -> Dataset:
    return dataset_overview()


@app.get("/api/designs")
def designs(ids: Annotated[list[int], Query(alias="id", min_length=1, max_length=24)]) -> list[Design]:
    """Designs in the requested order."""
    rows = {row["design_id"]: row for row in CoolMetadata.load().designs.filter(pl.col("design_id").is_in(ids)).iter_rows(named=True)}
    if missing := [design_id for design_id in ids if design_id not in rows]:
        raise HTTPException(404, f"Unknown designs: {missing}")
    return [design_from_row(rows[design_id]) for design_id in ids]


@app.post("/api/dataset/summary")
def dataset_summary(dataset_filter: DatasetFilter) -> DatasetSummary:
    in_ranges, matches = dataset_filter.apply()
    histograms = {
        column.name: np.histogram(matches[column.name].to_numpy(), bins=HISTOGRAM_BINS, range=(column.min, column.max))[0].tolist()
        for column in dataset_overview().columns
    }
    return DatasetSummary(matched=matches.height, histograms=histograms, runs=dict(in_ranges["run"].value_counts().iter_rows()))


@app.post("/api/dataset/rows")
def dataset_rows(query: RowsQuery) -> list[list[int | float | str]]:
    _, matches = query.apply()
    page = matches.sort(query.sort, descending=query.descending, maintain_order=True).slice(query.offset, query.limit)
    return [list(row) for row in page.rows()]


@app.get("/api/fields/{design_id}", response_class=Response)
async def fields(request: Request, design_id: int, channel: Annotated[int, Query(ge=0, lt=len(CHANNELS))] = 0) -> Response:
    """Channel values as float32 little-endian, then the uint8 fluid mask (fluid = finite value), both x-fastest."""
    cached = CoolData.is_cached(design_id, GRID)
    flow = await _load(request, cached, CoolData.load_flow_field, design_id, GRID, LOCAL_SOURCE)
    return _field_response(flow.fields[channel], flow, channel)


@app.get("/api/reconstruction/{design_id}", response_class=Response)
async def reconstruction(request: Request, design_id: int, channel: Annotated[int, Query(ge=0, lt=len(CHANNELS))] = 0) -> Response:
    """Zero-shot Tadpole-B reconstruction of one channel, laid out like `/api/fields` with the simulation's fluid mask."""
    cached = reconstruction_path(design_id).exists() and CoolData.is_cached(design_id, GRID)
    flow, reconstructed = await _load(request, cached, load_reconstruction, design_id)
    return _field_response(reconstructed[channel], flow, channel)


@app.get("/api/reconstruction/{design_id}/metrics")
async def reconstruction_metrics(request: Request, design_id: int) -> ReconstructionMetrics:
    """NRMSE and spectral loss of the zero-shot reconstruction, `coolfrog.lib.metrics` definitions."""
    cached = reconstruction_path(design_id).exists() and CoolData.is_cached(design_id, GRID)
    flow, reconstructed = await _load(request, cached, load_reconstruction, design_id)
    reference = flow.unsqueeze(0)
    stats = NormalizationStats.from_flowfields([reference])
    prediction = reference.replace(fields=reconstructed.unsqueeze(0)).normalize(stats)
    reference = reference.normalize(stats)
    return ReconstructionMetrics(
        nrmse=nrmse(prediction, reference).row(0)[1:],
        spectral_loss=spectral_loss(prediction, reference).row(0)[1:],
    )


def _field_response(values: torch.Tensor, flow: FlowField, channel: int) -> Response:
    """`values` then the fluid mask of `flow`'s channel, see `/api/fields`."""
    values = np.asarray(values.numpy(), dtype="<f4").ravel(order="F")
    mask = flow.mask.numpy().ravel(order="F") & np.isfinite(flow.fields[channel].numpy().ravel(order="F"))
    if not mask.any():
        raise HTTPException(422, f"Design {flow.design_id.item()} has no finite fluid values for {CHANNELS[channel]}")
    return Response(values.tobytes() + mask.astype(np.uint8).tobytes(), media_type="application/octet-stream")


@cache
def tadpole() -> Tadpole:
    return Tadpole(TadpoleConfig(size="B")).eval()


def reconstruction_path(design_id: int) -> Path:
    voxels_x, voxels_y, voxels_z = GRID.resolution
    return CACHE_DIR / f"tadpole_b_zero_shot_{voxels_x}x{voxels_y}x{voxels_z}" / f"{design_id}.pt"


def load_reconstruction(design_id: int) -> tuple[FlowField, torch.Tensor]:
    """Simulation and its cached zero-shot reconstruction (C, X, Y, Z) in physical units.

    - Each channel is scaled to [-1, 1] by this design's own fluid min and max, Tadpole's pre-training range
    - Takes about 45 s on 8 CPU cores when uncached
    """
    flow = CoolData.load_flow_field(design_id, GRID, LOCAL_SOURCE)
    path = reconstruction_path(design_id)
    if path.exists():
        return flow, torch.load(path, weights_only=True)
    batch = flow.unsqueeze(0)
    with torch.inference_mode():
        decoded, _ = tadpole()(batch.normalize(NormalizationStats.from_flowfields([batch])), sample=False)
    reconstructed = decoded.denormalize().fields[0].clone()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reconstructed, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
    return flow, reconstructed


@app.get("/api/surface/{design_id}", response_class=Response)
async def surface(request: Request, design_id: int) -> Response:
    """Boundary faces, all little-endian.

    - uint32 point count, polygon entry count, face count
    - float32 x, y, z per point, uint32 legacy VTK polygons [n, i0, …]
    - float32 face values per `SURFACE_CHANNELS` entry, NaN where a boundary doesn't store it
    """
    mesh = await _load(request, SurfaceMesh.cache_path(design_id).exists(), SurfaceMesh.load, design_id, LOCAL_SOURCE)
    counts = np.array([len(mesh.points), len(mesh.polygons), mesh.values.shape[1]], dtype="<u4")
    arrays = (counts, mesh.points.astype("<f4"), mesh.polygons.astype("<u4"), mesh.values.astype("<f4"))
    return Response(b"".join(array.tobytes() for array in arrays), media_type="application/octet-stream")


async def _load[Loaded](request: Request, cached: bool, load: Callable[..., Loaded], *args: object) -> Loaded:
    """Cached designs load concurrently, uncached ones wait in `VTK_QUEUE` unless the browser left meanwhile."""
    try:
        if cached:
            return await run_in_threadpool(load, *args)
        async with VTK_QUEUE:
            if await request.is_disconnected():
                raise HTTPException(499, "Client disconnected")
            return await run_in_threadpool(load, *args)
    except UnknownDesignError as error:
        raise HTTPException(404, str(error)) from error
