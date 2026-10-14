"""CoolData browser API. Run with `uvicorn coolfrog.api.main:app --port 8010`."""

import asyncio
from functools import cache
import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
import numpy as np
import polars as pl
from pydantic import BaseModel, Field, FiniteFloat, field_validator

from coolfrog.engine.data import UnknownDesignError, cache_path, load_flow_field, load_metadata, select_designs
from coolfrog.lib.datatypes import CHANNELS, PARAM_COLUMNS, PARAM_UNITS, FlowField, Grid

app = FastAPI(title="Coolfrog")
GRID = Grid((400, 80, 16))
LOCAL_SOURCE = Path("data/cooldata")
# Numeric columns the dataset explorer filters and sorts, `run` is filtered separately
DATASET_COLUMNS = ("design_id", *PARAM_COLUMNS)
HISTOGRAM_BINS = 40
logger = logging.getLogger(__name__)
# `load_flow_field` voxelizes non-thread-safe, so uncached designs queue here and abandoned requests are skipped
VOXELIZE_QUEUE = asyncio.Lock()


class Parameter(BaseModel):
    name: str
    unit: str | None
    min: float
    max: float


class Channel(BaseModel):
    name: str
    unit: str


class Config(BaseModel):
    """Sampling inputs and the grid every field is returned on, `dimensions` and `spacing` in x, y, z."""

    parameters: list[Parameter]
    channels: list[Channel]
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
    """Static overview, `header` is the column order of `/api/dataset/rows` rows."""

    total: int
    header: list[str]
    columns: list[DatasetColumn]
    runs: dict[str, int]


class DatasetFilter(BaseModel):
    """Inclusive column ranges and the runs to keep, None = every run."""

    ranges: dict[str, tuple[FiniteFloat, FiniteFloat]] = Field(default_factory=dict)
    runs: list[str] | None = None

    @field_validator("ranges")
    @classmethod
    def validate_ranges(cls, ranges: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        return check_ranges(ranges, DATASET_COLUMNS)

    def apply(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Rows inside the ranges, then those also inside the run filter."""
        in_ranges = select_designs(load_metadata(), ranges=self.ranges)
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


@app.exception_handler(Exception)
async def server_error(_request: Request, exc: Exception) -> JSONResponse:
    """Show download, voxelization and other server failures in the browser."""
    logger.error("Request failed", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/api/config")
def config() -> Config:
    metadata = load_metadata()
    parameters = []
    for name in PARAM_COLUMNS:
        finite = metadata[name].filter(metadata[name].is_finite())
        if finite.is_empty():
            raise HTTPException(503, f"No finite metadata bounds for {name}")
        parameters.append(Parameter(name=name, unit=PARAM_UNITS.get(name), min=finite.min(), max=finite.max()))
    channels = [Channel(name=channel.symbol, unit=channel.unit) for channel in CHANNELS]
    return Config(parameters=parameters, channels=channels, dimensions=GRID.resolution, spacing=GRID.spacing)


@app.post("/api/sample")
def sample(request: SampleRequest) -> Sample:
    matches = select_designs(load_metadata(), ranges=request.ranges)
    if matches.height < request.count:
        raise HTTPException(
            400,
            f"Only {matches.height} designs match these ranges; requested {request.count}. Widen the ranges or reduce the count.",
        )
    selected = select_designs(matches, ranges={}, n_samples=request.count, seed=request.seed)
    designs = [
        Design(id=row["design_id"], params={name: row[name] for name in PARAM_COLUMNS}) for row in selected.iter_rows(named=True)
    ]
    return Sample(designs=designs, matched=matches.height)


@cache
def dataset_overview() -> Dataset:
    metadata = load_metadata()
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
    flow = await _load_flow(request, design_id)
    values = np.asarray(flow.fields[channel].numpy(), dtype="<f4").ravel(order="F")
    mask = flow.mask.numpy().ravel(order="F") & np.isfinite(values)
    if not mask.any():
        raise HTTPException(422, f"Design {design_id} has no finite fluid values for {CHANNELS[channel]}")
    return Response(values.tobytes() + mask.astype(np.uint8).tobytes(), media_type="application/octet-stream")


async def _load_flow(request: Request, design_id: int) -> FlowField:
    """Cached designs load concurrently, uncached ones wait in `VOXELIZE_QUEUE` unless the browser left meanwhile."""
    try:
        if cache_path(design_id, GRID).exists():
            return await run_in_threadpool(load_flow_field, design_id, GRID, LOCAL_SOURCE)
        async with VOXELIZE_QUEUE:
            if await request.is_disconnected():
                raise HTTPException(499, "Client disconnected")
            return await run_in_threadpool(load_flow_field, design_id, GRID, LOCAL_SOURCE)
    except UnknownDesignError as error:
        raise HTTPException(404, str(error)) from error
