"""Web monitor of a running `MetricTracker`: FastAPI on 127.0.0.1 in a daemon thread, read by `coolfrog.api.monitor_hub`.

- `GET /api/monitor` returns the `DashboardSpec` charts with their data, progress bars and `nvidia-smi` GPU stats
- Loopback only, remote machines reach it through an SSH tunnel
"""

from collections.abc import Iterator
from contextlib import contextmanager
import socket
import subprocess
import threading
import time
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
from rich.text import Text
import uvicorn

from coolfrog.lib.dashboard import DashboardSpec, SeriesSpec, progress_times
from coolfrog.lib.log import get_logger
from coolfrog.lib.metrics import MetricTracker

logger = get_logger(__name__)

MONITOR_PORT = 8020  # the SSH tunnels of `coolfrog.api.monitor_hub` forward to it
NVIDIA_SMI_FIELDS = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
NVIDIA_SMI_TIMEOUT_SECONDS = 5
SHUTDOWN_TIMEOUT_SECONDS = 5


class SeriesData(BaseModel):
    label: str
    color: str
    side: Literal["left", "right"]
    steps: list[int]
    values: list[float]


class ChartData(BaseModel):
    title: str
    log_y: bool
    series: list[SeriesData]


class ProgressData(BaseModel):
    """One progress bar, `times` as in the terminal dashboard: elapsed, remaining, total."""

    label: str
    completed: int
    total: int
    times: str


class Gpu(BaseModel):
    index: int
    name: str
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    temperature_celsius: float


class MonitorState(BaseModel):
    """Everything the web monitor shows, `last_step` None before the first `log`."""

    title: str
    summary: str
    step_header: str
    last_step: int | None
    progress: list[ProgressData]
    charts: list[ChartData]
    gpus: list[Gpu]


def query_gpus() -> list[Gpu]:
    """Current `nvidia-smi` readings, empty without `nvidia-smi`."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={NVIDIA_SMI_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        index, name, utilization, memory_used, memory_total, temperature = (field.strip() for field in line.split(","))
        gpus.append(
            Gpu(
                index=int(index),
                name=name,
                utilization_percent=float(utilization),
                memory_used_mib=float(memory_used),
                memory_total_mib=float(memory_total),
                temperature_celsius=float(temperature),
            )
        )
    return gpus


def series_data(tracker: MetricTracker, series: SeriesSpec) -> SeriesData:
    steps, values = tracker.series(series.name)
    return SeriesData(label=series.label, color=series.color, side=series.side, steps=steps, values=values)


def monitor_state(tracker: MetricTracker, spec: DashboardSpec) -> MonitorState:
    now = time.monotonic()
    progress = [
        ProgressData(label=label, completed=bar.completed, total=bar.total, times=progress_times(bar, now))
        for label in spec.progress_bars
        if (bar := tracker.progress(label)) is not None
    ]
    charts = [
        ChartData(
            title=chart.title,
            log_y=chart.log_y,
            series=[series_data(tracker, series) for series in chart.series],
        )
        for chart in spec.charts
    ]
    return MonitorState(
        title=spec.title,
        summary=Text.from_markup(spec.summary).plain,
        step_header=spec.step_header,
        last_step=tracker.last_step,
        progress=progress,
        charts=charts,
        gpus=query_gpus(),
    )


def create_monitor_app(tracker: MetricTracker, spec: DashboardSpec) -> FastAPI:
    app = FastAPI(title="Coolfrog monitor")

    @app.get("/api/monitor")
    def monitor() -> MonitorState:
        return monitor_state(tracker, spec)

    return app


@contextmanager
def serve_monitor(tracker: MetricTracker, spec: DashboardSpec, port: int = MONITOR_PORT) -> Iterator[None]:
    """Serve `tracker` on 127.0.0.1:`port` meanwhile, only a warning when the port is taken, e.g. by a second run."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", port))
    except OSError as error:
        listener.close()
        logger.warning("Web monitor off, port %d: %s", port, error)
        yield
        return
    # No log config: uvicorn's records stay out of the terminal dashboard
    server = uvicorn.Server(uvicorn.Config(create_monitor_app(tracker, spec), log_config=None, access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, name="monitor", daemon=True)
    thread.start()
    logger.info("Web monitor on 127.0.0.1:%d, view it with `run-frontend.sh --monitor`", port)
    try:
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        listener.close()
