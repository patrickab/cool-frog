"""Web monitor of a running `Tracker`: FastAPI on 127.0.0.1 in a daemon thread, read by `coolfrog.api.monitor_hub`.

- `GET /api/state`: `Tracker.snapshot` plus `gpus`, the `nvidia-smi` readings at request time
- `GET /api/tables/{name}`: one table as columns and rows, from row `start` on, `tail` limits it to its last rows
- Loopback only, remote machines reach it through an SSH tunnel
"""

from collections.abc import Iterator
from contextlib import contextmanager
import logging
import socket
import subprocess
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn

from coolfrog.lib.metrics import Tracker, frame_payload

logger = logging.getLogger(__name__)

MONITOR_PORT = 8020  # the SSH tunnels of `coolfrog.api.monitor_hub` forward to it
NVIDIA_SMI_FIELDS = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
NVIDIA_SMI_TIMEOUT_SECONDS = 5
SHUTDOWN_TIMEOUT_SECONDS = 5


class Gpu(BaseModel):
    index: int
    name: str
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    temperature_celsius: float


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


def create_monitor_app(tracker: Tracker) -> FastAPI:
    app = FastAPI(title="Coolfrog monitor")

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return tracker.snapshot() | {"gpus": [gpu.model_dump() for gpu in query_gpus()]}

    @app.get("/api/tables/{name:path}")
    def table(name: str, tail: int | None = Query(default=None, ge=1), start: int = Query(default=0, ge=0)) -> dict[str, list[Any]]:
        if name not in tracker.table_names():
            raise HTTPException(404, f"No table {name}")
        return frame_payload(tracker.frame(name, tail, start))

    return app


@contextmanager
def serve_monitor(tracker: Tracker, port: int = MONITOR_PORT) -> Iterator[None]:
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
    server = uvicorn.Server(uvicorn.Config(create_monitor_app(tracker), log_config=None, access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, name="monitor", daemon=True)
    thread.start()
    logger.info("Web monitor on 127.0.0.1:%d, view it with `run-frontend.sh --monitor`", port)
    try:
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        listener.close()
