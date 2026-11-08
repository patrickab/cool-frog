"""Local backend of the training monitor page: SSH hosts from `monitor-hosts.json`, one tunnel per host to its training.

Run by `run-frontend.sh --monitor`, or `uvicorn coolfrog.api.monitor_hub:app --port 8015`.

- Address only -> plain `ssh` (keys, agent, Tailscale SSH), address and password -> `sshpass`
- Each host gets its own local port, forwarded to the training's `MONITOR_PORT` on that host
- `monitor-hosts.json` is gitignored and holds passwords in plain text, written owner-readable only
"""

from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import http
import http.client
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import polars as pl
from pydantic import BaseModel, Field

from coolfrog.lib import plots
from coolfrog.lib.log import setup_logging
from coolfrog.lib.monitor import MONITOR_PORT

HOSTS_PATH = Path("monitor-hosts.json")
TUNNEL_TIMEOUT_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 10
SSHPASS_WRONG_PASSWORD = 5  # exit code of `sshpass`
TABLE_CACHE: dict[tuple[str, str], pl.DataFrame] = {}  # (address, table) -> rows fetched so far, for `cached_table`

setup_logging()
logger = logging.getLogger(__name__)


class Host(BaseModel):
    """`address` is anything `ssh` accepts: `user@100.64.0.2`, a Tailscale name or a `~/.ssh/config` alias."""

    address: str = Field(min_length=1, pattern=r"^[^\s-][^\s]*$")  # no leading `-`, never parsed as an ssh option
    password: str | None = None


class HostInfo(BaseModel):
    """A configured host, never with its password."""

    address: str
    has_password: bool


class HostAddress(BaseModel):
    address: str


@dataclass
class Tunnel:
    process: subprocess.Popen[bytes]
    port: int


TUNNELS: dict[str, Tunnel] = {}
TUNNEL_LOCKS: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)  # one connect per host, others poll on


def load_hosts() -> list[Host]:
    return [Host(**host) for host in json.loads(HOSTS_PATH.read_text())] if HOSTS_PATH.exists() else []


def save_hosts(hosts: list[Host]) -> list[HostInfo]:
    """Write `hosts` readable by the owner only, as they may hold passwords."""
    descriptor = os.open(HOSTS_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump([host.model_dump(exclude_none=True) for host in hosts], file, indent=2)
    return host_infos(hosts)


def host_infos(hosts: list[Host]) -> list[HostInfo]:
    return [HostInfo(address=host.address, has_password=host.password is not None) for host in hosts]


def find_host(address: str) -> Host:
    for host in load_hosts():
        if host.address == address:
            return host
    raise HTTPException(404, f"No host {address} in {HOSTS_PATH}")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def is_listening(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def start_tunnel(host: Host) -> Tunnel:
    """Forward a free local port to the host's `MONITOR_PORT`, 502 when SSH fails, 504 when it hangs."""
    port = free_port()
    command = [
        "ssh", "-N",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "StrictHostKeyChecking=accept-new",  # first connection trusts the host key, as answering `yes` would
        "-o", "ServerAliveInterval=15",
        "-L", f"{port}:127.0.0.1:{MONITOR_PORT}",
        host.address,
    ]  # fmt: skip
    environment = dict(os.environ)
    if host.password is None:
        command[2:2] = ["-o", "BatchMode=yes"]  # fail instead of waiting for a password prompt nobody sees
    elif shutil.which("sshpass") is None:
        raise HTTPException(500, f"{host.address} has a password, which needs `sshpass` installed")
    else:
        command = ["sshpass", "-e", *command]
        environment["SSHPASS"] = host.password  # environment, not argv, keeps it out of `ps`
    # A file, not a pipe: "channel open failed" lines from polls without a training would fill a pipe and block ssh.
    # ssh keeps writing to its own descriptor after this one closes
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr)
        deadline = time.monotonic() + TUNNEL_TIMEOUT_SECONDS
        while not is_listening(port):
            if process.poll() is not None:
                stderr.seek(0)
                message = stderr.read().decode(errors="replace").strip()
                wrong_password = host.password is not None and process.returncode == SSHPASS_WRONG_PASSWORD
                reason = "wrong password" if wrong_password else message or f"exit code {process.returncode}"
                raise HTTPException(502, f"SSH to {host.address} failed: {reason}")
            if time.monotonic() > deadline:
                process.kill()
                raise HTTPException(504, f"SSH to {host.address} did not connect within {TUNNEL_TIMEOUT_SECONDS} s")
            time.sleep(0.1)
    logger.info("Tunnel 127.0.0.1:%d -> %s:%d", port, host.address, MONITOR_PORT)
    return Tunnel(process, port)


def tunnel_port(host: Host) -> int:
    """Local port of the host's tunnel, reconnecting when SSH exited."""
    with TUNNEL_LOCKS[host.address]:
        tunnel = TUNNELS.get(host.address)
        if tunnel is None or tunnel.process.poll() is not None:
            tunnel = TUNNELS[host.address] = start_tunnel(host)
        return tunnel.port


def close_tunnel(address: str) -> None:
    tunnel = TUNNELS.pop(address, None)
    if tunnel is not None:
        tunnel.process.terminate()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    for address in list(TUNNELS):
        close_tunnel(address)


app = FastAPI(title="Coolfrog monitor hub", lifespan=lifespan)


@app.exception_handler(Exception)
async def server_error(_request: Request, exc: Exception) -> JSONResponse:
    logger.error("Request failed", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/api/hosts")
def hosts() -> list[HostInfo]:
    return host_infos(load_hosts())


@app.post("/api/hosts")
def add_host(host: Host) -> list[HostInfo]:
    """Add `host`, replacing the one with the same address, e.g. to change its password."""
    close_tunnel(host.address)
    return save_hosts([*(other for other in load_hosts() if other.address != host.address), host])


@app.post("/api/hosts/remove")
def remove_host(target: HostAddress) -> list[HostInfo]:
    close_tunnel(target.address)
    return save_hosts([host for host in load_hosts() if host.address != target.address])


def forward(address: str, path: str, query: str = "") -> Response:
    """`coolfrog.lib.monitor` endpoint `/api/<path>` of the training on `address`, its errors passed through, 502 without training.

    Connects on the first request and reconnects once SSH exited, e.g. after the host rebooted.
    """
    port = tunnel_port(find_host(address))
    url = f"http://127.0.0.1:{port}/api/{urllib.parse.quote(path)}" + (f"?{query}" if query else "")
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return Response(response.read(), media_type="application/json")
    except urllib.error.HTTPError as error:  # the training answered, e.g. 404 for an unknown table
        return Response(error.read(), status_code=error.code, media_type="application/json")
    except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as error:
        raise HTTPException(502, f"No training serves the monitor on {address} port {MONITOR_PORT}") from error


def fetch_frame(address: str, table: str, start: int) -> pl.DataFrame:
    """Rows of the training's `table` from row `start` on, HTTPException with the training's status when it answers an error."""
    forwarded = forward(address, f"tables/{table}", f"start={start}")
    if forwarded.status_code != http.HTTPStatus.OK:
        raise HTTPException(forwarded.status_code, forwarded.body.decode())
    payload = json.loads(forwarded.body)
    return pl.DataFrame(payload["rows"], schema=payload["columns"], orient="row", infer_schema_length=None)  # columns may start null


def cached_table(address: str, table: str) -> pl.DataFrame:
    """All rows of the training's `table`, only rows after the cached ones cross the tunnel.

    - The last cached row is fetched again, if it differs, e.g. a new run started, the cache refills from row 0
    """
    key = (address, table)
    cached = TABLE_CACHE.get(key)
    if cached is not None and cached.height:
        new = fetch_frame(address, table, start=cached.height - 1)
        if new.height and new.row(0) == cached.row(-1):
            TABLE_CACHE[key] = pl.concat([cached, new.slice(1)], how="vertical_relaxed")
            return TABLE_CACHE[key]
    TABLE_CACHE[key] = fetch_frame(address, table, start=0)
    return TABLE_CACHE[key]


@app.get("/api/hosts/monitor/figures/{table:path}.svg")
def table_figure(table: str, address: str, dark: bool = False) -> Response:
    """`plots.figure_for_table` of `table` on `address` as SVG, 404 when nothing to plot, raw metric tables 404 without fetching."""
    if not plots.has_figure(table):
        raise HTTPException(404, f"No figure for table {table}")
    frame = cached_table(address, table)
    with plots.plot_theme(dark):
        figure = plots.figure_for_table(table, frame)
        if figure is None:
            raise HTTPException(404, f"Nothing to plot in table {table}")
        return Response(plots.svg_bytes(figure), media_type="image/svg+xml")


@app.get("/api/hosts/monitor/{path:path}")
def monitor(request: Request, path: str, address: str) -> Response:
    """`coolfrog.lib.monitor` endpoint `/api/<path>` of the training on `address`, passed through unchanged.

    e.g. `/api/hosts/monitor/state?address=...`, `/api/hosts/monitor/tables/train?address=...&tail=200`
    """
    query = urllib.parse.urlencode([(key, value) for key, value in request.query_params.multi_items() if key != "address"])
    return forward(address, path, query)
