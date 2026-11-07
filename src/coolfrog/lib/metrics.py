"""Run tracking: named tables that grow row by row plus one free-form state dictionary, read by dashboards and the API.

- Tables: one per logging rhythm, e.g. `train` every 50 updates, `validation` one row per design and channel
- State: anything that is not a series over time, e.g. title, configs, progress, debug values
"""

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

SAVE_EVERY_SECONDS = 30.0  # a killed process loses at most this much of a persisted tracker
RESERVED_STATE_KEYS = ("progress", "tables", "gpus")  # added by `snapshot` and the monitor


@dataclass(frozen=True)
class Progress:
    """One pass of a progress bar, times from `time.monotonic`, paused time excluded.

    Attributes
    ----------
    started : float
        Time of the pass's first report.
    finished : float | None
        Time `completed` reached `total`, None while running.
    paused_seconds : float
        Paused time of finished pauses, e.g. validation passes inside training.
    paused_at : float | None
        Start of the running pause, None while not paused.
    """

    completed: int
    total: int
    started: float
    finished: float | None = None
    paused_seconds: float = 0.0
    paused_at: float | None = None

    def elapsed(self, now: float) -> float:
        """Unpaused seconds since the pass started, frozen once it finished or while paused."""
        return (self.finished or self.paused_at or now) - self.started - self.paused_seconds

    def remaining(self, now: float) -> float | None:
        """Unpaused seconds left at the average rate so far, None before the first completed item."""
        if self.completed == 0:
            return None
        return self.elapsed(now) * (self.total - self.completed) / self.completed

    def to_state(self, now: float) -> dict[str, Any]:
        """JSON form in `Tracker.snapshot()["progress"]`."""
        remaining = None if self.finished is not None else self.remaining(now)
        return {
            "completed": self.completed,
            "total": self.total,
            "elapsed_seconds": round(self.elapsed(now), 1),
            "remaining_seconds": None if remaining is None else round(remaining, 1),
            "paused": self.paused_at is not None,
        }


class Table:
    """Rows stored per column, a column first seen in a later row is None in the earlier ones."""

    def __init__(self) -> None:
        self._columns: dict[str, list[Any]] = {}
        self._types: dict[str, str] = {}
        self.rows = 0

    def append(self, row: Mapping[str, Any]) -> None:
        for name in row:  # in row order, the first column stays the x of the charts
            if name not in self._columns:
                self._columns[name] = [None] * self.rows
        for name, values in self._columns.items():
            values.append(row.get(name))
        for name, value in row.items():
            if value is not None and name not in self._types:
                self._types[name] = type(value).__name__
        self.rows += 1

    @property
    def columns(self) -> dict[str, str]:
        """Column name -> Python type name of its first value (`int`, `float`, `str`), `NoneType` while all None."""
        return {name: self._types.get(name, "NoneType") for name in self._columns}

    def column(self, name: str) -> list[Any]:
        return list(self._columns.get(name, ()))

    def to_frame(self, tail: int | None = None, start: int = 0) -> pl.DataFrame:
        if tail is not None:
            start = max(self.rows - tail, start)
        return pl.DataFrame({name: values[start:] for name, values in self._columns.items()}, strict=False)


class Tracker:
    """Thread-safe tables and state of one run, saved into `folder` while it logs.

    - `append` adds one row to a table, created on first use, `extend` adds the rows of a frame
    - `update_state` sets top-level state keys, `snapshot` is a JSON-ready copy with progress and table infos
    - `set_progress`, `pause_progress`, `resume_progress` drive progress bars, shown as the snapshot's `progress`
    - `request_stop` makes the next `append` or `set_progress` raise `KeyboardInterrupt`, the cancellation point of a
      training thread
    - `folder` given -> each table as `<name>.parquet` plus `state.json`, at most every `SAVE_EVERY_SECONDS` while
      logging, `save` forces it
    - `version` grows with every change, readers redraw only when it changed

    Examples
    --------
    >>> tracker = Tracker(state={"title": "LoRA-32", "configs": configs_state([TRAIN_CONFIG])})
    >>> tracker.append("train", {"update": 50, "mse": 1.2e-3})
    >>> tracker.frame("train")
    """

    def __init__(self, folder: Path | None = None, state: Mapping[str, Any] | None = None) -> None:
        self.folder = folder
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._saved_at = time.monotonic()
        self._tables: dict[str, Table] = {}
        self._state: dict[str, Any] = _checked_state(state or {})
        self._progress: dict[str, Progress] = {}
        self._stop = threading.Event()
        self.version = 0

    def append(self, table: str, row: Mapping[str, Any]) -> None:
        """Add `row` to `table`, KeyboardInterrupt after `request_stop`."""
        self.extend(table, [row])

    def extend(self, table: str, rows: pl.DataFrame | Sequence[Mapping[str, Any]]) -> None:
        """Add `rows` to `table`, KeyboardInterrupt after `request_stop`."""
        self.raise_if_stopped()
        rows = rows.iter_rows(named=True) if isinstance(rows, pl.DataFrame) else rows
        with self._lock:
            target = self._tables.setdefault(table, Table())
            for row in rows:
                target.append(row)
            self.version += 1
        if self.folder is not None and time.monotonic() - self._saved_at > SAVE_EVERY_SECONDS:
            self.save()

    def update_state(self, values: Mapping[str, Any]) -> None:
        """Set top-level state keys, values must be JSON-ready, ValueError for `RESERVED_STATE_KEYS`."""
        values = _checked_state(values)
        with self._lock:
            self._state.update(values)
            self.version += 1

    def table_names(self) -> list[str]:
        """Tables in creation order."""
        with self._lock:
            return list(self._tables)

    def table_infos(self) -> dict[str, dict[str, Any]]:
        """Table name -> row count and column types, see `Table.columns`."""
        with self._lock:
            return {name: {"rows": table.rows, "columns": table.columns} for name, table in self._tables.items()}

    def column(self, table: str, name: str) -> list[Any]:
        """Copy of one column, empty when the table or column does not exist yet."""
        with self._lock:
            return self._tables[table].column(name) if table in self._tables else []

    def columns(self, table: str, names: Sequence[str]) -> list[list[Any]]:
        """Copies of several columns read together, so they have equal lengths, empty lists for a missing table."""
        with self._lock:
            return [self._tables[table].column(name) if table in self._tables else [] for name in names]

    def frame(self, table: str, tail: int | None = None, start: int = 0) -> pl.DataFrame:
        """`table` as a DataFrame from row `start` on, at most its last `tail` rows, KeyError for an unknown table."""
        with self._lock:
            return self._tables[table].to_frame(tail, start)

    def snapshot(self) -> dict[str, Any]:
        """JSON-ready copy of the state plus `progress` (label -> `Progress.to_state`) and `tables` (`table_infos`)."""
        now = time.monotonic()
        with self._lock:
            state = copy.deepcopy(self._state)
            progress = {label: bar.to_state(now) for label, bar in self._progress.items()}
        return state | {"progress": progress, "tables": self.table_infos()}

    def set_progress(self, label: str, completed: int, total: int) -> None:
        """Report `completed` of `total` for the bar `label`, KeyboardInterrupt after `request_stop`.

        - A first report or `completed` going back starts a new pass, e.g. the next validation, its clock restarts
        """
        self.raise_if_stopped()
        now = time.monotonic()
        with self._lock:
            previous = self._progress.get(label)
            if previous is None or completed < previous.completed:
                self._progress[label] = Progress(completed, total, started=now, finished=now if completed >= total else None)
            else:
                finished = None if completed < total else previous.finished or now
                self._progress[label] = replace(previous, completed=completed, total=total, finished=finished)
            self.version += 1

    def pause_progress(self, label: str) -> None:
        """Stop the clock of the bar `label` until `resume_progress`, e.g. training during validation."""
        with self._lock:
            previous = self._progress.get(label)
            if previous is not None and previous.paused_at is None:
                self._progress[label] = replace(previous, paused_at=time.monotonic())

    def resume_progress(self, label: str) -> None:
        with self._lock:
            previous = self._progress.get(label)
            if previous is not None and previous.paused_at is not None:
                paused_seconds = previous.paused_seconds + time.monotonic() - previous.paused_at
                self._progress[label] = replace(previous, paused_seconds=paused_seconds, paused_at=None)

    def progress(self) -> dict[str, Progress]:
        with self._lock:
            return dict(self._progress)

    def request_stop(self) -> None:
        """Ask the logging thread to stop at its next `append` or `set_progress`."""
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def raise_if_stopped(self) -> None:
        """KeyboardInterrupt after `request_stop`, call it in a loop that logs rarely."""
        if self._stop.is_set():
            raise KeyboardInterrupt("stop requested")

    def file_names(self) -> list[str]:
        """Files `save` writes into `folder`."""
        return [*(f"{name}.parquet" for name in self.table_names()), "state.json"]

    def save(self) -> None:
        """Write every table and the state into `folder` atomically, a reader never sees a half-written file."""
        if self.folder is None:
            return
        with self._save_lock:
            self.folder.mkdir(parents=True, exist_ok=True)
            for name in self.table_names():
                write_atomically(self.folder / f"{name}.parquet", lambda path, name=name: self.frame(name).write_parquet(path))
            snapshot = self.snapshot()
            write_atomically(self.folder / "state.json", lambda path: path.write_text(json.dumps(snapshot, indent=2)))
            self._saved_at = time.monotonic()


def _checked_state(values: Mapping[str, Any]) -> dict[str, Any]:
    reserved = sorted(set(values) & set(RESERVED_STATE_KEYS))
    if reserved:
        raise ValueError(f"Reserved state keys {reserved}, the snapshot sets them")
    return dict(values)


def write_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Run `write` on a temporary sibling of `path`, then rename it over `path`."""
    path_temporary = path.with_name(f".{path.name}.tmp")
    write(path_temporary)
    path_temporary.replace(path)


def configs_state(configs: Sequence["DataclassInstance"]) -> dict[str, dict[str, Any]]:
    """Config dataclasses as JSON-ready dicts keyed by class name, e.g. for `Tracker` state and `config.yaml`."""
    # json round trip turns Path, dtype and tuple into JSON types
    return json.loads(json.dumps({type(config).__name__: asdict(config) for config in configs}, default=str))


def frame_payload(frame: pl.DataFrame) -> dict[str, list[Any]]:
    """`{"columns": [...], "rows": [[...], ...]}` of `frame` for the API, non-finite floats as None."""
    rows = [[None if isinstance(value, float) and not math.isfinite(value) else value for value in row] for row in frame.rows()]
    return {"columns": frame.columns, "rows": rows}


class WindowMean:
    """Running means of named values, `flush` returns and resets them: loss averaged over the last N updates."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._count = 0

    def add(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            self._sums[name] = self._sums.get(name, 0.0) + value
        self._count += 1

    def flush(self) -> dict[str, float]:
        """Mean of everything added since the last flush, empty when nothing was added."""
        means = {name: total / self._count for name, total in self._sums.items()}
        self._sums, self._count = {}, 0
        return means
