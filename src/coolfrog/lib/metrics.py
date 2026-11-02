"""Generic metric logging: named scalar series over steps, shared by training loops, dashboards and table export."""

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping
import threading

import polars as pl


class MetricTracker:
    """Thread-safe store of named scalar series, each logged at increasing integer steps.

    - Series are sparse: validation can log every 1000 steps next to a loss logged every step
    - `version` grows with every `log`, readers redraw only when it changed
    - `request_stop` makes the next `log` raise `KeyboardInterrupt`, the cancellation point of a training thread
    - `set_progress` reports progress bars, e.g. updates done and validation designs done, also a cancellation point

    Examples
    --------
    >>> tracker = MetricTracker()
    >>> tracker.log(10, {"train/loss": 0.5, "train/lr": 1e-4})
    >>> tracker.log(10, {"val/loss": 0.7})
    >>> tracker.latest("train/loss")
    0.5
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: defaultdict[str, list[int]] = defaultdict(list)
        self._values: defaultdict[str, list[float]] = defaultdict(list)
        self._version = 0
        self._stop = threading.Event()
        self._progress: dict[str, tuple[int, int]] = {}

    @property
    def version(self) -> int:
        """Number of `log` calls so far."""
        return self._version

    @property
    def last_step(self) -> int | None:
        """Highest step of any series, None while empty."""
        with self._lock:
            return max((steps[-1] for steps in self._steps.values()), default=None)

    def request_stop(self) -> None:
        """Ask the logging thread to stop at its next `log`."""
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def raise_if_stopped(self) -> None:
        """KeyboardInterrupt after `request_stop`, call it in a loop that logs rarely."""
        if self._stop.is_set():
            raise KeyboardInterrupt("stop requested")

    def set_progress(self, label: str, completed: int, total: int) -> None:
        """Report `completed` of `total` for the progress bar `label`, KeyboardInterrupt after `request_stop`."""
        self.raise_if_stopped()
        with self._lock:
            self._progress[label] = (completed, total)

    def progress(self, label: str) -> tuple[int, int] | None:
        """Latest `(completed, total)` of the progress bar `label`, None before its first report."""
        with self._lock:
            return self._progress.get(label)

    def log(self, step: int, values: Mapping[str, float]) -> None:
        """Append `values` at `step`, ValueError when a series would go back in steps, KeyboardInterrupt after `request_stop`."""
        self.raise_if_stopped()
        with self._lock:
            for name in values:
                previous = self._steps[name][-1] if name in self._steps else step
                if step < previous:
                    raise ValueError(f"{name!r} logged at step {step} after step {previous}")
            for name, value in values.items():
                self._steps[name].append(step)
                self._values[name].append(float(value))
            self._version += 1

    def names(self) -> list[str]:
        """Series names in first-logged order."""
        with self._lock:
            return list(self._steps)

    def series(self, name: str) -> tuple[list[int], list[float]]:
        """Copy of (steps, values) of one series, empty when never logged."""
        with self._lock:
            return list(self._steps.get(name, ())), list(self._values.get(name, ()))

    def latest(self, name: str) -> float | None:
        """Last value of a series, None when never logged."""
        with self._lock:
            return self._values[name][-1] if self._steps.get(name) else None

    def value_at(self, name: str, step: int) -> float | None:
        """Value logged at exactly `step`, None when the series skipped it."""
        with self._lock:
            steps = self._steps.get(name, ())
            index = bisect_left(steps, step)
            return self._values[name][index] if index < len(steps) and steps[index] == step else None

    def recent_steps(self, names: list[str], count: int) -> list[int]:
        """The last `count` distinct steps at which any of `names` was logged, ascending."""
        with self._lock:
            steps = {step for name in names for step in self._steps.get(name, ())[-count:]}
        return sorted(steps)[-count:]

    def to_frame(self) -> pl.DataFrame:
        """Wide table with `step` and one column per series, null where a series skipped the step."""
        with self._lock:
            frames = [pl.DataFrame({"step": self._steps[name], name: self._values[name]}) for name in self._steps]
        if not frames:
            return pl.DataFrame({"step": []}, schema={"step": pl.Int64})
        frame = frames[0]
        for other in frames[1:]:
            frame = frame.join(other, on="step", how="full", coalesce=True)
        return frame.sort("step")


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
