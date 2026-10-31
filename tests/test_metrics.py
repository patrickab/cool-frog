import polars as pl
import pytest

from coolfrog.lib.dashboard import linear_ticks, log_ticks
from coolfrog.lib.metrics import MetricTracker, WindowMean


def test_sparse_series_join_on_step() -> None:
    tracker = MetricTracker()
    tracker.log(0, {"val": 3.0})
    tracker.log(10, {"loss": 1.0})
    tracker.log(10, {"val": 2.0})
    assert tracker.to_frame().to_dicts() == [{"step": 0, "val": 3.0, "loss": None}, {"step": 10, "val": 2.0, "loss": 1.0}]
    assert tracker.value_at("loss", 0) is None
    assert tracker.recent_steps(["loss", "val"], 5) == [0, 10]


def test_log_rejects_going_back_without_partial_write() -> None:
    tracker = MetricTracker()
    tracker.log(5, {"a": 1.0})
    with pytest.raises(ValueError, match="after step 5"):
        tracker.log(4, {"b": 1.0, "a": 2.0})
    assert tracker.names() == ["a"]
    assert tracker.series("a") == ([5], [1.0])


def test_stop_request_interrupts_next_log() -> None:
    tracker = MetricTracker()
    tracker.request_stop()
    with pytest.raises(KeyboardInterrupt):
        tracker.log(1, {"a": 1.0})


def test_window_mean_resets_on_flush() -> None:
    window = WindowMean()
    window.add({"a": 1.0})
    window.add({"a": 3.0})
    assert window.flush() == {"a": 2.0}
    assert window.flush() == {}


def test_empty_tracker_frame_has_step_column() -> None:
    assert MetricTracker().to_frame().schema == {"step": pl.Int64}


@pytest.mark.parametrize("values", [[3.1e-3, 4.2e-3], [1e-3, 1e-3], [5e-9, 2e3]])
def test_log_ticks_cover_narrow_and_wide_ranges(values: list[float]) -> None:
    ticks, labels = log_ticks(values)
    assert 1 <= len(ticks) <= 6
    assert len(ticks) == len(labels)
    assert all(min(values) <= tick <= max(values) for tick in ticks)


def test_linear_ticks_single_point() -> None:
    assert linear_ticks([0]) == ([0.0], ["0"])
