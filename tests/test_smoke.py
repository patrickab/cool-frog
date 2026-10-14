"""Behavior tests for normalization, solid filling, metrics and design selection."""

import polars as pl
import pytest
import torch

from coolfrog.engine.data import select_designs
from coolfrog.engine.tadpole import _fill_solids
from coolfrog.lib.datatypes import FlowField, Grid, NormalizationStats
from coolfrog.lib.metrics import nrmse


def make_flowfield(fields: torch.Tensor, mask: torch.Tensor) -> FlowField:
    batch = fields.shape[0]
    return FlowField(fields=fields, mask=mask, design_id=torch.arange(batch), params=torch.zeros(batch, 35), batch_size=[batch])


def test_grid_rejects_axis_not_multiple_of_16() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        Grid(resolution=(400, 80, 15))


def test_stats_ignore_solids_and_batching() -> None:
    torch.manual_seed(0)
    fields = torch.randn(4, 5, 16, 16, 16)
    mask = torch.rand(4, 16, 16, 16) > 0.3
    fields[~mask.unsqueeze(1).expand_as(fields)] = 1e6
    whole = NormalizationStats.from_flowfields([make_flowfield(fields, mask)])
    batched = NormalizationStats.from_flowfields([make_flowfield(fields[:1], mask[:1]), make_flowfield(fields[1:], mask[1:])])
    torch.testing.assert_close(whole.channel_max, batched.channel_max)
    assert whole.channel_max.max() < 1e6


def test_stats_reject_constant_channel() -> None:
    fields = torch.randn(2, 5, 16, 16, 16)
    fields[:, 3] = 1.0
    with pytest.raises(ValueError, match="constant"):
        NormalizationStats.from_flowfields([make_flowfield(fields, torch.ones(2, 16, 16, 16, dtype=torch.bool))])


def test_normalize_maps_fluid_to_unit_range_and_roundtrips() -> None:
    torch.manual_seed(0)
    flowfield = make_flowfield(torch.randn(3, 5, 16, 16, 16) * 50 + 300, torch.rand(3, 16, 16, 16) > 0.2)
    normalized = flowfield.normalize(NormalizationStats.from_flowfields([flowfield]))
    fluid_values = normalized.fields[normalized.mask.unsqueeze(1).expand_as(normalized.fields)]
    assert fluid_values.min().item() == pytest.approx(-1, abs=1e-5)
    assert fluid_values.max().item() == pytest.approx(1, abs=1e-5)
    torch.testing.assert_close(normalized.denormalize().fields, flowfield.fields, rtol=1e-5, atol=1e-4)
    with pytest.raises(ValueError, match="already normalized"):
        normalized.normalize(normalized.stats[0])


def test_fill_solids_keeps_fluid_and_extends_constant_field() -> None:
    mask = torch.ones(1, 16, 16, 16, dtype=torch.bool)
    mask[:, 4:12, 4:12, 4:12] = False
    fields = torch.full((1, 5, 16, 16, 16), 2.5)
    fields[~mask.unsqueeze(1).expand_as(fields)] = 0.0
    torch.testing.assert_close(_fill_solids(make_flowfield(fields, mask)).fields, torch.full_like(fields, 2.5))


def test_nrmse_uses_fluid_voxels_only() -> None:
    torch.manual_seed(0)
    mask = torch.rand(2, 16, 16, 16) > 0.3
    reference = make_flowfield(torch.randn(2, 5, 16, 16, 16), mask)
    prediction_fields = reference.fields + 0.1
    prediction_fields[~mask.unsqueeze(1).expand_as(prediction_fields)] = 1e6
    table = nrmse(make_flowfield(prediction_fields, mask), reference)
    fluid_std = torch.stack([reference.fields[0, channel][mask[0]].std() for channel in range(5)])
    assert table.columns[0] == "design_id"
    torch.testing.assert_close(torch.tensor(table.row(0)[1:], dtype=torch.float32), 100 * 0.1 / fluid_std, rtol=1e-4, atol=1e-4)


def test_selection_is_inclusive_sorted_and_independent_of_row_order() -> None:
    metadata = pl.DataFrame({"design_id": list(range(100)), "V": [design_id / 10 for design_id in range(100)]})
    selected = select_designs(metadata, ranges={"V": (2.0, None)}, n_samples=5, seed=3)["design_id"].to_list()
    assert selected == sorted(selected)
    assert min(selected) >= 20
    assert select_designs(metadata.reverse(), ranges={"V": (2.0, None)}, n_samples=5, seed=3)["design_id"].to_list() == selected
    assert select_designs(metadata, ranges={"V": (2.0, 5.0)})["design_id"].to_list() == list(range(20, 51))
