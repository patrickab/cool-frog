# CoolData value distribution

## Goal

Measure the fluid value distribution of every channel over all CoolData designs: histograms, quantiles and per-design stats, to identify outlier designs and choose a normalization.

## Result

All 5692 designs at 1600x320x64, 2 h 10 min sweep. Analysis and plots: [`notebooks/02_cooldata_value_distribution.ipynb`](../../notebooks/02_cooldata_value_distribution.ipynb).

- [`results/excluded_designs.csv`](results/excluded_designs.csv): `design_id` and `reason` of every design breaking the outlier rule
- [`results/normalization_stats.csv`](results/normalization_stats.csv): `channel`, `transform`, `low`, `high`, read by `NormalizationStats.load`
- [`results/channel_stats.csv`](results/channel_stats.csv): min, max, mean, std and quantiles per channel, all designs
- [`results/design_stats.parquet`](results/design_stats.parquet): the same per design and channel
- [`results/histograms.npz`](results/histograms.npz): 2048-bin histograms per design and channel, asinh-spaced

| channel | q0.0001 | median | q0.9999 | max |
|---|---|---|---|---|
| Vx [m/s] | -13.7 | 2.60 | 47.5 | 534 |
| Vy [m/s] | -13.7 | 0.00 | 13.9 | 1257 |
| Vz [m/s] | -5.4 | 0.00 | 9.1 | 2811 |
| p [Pa] | -1259 | 0.01 | 2408 | 5.2e7 |
| T [K] | 293.12 | 295.6 | 3862 | 4068 |
| k [m²/s²] | 9.1e-7 | 0.14 | 266 | 9.3e4 |
| ε [m²/s³] | 4.7e-5 | 9.75 | 5.7e5 | 1.4e12 |

26 designs break the outlier rule (any velocity component > 100 m/s or T outside 250 to 600 K):

- 3 entirely (243826, 128001, 245751): T median 2234 to 3862 K
- 23 locally: rule-breaking voxels mostly at the inlet face, plus one coherent 100 to 164 m/s jet (2248)
- 247781 only at 1600x320x64: its 116 m/s peak is not resolved at 400x80x16

Without them the tails collapse, e.g. q0.9999 of T 3862 → 344 K, of k 266 → 70 m²/s².

Candidate normalizations of the 5666 curated designs, log10 for k and ε, nothing clipped:

- **qlow / qhigh**: quantiles mapped to -1 and 1
- **voxels beyond ±1**: share of curated voxels outside [-1, 1]
- **spread**: normalized width of the central 50 % of voxels (q0.25 to q0.75), larger resolves the bulk better
- **peak**: largest |normalized value| of any curated voxel, smaller keeps outliers closer to the training range

| qlow / qhigh | voxels beyond ±1 | spread Vx / Vy / Vz / p | spread T / k / ε | peak Vx / Vy / Vz / p | peak T / k / ε |
|---|---|---|---|---|---|
| 1e-4 / 0.9999 | 0.02 % | 0.149 / 0.012 / 0.006 / 0.001 | 0.24 / 0.33 / 0.40 | 2.8 / 7.4 / 14.3 / 13.3 | 1.7 / 2.0 / 2.2 |
| 1e-3 / 0.999 | 0.2 % | 0.254 / 0.024 / 0.015 / 0.002 | 0.30 / 0.42 / 0.46 | 5.2 / 15.1 / 35.0 / 28.4 | 2.1 / 2.6 / 2.5 |
| 1e-2 / 0.99 | 2 % | 0.472 / 0.067 / 0.054 / 0.010 | 0.58 / 0.57 / 0.56 | 10.2 / 42.1 / 123.9 / 140.9 | 4.9 / 3.8 / 3.0 |

[`results/normalization_stats.csv`](results/normalization_stats.csv) holds q0.001 / q0.999, derived from `histograms.npz` with `NORMALIZATION_QUANTILES = (1e-3, 0.999)` after the sweep (`run.log` still lists the q1e-4 / q0.9999 stats of the sweep itself):

| channel | transform | low | high |
|---|---|---|---|
| Vx | linear | -3.82 m/s | 26.2 m/s |
| Vy | linear | -5.36 m/s | 5.36 m/s |
| Vz | linear | -2.11 m/s | 3.51 m/s |
| p | linear | -136 Pa | 1175 Pa |
| T | linear | 293.15 K | 332.80 K |
| k | log10 | 1.4e-5 m²/s² | 28.4 m²/s² |
| ε | log10 | 1.6e-4 m²/s³ | 4.2e4 m²/s³ |

## Conclusion

- Exclude the 26 flagged designs (0.5 % of the data) as physically implausible, likely diverged simulations.
- Put k and ε on a log scale: they span ~8 to 10 decades between q0.0001 and q0.9999 of the curated set, which no linear [-1, 1] scale resolves.
- Normalize every channel by quantiles of the curated set, not min-max, and leave values beyond ±1 unclipped.
- Choose q0.001 / q0.999 as the trade-off between bulk resolution and peaks:
  - Narrower quantiles resolve the bulk better: the spread of Vy, Vz and p grows 5 to 10× from q1e-4 to q0.01.
  - They also grow the peaks: q0.01 / q0.99 puts Vz and p voxels at 124 and 141, and under MSE one such voxel outweighs ~10⁴ bulk voxels.
  - q0.001 / q0.999 roughly doubles the bulk spread of q1e-4 / q0.9999 and keeps peaks at 35 and below, 0.2 % of voxels beyond ±1.
  - k, ε and T stay within ±2.6 under this choice, the log scale already tames k and ε.
- The bulk of Vy, Vz and p stays narrow under every choice (spread ≤ 0.067 even at q0.01): most fluid is nearly at rest, and a few jets and pressure peaks set the scale.
- Training uses the normalization stats and exclusion list of this 1600x320x64 run. The 400x80x16 run (in `history/`) agrees except for one design.

## Open questions

- MSE in log10 space weighs relative errors of k and ε in quiescent regions as much as in turbulent ones: keep it, or weight the loss?
