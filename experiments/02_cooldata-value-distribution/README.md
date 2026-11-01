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

Normalization of the 5666 curated designs: q0.0001 → -1, q0.9999 → 1, log10 for k and ε, nothing clipped. Curated extremes after normalization:

| channel | transform | low | high | normalized min | normalized max |
|---|---|---|---|---|---|
| Vx | linear | -7.98 m/s | 43.1 m/s | -2.2 | 2.8 |
| Vy | linear | -10.9 m/s | 11.0 m/s | -7.0 | 7.4 |
| Vz | linear | -4.86 m/s | 8.68 m/s | -4.5 | 14.3 |
| p | linear | -314 Pa | 2408 Pa | -2.6 | 13.3 |
| T | linear | 293.12 K | 343.99 K | -1.7 | 1.4 |
| k | log10 | 9.0e-7 m²/s² | 69.6 m²/s² | -2.0 | 1.3 |
| ε | log10 | 4.7e-5 m²/s³ | 2.3e5 m²/s³ | -2.2 | 1.5 |

## Conclusion

- Exclude the 26 flagged designs (0.5 % of the data) as physically implausible, likely diverged simulations.
- Put k and ε on a log scale: they span ~8 to 10 decades between q0.0001 and q0.9999 of the curated set, which no linear [-1, 1] scale resolves.
- Normalize the remaining channels by quantiles of the curated set, not min-max. p keeps a minor bump around 1e4 Pa from unflagged designs.
- Heavy tails remain after normalization (Vz up to 14.3, p 13.3, Vy 7.4) and are left unclipped for now.
- Training uses the normalization stats and exclusion list of this 1600x320x64 run. The 400x80x16 run (in `history/`) agrees except for one design.

## Open questions

- MSE in log10 space weighs relative errors of k and ε in quiescent regions as much as in turbulent ones: keep it, or weight the loss?
