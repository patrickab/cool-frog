# CoolData value distribution

## Goal

Measure the fluid value distribution of every channel over all CoolData designs: histograms, quantiles and per-design stats, to identify outlier designs and choose a normalization.

## Result

All 5692 designs at 400x80x16, 1.5 min sweep. Analysis and plots: [`notebooks/02_cooldata_value_distribution.ipynb`](../../notebooks/02_cooldata_value_distribution.ipynb).

- [`results/excluded_designs.csv`](results/excluded_designs.csv): `design_id` and `reason` of every design breaking the outlier rule
- [`results/normalization_stats.csv`](results/normalization_stats.csv): `channel`, `transform`, `low`, `high`, read by `NormalizationStats.load`
- [`results/channel_stats.csv`](results/channel_stats.csv): min, max, mean, std and quantiles per channel, all designs
- [`results/design_stats.parquet`](results/design_stats.parquet): the same per design and channel
- [`results/histograms.npz`](results/histograms.npz): 2048-bin histograms per design and channel, asinh-spaced

| channel | q0.0001 | median | q0.9999 | max |
|---|---|---|---|---|
| Vx [m/s] | -13.3 | 2.36 | 46.2 | 204 |
| Vy [m/s] | -13.3 | 0.00 | 13.5 | 1257 |
| Vz [m/s] | -5.4 | 0.00 | 9.1 | 2811 |
| p [Pa] | -1259 | 0.01 | 2408 | 5.2e7 |
| T [K] | 293.12 | 295.5 | 3862 | 4055 |
| k [m²/s²] | 1.6e-7 | 0.12 | 251 | 9.3e4 |
| ε [m²/s³] | 3.4e-5 | 10.3 | 7.3e5 | 1.4e12 |

25 designs break the outlier rule (any velocity component > 100 m/s or T outside 250 to 600 K):

- 3 entirely (243826, 128001, 245751): T median 2234 to 3862 K
- 22 locally: rule-breaking voxels mostly at the inlet face, plus one coherent 100 to 164 m/s jet (2248)

Without them the tails collapse, e.g. q0.9999 of T 3862 → 343 K, of k 251 → 70 m²/s².

Normalization of the 5667 curated designs: q0.0001 → -1, q0.9999 → 1, log10 for k and ε, nothing clipped. Curated extremes after normalization:

| channel | transform | normalized min | normalized max |
|---|---|---|---|
| Vx | linear | -2.2 | 2.9 |
| Vy | linear | -7.3 | 7.5 |
| Vz | linear | -4.6 | 14.5 |
| p | linear | -2.2 | 13.3 |
| T | linear | -1.7 | 1.4 |
| k | log10 | -1.7 | 1.2 |
| ε | log10 | -2.1 | 1.3 |

## Conclusion

- Exclude the 25 flagged designs (0.4 % of the data) as physically implausible, likely diverged simulations.
- Put k and ε on a log scale: they span ~9 to 10 decades between q0.0001 and q0.9999, which no linear [-1, 1] scale resolves.
- Normalize the remaining channels by quantiles of the curated set, not min-max. p keeps a minor bump around 1e4 Pa from unflagged designs.
- Heavy tails remain after normalization (Vz up to 14.5, p 13.3, Vy 7.5) and are left unclipped for now.
- Next: rerun at 1600x320x64 to confirm the distribution and the outlier list at training resolution.

## Open questions

- MSE in log10 space weighs relative errors of k and ε in quiescent regions as much as in turbulent ones: keep it, or weight the loss?
