# CoolData value distribution

## Goal

Measure the fluid value distribution of every channel over all CoolData designs: histograms, quantiles and per-design stats, to identify outlier designs and choose a normalization.

## Result

All 56916 designs at 800x160x32, the training grid, 4 h 21 min sweep on cube1. Analysis and plots: [`notebooks/02_cooldata_value_distribution.ipynb`](../../notebooks/02_cooldata_value_distribution.ipynb).

- [`results/excluded_designs.csv`](results/excluded_designs.csv): `design_id` and `reason` of every design breaking the outlier rule
- [`results/normalization_stats.csv`](results/normalization_stats.csv): `channel`, `transform`, `low`, `high`, read by `NormalizationStats.load`
- [`results/channel_stats.csv`](results/channel_stats.csv): min, max, mean, std and quantiles per channel, all designs
- [`results/design_stats.parquet`](results/design_stats.parquet): the same per design and channel
- [`results/histograms.npz`](results/histograms.npz): 2048-bin histograms per design and channel, asinh-spaced

| channel | q0.0001 | median | q0.9999 | max |
|---|---|---|---|---|
| Vx [m/s] | -10.6 | 2.53 | 52.5 | 1400 |
| Vy [m/s] | -12.9 | 0.00 | 12.7 | 3960 |
| Vz [m/s] | -5.2 | 0.00 | 9.2 | 8576 |
| p [Pa] | -890 | 0.01 | 2767 | 5.5e8 |
| T [K] | 293.11 | 295.5 | 3524 | 5000 |
| k [m²/s²] | 5.1e-7 | 0.14 | 188 | 2.3e7 |
| ε [m²/s³] | 5.6e-5 | 9.75 | 5.4e5 | 4.4e13 |

310 designs (0.5 %) break the outlier rule (any velocity component > 100 m/s or T outside 250 to 600 K):

- 33 entirely: T median 630 to 4701 K, 15 designs reach exactly 100 K or 5000 K, likely the solver's temperature clamps
- 277 locally: half of them in under 0.008 % of their fluid voxels, 13 in more than 1 %
- All 26 designs excluded by the 10 % sweep at 1600x320x64 ([`02_cooldata-value-distribution_partial`](../02_cooldata-value-distribution_partial)) are among them

Without them the tails collapse, e.g. q0.9999 of T 3524 → 344 K, of k 188 → 72 m²/s².

Candidate normalizations of the 56606 curated designs, log10 for k and ε, nothing clipped (definitions in the notebook):

| qlow / qhigh | voxels beyond ±1 | spread Vx / Vy / Vz / p | spread T / k / ε | peak Vx / Vy / Vz / p | peak T / k / ε |
|---|---|---|---|---|---|
| 1e-4 / 0.9999 | 0.02 % | 0.144 / 0.011 / 0.006 / 0.001 | 0.24 / 0.33 / 0.40 | 2.9 / 9.0 / 14.6 / 19.5 | 6.4 / 1.9 / 2.2 |
| 1e-3 / 0.999 | 0.2 % | 0.253 / 0.024 / 0.015 / 0.002 | 0.33 / 0.40 / 0.46 | 5.6 / 18.5 / 35.2 / 40.9 | 9.2 / 2.5 / 2.5 |
| 1e-2 / 0.99 | 1.9 % | 0.478 / 0.066 / 0.052 / 0.010 | 0.58 / 0.57 / 0.56 | 11.2 / 51.5 / 125 / 211 | 17.2 / 3.7 / 3.0 |

[`results/normalization_stats.csv`](results/normalization_stats.csv) holds q0.001 / q0.999 of the curated designs:

| channel | transform | low | high |
|---|---|---|---|
| Vx | linear | -3.87 m/s | 26.2 m/s |
| Vy | linear | -5.29 m/s | 5.21 m/s |
| Vz | linear | -2.11 m/s | 3.51 m/s |
| p | linear | -143 Pa | 1202 Pa |
| T | linear | 293.15 K | 329.95 K |
| k | log10 | 7.3e-6 m²/s² | 29.0 m²/s² |
| ε | log10 | 1.7e-4 m²/s³ | 4.7e4 m²/s³ |

Compared with the 10 % sweep, the bounds move by up to 8 % of their range: T high 332.8 → 329.9 K (8 %), k low 1.4e-5 → 7.3e-6 m²/s² (5 % in log10), p high 1175 → 1202 Pa (2 %).

## Conclusion

- Exclude the 310 flagged designs as physically implausible, likely diverged simulations. The 10 % sweep caught only the 26 in its share, the full-dataset fine-tuning trained on the other 284 and its train MSE spiked by up to 10³.
- Keep k and ε on a log scale and normalize every channel by q0.001 / q0.999 of the curated designs, unclipped: the trade-off of the 10 % sweep holds on the full dataset, peaks stay at 41 and below.
- Training on every host uses this normalization and exclusion list (`PATH_VALUE_DISTRIBUTION` in `src/coolfrog/experiments/autoencoding.py`), the 10 % sweep stays as a record only.
