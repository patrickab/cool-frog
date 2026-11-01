# Min-max normalization stats on CoolData

## Goal

Measure the global per-channel fluid min and max over every CoolData design at 1600x320x64, the shared normalization of all subsequent models at every resolution.

## Result

Global fluid min and max over all 5692 designs at 1600x320x64: [`results/normalization_stats.csv`](results/normalization_stats.csv), 56 min sweep. Visualized in [`notebooks/01_min_max_normalization_stats.ipynb`](../../notebooks/01_min_max_normalization_stats.ipynb).

| channel | min | max |
|---|---|---|
| Vx [m/s] | -131.9 | 533.6 |
| Vy [m/s] | -1744 | 1257 |
| Vz [m/s] | -424.4 | 2811 |
| p [Pa] | -4.39e4 | 5.18e7 |
| T [K] | 139.3 | 4068 |
| k [m²/s²] | 1e-10 | 9.25e4 |
| ε [m²/s³] | 1e-10 | 1.43e12 |

## Conclusion

- The extremes are physically implausible for a cooling flow: speeds up to 2811 m/s, T up to 4068 K. Might be diverged simulations.
- If these extremes are outliers, plain global min-max is unusable: they stretch the range so far that typical values collapse into a tiny share of [-1, 1] and the model would see near-constant fields.
- Depending on the distribution, a log transform may make sense: k and ε range from the 1e-10 floor to 9.25e4 and 1.43e12, more decades than a linear [-1, 1] scale can resolve.
- Next: analyze the full value distribution (histograms, quantiles, per-design stats) before choosing outlier handling and normalization.
