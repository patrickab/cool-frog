# Zero-shot Tadpole reconstruction on CoolData

## Goal

Measure NRMSE and spectral error of the frozen, pretrained Tadpole-B autoencoder on every curated CoolData design, normalized with the quantile stats of experiment 02.

## Result

5666 curated designs at 400x80x16, whole domain encoded at once, q0.001 / q0.999 normalization of experiment 02, 57 min.

- [`results/nrmse_per_design.csv`](results/nrmse_per_design.csv), [`results/nrmse_summary.csv`](results/nrmse_summary.csv): fluid RMSE / per-design fluid std, in %
- [`results/spectral_loss_per_design.csv`](results/spectral_loss_per_design.csv), [`results/spectral_loss_summary.csv`](results/spectral_loss_summary.csv): L1 of log1p FFT amplitudes, normalized units
- [`results/slices_first_design.png`](results/slices_first_design.png): reference, prediction and error of design 3

| NRMSE [%] | Vx | Vy | Vz | p | T | k | ε |
|---|---|---|---|---|---|---|---|
| q0.1 | 25.3 | 19.3 | 22.4 | 43 | 17.5 | 21.0 | 13.7 |
| median | 33.2 | 23.5 | 31.3 | 383 | 31.3 | 28.8 | 16.9 |
| q0.9 | 64.5 | 31.3 | 76.7 | 3822 | 66.7 | 38.1 | 21.5 |
| designs > 100 % | 2.2 % | 0.1 % | 5.6 % | 76.8 % | 4.2 % | 0 % | 0 % |
| spectral loss, median | 0.22 | 0.39 | 0.38 | 0.06 | 0.41 | 0.67 | 0.49 |

p is offset, not noisy: its RMSE is 23 to 24 Pa in 80 % of designs (q0.1 to q0.9), 0.036 in normalized units, while the per-design p std spans 0.6 to 55 Pa over the same quantiles. NRMSE of p therefore tracks 1 / std almost exactly (Spearman -0.999). The slice plot shows the same: the predicted p is a near-uniform +20 to 30 Pa above the reference across the domain.

## Conclusion

- Zero-shot Tadpole-B reproduces the flow structure of every channel except p, with median NRMSE of 17 to 33 %, far from a usable autoencoder.
- p fails through a constant offset of ~0.036 normalized units. That error is small in the normalized space the loss sees, but large next to the few-Pa variation of most designs, so p NRMSE (median 383 %) mostly measures that offset.
- Fine-tuning has to remove that offset. MSE in normalized units weighs it little (0.036² ≈ 1.3e-3 per voxel), so check p NRMSE explicitly during training rather than relying on the loss.
