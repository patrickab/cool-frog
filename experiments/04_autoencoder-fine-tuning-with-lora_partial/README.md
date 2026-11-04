# Autoencoder fine-tuning with LoRA

## Goal

Fine-tune Tadpole-B with rank-32 LoRA on crops spanning the full channel height at 800x160x32 (32³, unlike the 64³ pre-training crops) and 1600x320x64 (64³), and compare both to zero-shot on the same validation designs.

## Hypothesis

LoRA-32 lowers the validation NRMSE of every channel below zero-shot at both resolutions. Zero-shot is worse on 32³ tiles, as Tadpole never saw that crop size, and fine-tuning closes most of that gap.

## Result

Median over the 128 validation designs. NRMSE in %, the column unit names the channel. Training curves and slices of the first validation design: [`results/latent_scale_2/`](results/latent_scale_2/), [`results/latent_scale_4/`](results/latent_scale_4/).

<!-- table: nrmse -->
| latent_scale | model     | Vx [m/s] | Vy [m/s] | Vz [m/s] | p [Pa] | T [K] | k [m²/s²] | ε [m²/s³] |
|--------------|-----------|----------|----------|----------|--------|-------|-----------|-----------|
| 2            | zero-shot | 19.6     | 17.1     | 20.1     | 176    | 24    | 20.4      | 12.3      |
| 2            | LoRA-32   | 6.78     | 11.2     | 16.4     | 30.4   | 10.6  | 7.37      | 6.69      |
| 4            | zero-shot | 9.07     | 9.06     | 13.9     | 40.9   | 13    | 10.9      | 8.08      |
| 4            | LoRA-32   | 4.91     | 7.81     | 10.5     | 23     | 7.58  | 5.36      | 4.64      |
<!-- /table -->

<!-- table: spectral-loss -->
| latent_scale | model     | Vx [m/s] | Vy [m/s] | Vz [m/s] | p [Pa] | T [K] | k [m²/s²] | ε [m²/s³] |
|--------------|-----------|----------|----------|----------|--------|-------|-----------|-----------|
| 2            | zero-shot | 0.29     | 0.527    | 0.521    | 0.061  | 0.574 | 0.786     | 0.589     |
| 2            | LoRA-32   | 0.276    | 0.505    | 0.484    | 0.061  | 0.513 | 0.675     | 0.548     |
| 4            | zero-shot | 0.421    | 0.721    | 0.728    | 0.0559 | 0.758 | 0.971     | 0.773     |
| 4            | LoRA-32   | 0.408    | 0.675    | 0.661    | 0.0715 | 0.722 | 0.837     | 0.718     |
<!-- /table -->

<!-- table: training -->
| latent_scale | updates | training time [h] | final train MSE | final train spectral loss | final KL |
|--------------|---------|-------------------|-----------------|---------------------------|----------|
| 2            | 14000   | 1.44              | 0.000353        | 0.215                     | 16.1     |
| 4            | 14000   | 9.35              | 0.000132        | 0.337                     | 13.2     |
<!-- /table -->

## Conclusion

- Hypothesis holds for NRMSE: LoRA-32 beats zero-shot on every channel, by 18 to 83 % at 32³ and 14 to 51 % at 64³
- Zero-shot is worse on 32³ for every channel. Fine-tuning closes 52 to 95 % of that gap, except Vz (5 %, 16.4 vs 10.5)
- Spectral loss improves by at most 14 %. p is flat at 32³ and worse at 64³ (0.0715 vs 0.0559)
- 64³ trains 6.5x longer than 32³ for the same 14000 updates (9.35 h vs 1.44 h)

