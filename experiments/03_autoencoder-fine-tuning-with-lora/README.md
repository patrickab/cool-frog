# Autoencoder fine-tuning with LoRA

## Goal

Fine-tune Tadpole-B with rank-32 LoRA on whole single-channel domains, no crops or tiles, at 800x160x32, and compare it to zero-shot on the same validation designs.

## Hypothesis

LoRA-32 lowers the validation NRMSE of every channel below zero-shot.

## Result


## Conclusion

<!-- table: nrmse -->
| latent_scale | model     | Vx [m/s] | Vy [m/s] | Vz [m/s] | p [Pa] | T [K] | k [m²/s²] | ε [m²/s³] |
|--------------|-----------|----------|----------|----------|--------|-------|-----------|-----------|
| 1            | zero-shot | 34.5     | 24.4     | 29.4     | 323    | 29.4  | 27.3      | 16.9      |
| 1            | LoRA-32   | 6.67     | 11.4     | 16.6     | 15.4   | 11.3  | 7.24      | 6.42      |
<!-- /table -->

<!-- table: mse -->
| latent_scale | model     | Vx [m/s] | Vy [m/s] | Vz [m/s] | p [Pa]   | T [K]    | k [m²/s²] | ε [m²/s³] |
|--------------|-----------|----------|----------|----------|----------|----------|-----------|-----------|
| 1            | zero-shot | 0.00175  | 0.000254 | 0.00034  | 0.0013   | 0.0023   | 0.00572   | 0.00229   |
| 1            | LoRA-32   | 6.06e-05 | 5.33e-05 | 0.000117 | 3.01e-06 | 0.000348 | 0.000419  | 0.00034   |
<!-- /table -->

<!-- table: spectral-loss -->
| latent_scale | model     | Vx [m/s] | Vy [m/s] | Vz [m/s] | p [Pa] | T [K] | k [m²/s²] | ε [m²/s³] |
|--------------|-----------|----------|----------|----------|--------|-------|-----------|-----------|
| 1            | zero-shot | 0.219    | 0.396    | 0.396    | 0.0627 | 0.441 | 0.658     | 0.51      |
| 1            | LoRA-32   | 0.184    | 0.371    | 0.35     | 0.0379 | 0.388 | 0.528     | 0.445     |
<!-- /table -->

<!-- table: training -->
| latent_scale | updates | training time [h] | final train MSE | final train spectral loss | final KL |
|--------------|---------|-------------------|-----------------|---------------------------|----------|
| 1            | 17650   | 17.5              | 0.00029         | 0.352                     | 10.5     |
<!-- /table -->
