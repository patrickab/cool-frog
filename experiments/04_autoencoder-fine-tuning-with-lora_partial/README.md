# Autoencoder fine-tuning with LoRA

## Goal

Fine-tune Tadpole-B with rank-32 LoRA on crops spanning the full channel height at 800x160x32 (32³, unlike the 64³ pre-training crops) and 1600x320x64 (64³), and compare both to zero-shot on the same validation designs.

## Hypothesis

LoRA-32 lowers the validation NRMSE of every channel below zero-shot at both resolutions. Zero-shot is worse on 32³ tiles, as Tadpole never saw that crop size, and fine-tuning closes most of that gap.

## Result


## Conclusion
