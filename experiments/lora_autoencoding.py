"""LoRA-32 fine-tuning of Tadpole-B on the curated designs at 2x, then 4x, compared to zero-shot on the same validation designs.

Run from the repo root. Results:

- `results/latent_scale_<n>/metrics.parquet` (Hugging Face): train MSE, spectral loss and KL per update, validation medians
- `results/latent_scale_<n>/adapter.pt` (Hugging Face): LoRA weights
- `results/latent_scale_<n>/training_curves.png`, `slices_first_validation_design.png`
- `results/per_design.parquet` (Hugging Face): final NRMSE and spectral loss per latent scale, design and channel
- `results/nrmse.csv`, `results/spectral-loss.csv`, `results/training.csv` (README tables)
"""

import time

from autoencoding import (
    LATENT_SCALES,
    TRAIN_CONFIG,
    curated_metadata,
    data_config,
    load_stats,
    per_design_frame,
    quantile_table,
    tadpole_config,
    train_validation_split,
    validation_loader,
)
from matplotlib.figure import Figure
import polars as pl
import torch

from coolfrog.engine.cooldata import create_dataloader
from coolfrog.engine.model_evaluation import evaluate
from coolfrog.engine.tadpole import Tadpole
from coolfrog.engine.training import RandomCrops, fine_tune_autoencoder
from coolfrog.engine.training_metrics import KL, MSE, NRMSE, SPECTRAL, SPECTRAL_LOSS, validation_name
from coolfrog.lib.datatypes import CHANNELS
from coolfrog.lib.experiment import inject_table, run_experiment, write_configs
from coolfrog.lib.huggingface import HuggingFaceHub
from coolfrog.lib.log import get_logger, setup_logging
from coolfrog.lib.plot import plot_slices

TITLE = "LoRA autoencoding"
GOAL = (
    "Fine-tune Tadpole-B with rank-32 LoRA on crops spanning the full channel height at 800x160x32 (32³, unlike the 64³ "
    "pre-training crops) and 1600x320x64 (64³), and compare both to zero-shot on the same validation designs."
)
HYPOTHESIS = (
    "LoRA-32 lowers the validation NRMSE of every channel below zero-shot at both resolutions. "
    "Zero-shot is worse on 32³ tiles, as Tadpole never saw that crop size, and fine-tuning closes most of that gap."
)

LORA_RANK = 32
MODEL = f"LoRA-{LORA_RANK}"
DEVICE = "cuda"


def plot_training_curves(metrics: pl.DataFrame, latent_scale: int) -> Figure:
    """Train MSE and spectral loss per update, validation medians of both metrics per channel and update."""
    figure = Figure(figsize=(14, 8), layout="constrained")
    axes = figure.subplots(2, 2)
    for axis, series, title in (
        (axes[0, 0], MSE, "Train MSE, normalized units"),
        (axes[0, 1], SPECTRAL, "Train spectral loss, crops"),
    ):
        train = metrics.select("update", series).drop_nulls()
        axis.plot(train["update"], train[series], color="#3b5b7a", linewidth=0.8)
        axis.set_title(title)
        axis.set_yscale("log")
    for axis, metric, title in (
        (axes[1, 0], NRMSE, "Validation NRMSE [%], median"),
        (axes[1, 1], SPECTRAL_LOSS, "Validation spectral loss, median"),
    ):
        for channel in CHANNELS:
            validation = metrics.select("update", validation_name(metric, channel.value)).drop_nulls()
            axis.plot(
                validation["update"], validation[validation_name(metric, channel.value)], marker="o", markersize=3, label=channel.value
            )
        axis.set_title(title)
        axis.set_yscale("log")
    axes[1, 0].legend(fontsize=8, ncols=2)
    for axis in axes.flat:
        axis.set_xlabel("update")
        axis.grid(alpha=0.3)
    figure.suptitle(f"{MODEL} at latent scale {latent_scale}")
    return figure


def zero_shot_medians(metrics: pl.DataFrame, latent_scale: int) -> pl.DataFrame:
    """Validation medians at update 0, where LoRA is still the identity, in the layout of `quantile_table`."""
    update_zero = metrics.filter(pl.col("update") == 0)
    return pl.concat(
        pl.DataFrame(
            {
                "latent_scale": [latent_scale],
                "model": ["zero-shot"],
                "metric": [metric],
                "statistic": ["median"],
                **{channel.value: update_zero[validation_name(metric, channel.value)] for channel in CHANNELS},
            }
        )
        for metric in (NRMSE, SPECTRAL_LOSS)
    )


def fine_tune(latent_scale: int) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float]]:
    """Fine-tune and evaluate one resolution into `results/latent_scale_<n>/`.

    Returns the final per-design table, the zero-shot medians and a row of the training table.
    """
    config = data_config(latent_scale)
    model_config = tadpole_config(latent_scale, LORA_RANK)
    path_run = path_results / f"latent_scale_{latent_scale}"
    path_run.mkdir(exist_ok=True)
    write_configs([config, model_config], path_run / "config.yaml")

    train, validation = train_validation_split(metadata, config)
    log.info("%d train and %d validation designs on grid %s", len(train), len(validation), config.grid.shape)
    crops = RandomCrops(stats, model_config.encoder_crop_size, TRAIN_CONFIG.crops_per_step)
    train_loader = create_dataloader(train, config, shuffle=True, collate_fn=crops)
    loader = validation_loader(validation, config)

    model = Tadpole(model_config).to(DEVICE)
    start = time.perf_counter()
    metrics = fine_tune_autoencoder(model, train_loader, loader, stats, TRAIN_CONFIG).to_frame()
    hours = (time.perf_counter() - start) / 3600
    metrics.write_parquet(path_run / "metrics.parquet")
    model.save_adapter(path_run / "adapter.pt")
    plot_training_curves(metrics, latent_scale).savefig(path_run / "training_curves.png", dpi=150)

    per_design = per_design_frame(evaluate(model, loader, stats), latent_scale, MODEL)
    model.eval()
    with torch.no_grad():
        reference = next(iter(loader)).to(DEVICE).normalize(stats)
        prediction = model.decode(model.encode(reference))
    plot_slices(reference, prediction, config.grid).savefig(path_run / "slices_first_validation_design.png", dpi=150)

    last = metrics.select(MSE, SPECTRAL, KL).drop_nulls().tail(1).row(0, named=True)
    training = {
        "latent_scale": latent_scale,
        "updates": TRAIN_CONFIG.updates,
        "duration [h]": hours,
        "final train MSE": last[MSE],
        "final train spectral loss": last[SPECTRAL],
        "final KL": last[KL],
    }
    return per_design, zero_shot_medians(metrics, latent_scale), training


with run_experiment(TITLE, GOAL, HYPOTHESIS, [TRAIN_CONFIG], hub=HuggingFaceHub()) as path_results:
    setup_logging(log_file=path_results / "run.log")
    log = get_logger("coolfrog.experiment")
    stats = load_stats()
    metadata = curated_metadata()

    per_design_tables, zero_shot_tables, training_rows = [], [], []
    for latent_scale in LATENT_SCALES:
        per_design, zero_shot, training = fine_tune(latent_scale)
        per_design_tables.append(per_design)
        zero_shot_tables.append(zero_shot)
        training_rows.append(training)
        torch.cuda.empty_cache()

    per_design = pl.concat(per_design_tables)
    per_design.write_parquet(path_results / "per_design.parquet")
    zero_shot = pl.concat(zero_shot_tables)
    for metric, name in ((NRMSE, "nrmse"), (SPECTRAL_LOSS, "spectral-loss")):
        fine_tuned = quantile_table(per_design, metric, (0.5,))
        table = pl.concat([zero_shot.filter(pl.col("metric") == metric).drop("metric"), fine_tuned], how="vertical_relaxed").sort(
            "latent_scale", maintain_order=True
        )
        log.info("%s\n%s", name, table)
        inject_table(path_results, name, table.drop("statistic"))
    inject_table(path_results, "training", pl.DataFrame(training_rows))
