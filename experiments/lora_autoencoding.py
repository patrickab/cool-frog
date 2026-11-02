"""LoRA-32 fine-tuning of Tadpole-B on the curated designs at 2x, then 4x, compared to zero-shot on the same validation designs.

Run from the repo root, `--resume` continues an unfinished run. One `Stage` per latent scale, `results/latent_scale_<n>/`:

- `metrics.parquet` (Hugging Face): train MSE, spectral loss and KL per update, validation medians
- `per_design.parquet` (Hugging Face): final NRMSE and spectral loss per design and channel
- `adapter.pt` (Hugging Face): LoRA weights
- `training_curves.png`, `slices_first_validation_design.png`, `config.yaml`, `stage.json`

README tables: `results/nrmse.csv`, `results/spectral-loss.csv`, `results/training.csv`.
"""

from pathlib import Path
import sys

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
from coolfrog.engine.training_metrics import KL, MSE, NRMSE, SECONDS_PER_UPDATE, SPECTRAL, SPECTRAL_LOSS, validation_name
from coolfrog.lib.datatypes import CHANNELS
from coolfrog.lib.experiment import Stage, inject_table, run_experiment, write_configs
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
RESUME = "--resume" in sys.argv  # continue an unfinished run, complete latent scale stages are loaded, not rerun


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


def training_hours(metrics: pl.DataFrame) -> float:
    """Optimizer time without validation: mean seconds per update of each logged window times its updates."""
    windows = (
        metrics.select("update", SECONDS_PER_UPDATE).drop_nulls().with_columns(pl.col("update").diff().fill_null(pl.col("update")))
    )
    return (windows["update"] * windows[SECONDS_PER_UPDATE]).sum() / 3600


def training_row(metrics: pl.DataFrame, latent_scale: int) -> dict[str, float]:
    """Row of the README training table."""
    last = metrics.select(MSE, SPECTRAL, KL).drop_nulls().tail(1).row(0, named=True)
    return {
        "latent_scale": latent_scale,
        "updates": TRAIN_CONFIG.updates,
        "training time [h]": training_hours(metrics),
        "final train MSE": last[MSE],
        "final train spectral loss": last[SPECTRAL],
        "final KL": last[KL],
    }


def fine_tune(latent_scale: int, path_stage: Path) -> dict[str, object]:
    """Fine-tune and evaluate one resolution, return the outputs of its stage (file name -> data)."""
    config = data_config(latent_scale)
    model_config = tadpole_config(latent_scale, LORA_RANK)
    path_stage.mkdir(exist_ok=True)
    write_configs([config, model_config], path_stage / "config.yaml")

    train, validation = train_validation_split(metadata, config)
    log.info("%d train and %d validation designs on grid %s", len(train), len(validation), config.grid.shape)
    crops = RandomCrops(stats, model_config.encoder_crop_size, TRAIN_CONFIG.crops_per_step)
    train_loader = create_dataloader(train, config, shuffle=True, collate_fn=crops)
    loader = validation_loader(validation, config)

    model = Tadpole(model_config).to(DEVICE)
    metrics = fine_tune_autoencoder(model, train_loader, loader, stats, TRAIN_CONFIG).to_frame()
    per_design = per_design_frame(evaluate(model, loader, stats), latent_scale, MODEL)
    model.eval()
    with torch.no_grad():
        reference = next(iter(loader)).to(DEVICE).normalize(stats)
        prediction = model.decode(model.encode(reference))
    return {
        "metrics.parquet": metrics,
        "per_design.parquet": per_design,
        "adapter.pt": model.adapter(),
        "training_curves.png": plot_training_curves(metrics, latent_scale),
        "slices_first_validation_design.png": plot_slices(reference, prediction, config.grid),
    }


with run_experiment(TITLE, GOAL, HYPOTHESIS, [TRAIN_CONFIG], hub=HuggingFaceHub(), resume=RESUME) as path_results:
    setup_logging(log_file=path_results / "run.log")
    log = get_logger("coolfrog.experiment")
    stats = load_stats()
    metadata = curated_metadata()

    per_design_tables, zero_shot_tables, training_rows = [], [], []
    for latent_scale in LATENT_SCALES:
        stage = Stage(path_results / f"latent_scale_{latent_scale}")
        if stage.is_complete():
            log.info("latent scale %d complete in %s, loading its results", latent_scale, stage.folder)
        else:
            stage.save(fine_tune(latent_scale, stage.folder))
            torch.cuda.empty_cache()
        metrics = stage.load("metrics.parquet")
        per_design_tables.append(stage.load("per_design.parquet"))
        zero_shot_tables.append(zero_shot_medians(metrics, latent_scale))
        training_rows.append(training_row(metrics, latent_scale))

    per_design = pl.concat(per_design_tables)
    zero_shot = pl.concat(zero_shot_tables)
    for metric, name in ((NRMSE, "nrmse"), (SPECTRAL_LOSS, "spectral-loss")):
        fine_tuned = quantile_table(per_design, metric, (0.5,))
        table = pl.concat([zero_shot.filter(pl.col("metric") == metric).drop("metric"), fine_tuned], how="vertical_relaxed").sort(
            "latent_scale", maintain_order=True
        )
        log.info("%s\n%s", name, table)
        inject_table(path_results, name, table.drop("statistic"))
    inject_table(path_results, "training", pl.DataFrame(training_rows))
