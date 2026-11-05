"""LoRA-32 fine-tuning of Tadpole-B on whole curated domains at 2x, then 1x, compared to zero-shot on the same validation designs.

`--resume` continues an unfinished run. One `Stage` per latent scale, `results/latent_scale_<n>/`:

- `metrics.parquet` (Hugging Face): train MSE, spectral loss and KL per update, validation medians
- `per_design.parquet` (Hugging Face): final NRMSE and spectral loss per design and channel
- `adapter.pt` (Hugging Face): LoRA weights
- `checkpoints/adapter_update_<update>.pt` (Hugging Face): LoRA weights after every validation
- `training_curves.png`, `slices_first_validation_design.png`, `config.yaml`, `stage.json`

README tables: `results/nrmse.csv`, `results/spectral-loss.csv`, `results/training.csv`.
"""

from pathlib import Path
import sys

from matplotlib.figure import Figure
import polars as pl
import torch

from coolfrog.engine.cooldata import create_dataloader
from coolfrog.engine.model_evaluation import evaluate
from coolfrog.engine.tadpole import Tadpole
from coolfrog.engine.training import RandomCrops, fine_tune_autoencoder
from coolfrog.engine.training_metrics import KL, MSE, NRMSE, SECONDS_PER_UPDATE, SPECTRAL, SPECTRAL_LOSS, validation_name
from coolfrog.experiments.autoencoding import (
    N_VALIDATION_DESIGNS,
    curated_metadata,
    load_stats,
    per_design_frame,
    quantile_table,
    train_validation_split,
    validation_loader,
)
from coolfrog.lib.config import DataConfig, Grid, TadpoleConfig, TrainConfig
from coolfrog.lib.datatypes import CHANNELS
from coolfrog.lib.experiment import Stage, inject_table, run_experiment, write_configs
from coolfrog.lib.huggingface import HuggingFaceHub
from coolfrog.lib.log import get_logger
from coolfrog.lib.plot import plot_slices

TITLE = "Autoencoder fine-tuning with LoRA"
GOAL = (
    "Fine-tune Tadpole-B with rank-32 LoRA on whole single-channel domains, no crops or tiles, at 400x80x16 and "
    "800x160x32, and compare both to zero-shot on the same validation designs."
)
HYPOTHESIS = "LoRA-32 lowers the validation NRMSE of every channel below zero-shot at both resolutions."

LORA_RANK = 32
MODEL = f"LoRA-{LORA_RANK}"
DEVICE = "cuda"
RESUME = "--resume" in sys.argv  # continue an unfinished run, complete latent scale stages are loaded, not rerun
LATENT_SCALES = (2, 1)  # 2x first, its voxel cache exists, 1x waits on `build.lock` until its cache is built

MODEL_CONFIG = TadpoleConfig(
    size="B",
    path_weights=None,
    encoder_state=LORA_RANK,
    decoder_state=LORA_RANK,
    encoder_crop_size=None,  # whole domains in training and validation, no tiles
    max_internal_batchsize=1,  # one scalar field per encoder/decoder call, whole domains fill the 12 GB
    activation_checkpointing=True,  # whole-domain backward only fits at latent scale 1 without it
)
TRAIN_CONFIG = TrainConfig(
    updates=50_000,
    crops_per_step=1,  # one whole single-channel domain per pass
    accumulation_steps=4,
    learning_rate=5e-5,
    warmup_updates=1000,
    weight_decay=1e-15,
    kl_weight=1e-6,
    validate_every=1000,
    n_validation_designs=N_VALIDATION_DESIGNS,
)


def data_config(latent_scale: int) -> DataConfig:
    """Grid of `latent_scale`, one design per training pass, as each pass uses a single domain."""
    return DataConfig(grid=Grid(latent_scale=latent_scale), source=None, cache=True, batch_size=1, num_workers=6)


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


def plot_training_curves(metrics: pl.DataFrame, latent_scale: int) -> Figure:
    """Train MSE and spectral loss per update, validation medians of both metrics per channel and update."""
    figure = Figure(figsize=(14, 8), layout="constrained")
    axes = figure.subplots(2, 2)
    for axis, series, title in (
        (axes[0, 0], MSE, "Train MSE, normalized units"),
        (axes[0, 1], SPECTRAL, "Train spectral loss, whole domains"),
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


def fine_tune(latent_scale: int, path_stage: Path) -> dict[str, object]:
    """Fine-tune and evaluate one resolution, return the outputs of its stage (file name -> data)."""
    config = data_config(latent_scale)
    path_stage.mkdir(exist_ok=True)
    write_configs([config, MODEL_CONFIG], path_stage / "config.yaml")

    train, validation = train_validation_split(metadata, config)
    log.info("%d train and %d validation designs on grid %s", len(train), len(validation), config.grid.shape)
    whole_domains = RandomCrops(stats, MODEL_CONFIG.encoder_crop_size, TRAIN_CONFIG.crops_per_step)
    train_loader = create_dataloader(train, config, shuffle=True, collate_fn=whole_domains)
    loader = validation_loader(validation, config)

    model = Tadpole(MODEL_CONFIG).to(DEVICE)
    title = f"{MODEL} · latent scale {latent_scale} · {'x'.join(map(str, config.grid.shape))}"
    metrics = fine_tune_autoencoder(model, train_loader, loader, stats, TRAIN_CONFIG, path_stage / "checkpoints", title).to_frame()
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


with run_experiment(TITLE, GOAL, HYPOTHESIS, [MODEL_CONFIG, TRAIN_CONFIG], hub=HuggingFaceHub(), resume=RESUME) as path_results:
    log = get_logger("coolfrog.experiment")
    stats = load_stats()
    metadata = curated_metadata()

    per_design_tables, zero_shot_tables, training_rows = [], [], []
    for latent_scale in LATENT_SCALES:
        stage = Stage(path_results / f"latent_scale_{latent_scale}")
        if stage.is_complete():
            log.info("Latent scale %d: stage complete, loading its results from %s", latent_scale, stage.folder)
        else:
            log.info("Latent scale %d: training", latent_scale)
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
    inject_table(path_results, "training", pl.DataFrame(training_rows).sort("latent_scale"))
