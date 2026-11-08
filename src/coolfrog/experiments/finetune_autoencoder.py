"""LoRA-32 fine-tuning of Tadpole-B on whole curated domains at 2x, compared to zero-shot on the same validation designs.

`--resume` continues an unfinished run. On several GPUs: `torchrun --standalone --nproc_per_node <n_gpus> -m <this module>`.
One `Stage` per latent scale, `results/latent_scale_<n>/`:

- `metrics_training.parquet`, `metrics_validation.parquet` (Hugging Face): every train sample of every update and every
  validation design, each one channel, see `TRAINING_SCHEMA` and `VALIDATION_SCHEMA`
- `metrics_training_aggregates.parquet`, `metrics_validation_aggregates.parquet` (Hugging Face): their `aggregate` per update,
  for validation also per channel, the curves and README tables read these
- `state.json`: title and configs, progress of the last save, all tables rewritten while training runs
- `per_design.parquet` (Hugging Face): the final validation of `metrics_validation.parquet` in the layout of `per_design_frame`
- `adapter.pt` (Hugging Face): LoRA weights
- `checkpoints/adapter_update_<update>.pt` (Hugging Face): LoRA weights after every validation
- `training_curves.png`, `optimizer_curves.png` (learning rate, gradient norms), `validation_curves.png`,
  `validation_error.png` (zero-shot vs. final mean and median NRMSE per channel), `slices_first_validation_design.png`,
  `config.yaml`, `stage.json`

README tables: `results/nrmse.csv`, `results/mse.csv`, `results/spectral-loss.csv`, `results/training.csv`.
"""

from dataclasses import replace
import logging
import sys

import matplotlib as mpl
import polars as pl
import torch

from coolfrog.engine.cooldata import CoolMetadata, create_dataloader
from coolfrog.engine.model_evaluation import KL, MSE, NRMSE, SPECTRAL_LOSS
from coolfrog.engine.tadpole import Tadpole
from coolfrog.engine.training import RandomCrops, fine_tune_autoencoder
from coolfrog.engine.training_metrics import SCHEMAS
from coolfrog.experiments.autoencoding import (
    curated_metadata,
    data_config,
    load_stats,
    per_design_frame,
    quantile_table,
    train_validation_split,
    validation_loader,
)
from coolfrog.lib.config import TadpoleConfig, TrainConfig
from coolfrog.lib.datatypes import NormalizationStats
from coolfrog.lib.experiment import Stage, run_experiment, write_configs
from coolfrog.lib.huggingface import HuggingFaceHub
from coolfrog.lib.metrics import configs_state
from coolfrog.lib.plots import (
    CHANNEL_LABELS,
    METRICS_TRAINING_AGGREGATES,
    METRICS_VALIDATION,
    METRICS_VALIDATION_AGGREGATES,
    STYLE,
    figure_title,
    optimizer_curves,
    plot_slices,
    training_curves,
    validation_curves,
    validation_error_comparison,
)

TITLE = "Autoencoder fine-tuning with LoRA"
GOAL = (
    "Fine-tune Tadpole-B with rank-32 LoRA on whole single-channel domains, no crops or tiles, at 800x160x32, and compare "
    "it to zero-shot on the same validation designs."
)
HYPOTHESIS = "LoRA-32 lowers the validation NRMSE of every channel below zero-shot."

LORA_RANK = 32
MODEL = f"LoRA-{LORA_RANK}"
DEVICE = "cuda"
RESUME = "--resume" in sys.argv  # continue an unfinished run, complete latent scale stages are loaded, not rerun
LATENT_SCALES = (2,)
BATCH_SIZES = {1: 4, 2: 2}  # designs per pass and GPU that fit 11 GB (1080 Ti), at scale 2 only with activation checkpointing
log = logging.getLogger("coolfrog.experiment")

MODEL_CONFIG = TadpoleConfig(
    size="B",
    path_weights=None,
    encoder_state=LORA_RANK,
    decoder_state=LORA_RANK,
    encoder_crop_size=None,  # whole domains in training and validation, no tiles
    max_internal_batchsize=1,  # one scalar field per encoder/decoder call
)
TRAIN_CONFIG = TrainConfig(
    epochs=10,
    designs_per_update=32,
    learning_rate=5e-5,
    warmup_updates=1000,
    weight_decay=1e-15,
    kl_weight=1e-6,
    validate_every=200,
)


def training_row(aggregates: pl.DataFrame, latent_scale: int) -> dict[str, float]:
    """Row of the README training table from the training aggregates, final values are means over the last update's samples."""
    last = aggregates.sort("update").row(-1, named=True)
    return {
        "latent_scale": latent_scale,
        "updates": last["update"],  # resolved from `TRAIN_CONFIG.epochs` when set
        "training time [h]": aggregates["seconds_per_update"].sum() / 3600,  # optimizer time, validation excluded
        "final train MSE": last[f"{MSE}/mean"],
        "final train spectral loss": last[f"{SPECTRAL_LOSS}/mean"],
        "final KL": last[f"{KL}/mean"],
    }


def zero_shot_medians(aggregates: pl.DataFrame, latent_scale: int) -> pl.DataFrame:
    """Validation medians over designs at update 0, where LoRA is still the identity, in the layout of `quantile_table`."""
    medians = (
        aggregates.filter(pl.col("update") == 0)
        .select("channel", *(pl.col(f"{metric}/median").alias(metric) for metric in (NRMSE, MSE, SPECTRAL_LOSS)))
        .unpivot(index="channel", variable_name="metric")
        .with_columns(pl.col("channel").cast(pl.String).replace_strict(CHANNEL_LABELS))
        .pivot(on="channel", index="metric", values="value")
    )
    return medians.select(
        pl.lit(latent_scale).alias("latent_scale"),
        pl.lit("zero-shot").alias("model"),
        "metric",
        pl.lit("median").alias("statistic"),
        *CHANNEL_LABELS.values(),
    )


def final_per_design(validation: pl.DataFrame, latent_scale: int, model: str = MODEL) -> pl.DataFrame:
    """The last validation of the raw validation rows, after the final update, in the layout of `per_design_frame`."""
    final = validation.filter(pl.col("update") == validation["update"].max())
    final = final.with_columns(pl.col("channel").cast(pl.String).replace_strict(CHANNEL_LABELS))
    tables = {
        metric: final.pivot(on="channel", index="design_id", values=metric).select("design_id", *CHANNEL_LABELS.values())
        for metric in (NRMSE, MSE, SPECTRAL_LOSS)
    }
    return per_design_frame(tables, latent_scale, model)


def fine_tune(  # noqa: PLR0913, resolution, stage, data, stats and the training variant
    latent_scale: int,
    stage: Stage,
    metadata: CoolMetadata,
    stats: NormalizationStats,
    train_config: TrainConfig = TRAIN_CONFIG,
    model_label: str = MODEL,
) -> dict[str, object]:
    """Fine-tune and evaluate one resolution into `stage`, return the outputs of the stage besides its tracker tables.

    - `model_label` names the trained model in titles and `per_design.parquet`, e.g. "LoRA-32 + spectral"
    """
    config = replace(data_config(latent_scale), batch_size=BATCH_SIZES[latent_scale])
    stage.folder.mkdir(parents=True, exist_ok=True)
    write_configs([config, MODEL_CONFIG, train_config], stage.folder / "config.yaml")

    train, validation = train_validation_split(metadata, config)
    log.info("%d train and %d validation designs on grid %s", len(train), len(validation), config.grid.shape)
    whole_domains = RandomCrops(stats, MODEL_CONFIG.encoder_crop_size)
    train_loader = create_dataloader(train, config, shuffle=True, collate_fn=whole_domains)
    loader = validation_loader(validation, config)

    model = Tadpole(MODEL_CONFIG).to(DEVICE)
    title = f"{model_label} · latent scale {latent_scale} · {'x'.join(map(str, config.grid.shape))}"
    tracker = stage.track({"title": title, "configs": configs_state([config, MODEL_CONFIG, train_config])}, SCHEMAS)
    fine_tune_autoencoder(model, train_loader, loader, stats, train_config, stage.folder / "checkpoints", tracker)
    model.eval()
    with torch.no_grad():
        reference = next(iter(loader)).to(DEVICE).normalize(stats)
        prediction = model.decode(model.encode(reference))
    with mpl.rc_context(STYLE):
        curves = {
            "training_curves.png": training_curves(tracker.frame(METRICS_TRAINING_AGGREGATES), train_config.designs_per_update),
            "optimizer_curves.png": optimizer_curves(tracker.frame(METRICS_TRAINING_AGGREGATES)),
            "validation_curves.png": validation_curves(tracker.frame(METRICS_VALIDATION_AGGREGATES)),
            "validation_error.png": validation_error_comparison(
                tracker.frame(METRICS_VALIDATION_AGGREGATES), model_label, config.grid.shape
            ),
        }
        titles = {
            "training_curves.png": "Training metrics",
            "optimizer_curves.png": "Optimizer",
            "validation_curves.png": "Validation metrics",
        }
        for name, title in titles.items():
            figure_title(curves[name], f"{title} · {model_label} on {config.voxel_resolution}")
        slices = plot_slices(reference, prediction, config.grid)
    return {
        "per_design.parquet": final_per_design(tracker.frame(METRICS_VALIDATION), latent_scale, model_label),
        "adapter.pt": model.adapter(),
        **curves,
        "slices_first_validation_design.png": slices,
    }


if __name__ == "__main__":
    with run_experiment(TITLE, GOAL, HYPOTHESIS, [MODEL_CONFIG, TRAIN_CONFIG], hub=HuggingFaceHub(), resume=RESUME) as run:
        stats = load_stats()
        metadata = curated_metadata()

        per_design_tables, zero_shot_tables, training_rows = [], [], []
        for latent_scale in LATENT_SCALES:
            stage = run.stage(f"latent_scale_{latent_scale}")
            if stage.is_complete():
                log.info("Latent scale %d: stage complete, loading its results from %s", latent_scale, stage.folder)
            else:
                log.info("Latent scale %d: training", latent_scale)
                stage.save(fine_tune(latent_scale, stage, metadata, stats))
                torch.cuda.empty_cache()
            per_design_tables.append(stage.load("per_design.parquet"))
            zero_shot_tables.append(zero_shot_medians(stage.load(f"{METRICS_VALIDATION_AGGREGATES}.parquet"), latent_scale))
            training_rows.append(training_row(stage.load(f"{METRICS_TRAINING_AGGREGATES}.parquet"), latent_scale))

        per_design = pl.concat(per_design_tables)
        zero_shot = pl.concat(zero_shot_tables)
        for metric, name in ((NRMSE, "nrmse"), (MSE, "mse"), (SPECTRAL_LOSS, "spectral-loss")):
            fine_tuned = quantile_table(per_design, metric, (0.5,))
            table = pl.concat([zero_shot.filter(pl.col("metric") == metric).drop("metric"), fine_tuned], how="vertical_relaxed").sort(
                "latent_scale", maintain_order=True
            )
            log.info("%s\n%s", name, table)
            run.publish(name, table.drop("statistic"))
        run.publish("training", pl.DataFrame(training_rows).sort("latent_scale"))
