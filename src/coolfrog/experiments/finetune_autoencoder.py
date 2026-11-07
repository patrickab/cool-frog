"""LoRA-32 fine-tuning of Tadpole-B on whole curated domains at 2x, compared to zero-shot on the same validation designs.

`--resume` continues an unfinished run. On several GPUs: `torchrun --standalone --nproc_per_node <n_gpus> -m <this module>`.
One `Stage` per latent scale, `results/latent_scale_<n>/`:

- `train.parquet` (Hugging Face): train MSE, spectral loss, KL, learning rate and seconds per update, mean over 10 updates
- `validation.parquet` (Hugging Face): NRMSE, MSE and spectral loss per validation update, design and channel
- `validation_medians.parquet` (Hugging Face): medians over designs of `validation.parquet` per validation update and channel
- `state.json`: title and configs, progress of the last save, all tables rewritten while training runs
- `per_design.parquet` (Hugging Face): the final validation of `validation.parquet` in the layout of `per_design_frame`
- `adapter.pt` (Hugging Face): LoRA weights
- `checkpoints/adapter_update_<update>.pt` (Hugging Face): LoRA weights after every validation
- `training_curves.png`, `validation_curves.png`, `slices_first_validation_design.png`, `config.yaml`, `stage.json`

README tables: `results/nrmse.csv`, `results/mse.csv`, `results/spectral-loss.csv`, `results/training.csv`.
"""

import logging
import sys

import matplotlib as mpl
import polars as pl
import torch

from coolfrog.engine.cooldata import CoolMetadata, create_dataloader
from coolfrog.engine.model_evaluation import MSE, NRMSE, SPECTRAL_LOSS
from coolfrog.engine.tadpole import Tadpole
from coolfrog.engine.training import RandomCrops, fine_tune_autoencoder
from coolfrog.engine.training_metrics import SCHEMAS, TRAIN, VALIDATION, VALIDATION_MEDIANS
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
from coolfrog.lib.plots import CHANNEL_LABELS, STYLE, plot_slices, training_curves, validation_curves

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
LATENT_SCALES = (1,)
log = logging.getLogger("coolfrog.experiment")

MODEL_CONFIG = TadpoleConfig(
    size="B",
    path_weights=None,
    encoder_state=LORA_RANK,
    decoder_state=LORA_RANK,
    encoder_crop_size=None,  # whole domains in training and validation, no tiles
    max_internal_batchsize=1,  # one scalar field per encoder/decoder call
    activation_checkpointing=False,  # latent scale 1 fits without it, saves the recompute in backward
)
TRAIN_CONFIG = TrainConfig(
    updates=10_000,
    designs_per_update=32,
    learning_rate=5e-5,
    warmup_updates=1000,
    weight_decay=1e-15,
    kl_weight=1e-6,
    validate_every=200,
)


def training_row(train: pl.DataFrame, latent_scale: int) -> dict[str, float]:
    """Row of the README training table."""
    last = train.tail(1).row(0, named=True)
    return {
        "latent_scale": latent_scale,
        "updates": TRAIN_CONFIG.updates,
        "training time [h]": training_hours(train),
        "final train MSE": last["mse"],
        "final train spectral loss": last["spectral_loss"],
        "final KL": last["kl"],
    }


def zero_shot_medians(medians: pl.DataFrame, latent_scale: int) -> pl.DataFrame:
    """Validation medians over designs at update 0, where LoRA is still the identity, in the layout of `quantile_table`."""
    medians = (
        medians.filter(pl.col("update") == 0)
        .unpivot(on=[NRMSE, MSE, SPECTRAL_LOSS], index="channel", variable_name="metric")
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


def training_hours(train: pl.DataFrame) -> float:
    """Optimizer time without validation: mean seconds per update of each logged window times its updates."""
    windows = train.select("update", "seconds_per_update").with_columns(pl.col("update").diff().fill_null(pl.col("update")))
    return (windows["update"] * windows["seconds_per_update"]).sum() / 3600


def final_per_design(validation: pl.DataFrame, latent_scale: int) -> pl.DataFrame:
    """The last validation, after the final update, in the layout of `per_design_frame`."""
    final = validation.filter(pl.col("update") == validation["update"].max())
    final = final.with_columns(pl.col("channel").cast(pl.String).replace_strict(CHANNEL_LABELS))
    tables = {
        metric: final.pivot(on="channel", index="design_id", values=metric).select("design_id", *CHANNEL_LABELS.values())
        for metric in (NRMSE, MSE, SPECTRAL_LOSS)
    }
    return per_design_frame(tables, latent_scale, MODEL)


def fine_tune(latent_scale: int, stage: Stage, metadata: CoolMetadata, stats: NormalizationStats) -> dict[str, object]:
    """Fine-tune and evaluate one resolution into `stage`, return the outputs of the stage besides its tracker tables."""
    config = data_config(latent_scale)
    stage.folder.mkdir(parents=True, exist_ok=True)
    write_configs([config, MODEL_CONFIG], stage.folder / "config.yaml")

    train, validation = train_validation_split(metadata, config)
    log.info("%d train and %d validation designs on grid %s", len(train), len(validation), config.grid.shape)
    whole_domains = RandomCrops(stats, MODEL_CONFIG.encoder_crop_size)
    train_loader = create_dataloader(train, config, shuffle=True, collate_fn=whole_domains)
    loader = validation_loader(validation, config)

    model = Tadpole(MODEL_CONFIG).to(DEVICE)
    title = f"{MODEL} · latent scale {latent_scale} · {'x'.join(map(str, config.grid.shape))}"
    tracker = stage.track({"title": title, "configs": configs_state([config, MODEL_CONFIG, TRAIN_CONFIG])}, SCHEMAS)
    fine_tune_autoencoder(model, train_loader, loader, stats, TRAIN_CONFIG, stage.folder / "checkpoints", tracker)
    model.eval()
    with torch.no_grad():
        reference = next(iter(loader)).to(DEVICE).normalize(stats)
        prediction = model.decode(model.encode(reference))
    with mpl.rc_context(STYLE):
        curves = {
            "training_curves.png": training_curves(tracker.frame(TRAIN)),
            "validation_curves.png": validation_curves(tracker.frame(VALIDATION_MEDIANS)),
        }
        for figure in curves.values():
            figure.suptitle(f"{MODEL} at latent scale {latent_scale}")
        slices = plot_slices(reference, prediction, config.grid)
    return {
        "per_design.parquet": final_per_design(tracker.frame(VALIDATION), latent_scale),
        "adapter.pt": model.adapter(),
        **curves,
        "slices_first_validation_design.png": slices,
    }


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
        train = stage.load(f"{TRAIN}.parquet")
        per_design_tables.append(stage.load("per_design.parquet"))
        zero_shot_tables.append(zero_shot_medians(stage.load(f"{VALIDATION_MEDIANS}.parquet"), latent_scale))
        training_rows.append(training_row(train, latent_scale))

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
