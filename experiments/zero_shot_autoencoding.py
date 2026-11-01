"""Zero-shot Tadpole-B reconstruction of the curated validation designs at 2x and 4x, the baseline of all later experiments.

Run from the repo root. Results:

- `results/per_design.parquet` (Hugging Face): NRMSE and spectral loss per latent scale, design and channel
- `results/zero-shot-nrmse.csv`, `results/zero-shot-spectral-loss.csv` (README tables): median and q0.9 per channel
- `results/latent_scale_<n>/slices_first_validation_design.png`
"""

from autoencoding import (
    LATENT_SCALES,
    curated_metadata,
    data_config,
    load_stats,
    per_design_frame,
    quantile_table,
    tadpole_config,
    train_validation_split,
    validation_loader,
)
import polars as pl
import torch

from coolfrog.engine.model_evaluation import evaluate
from coolfrog.engine.tadpole import Tadpole
from coolfrog.lib.experiment import inject_table, run_experiment, write_configs
from coolfrog.lib.huggingface import HuggingFaceHub
from coolfrog.lib.log import get_logger, setup_logging, track
from coolfrog.lib.plot import plot_slices

TITLE = "Zero-shot Tadpole reconstruction on CoolData"
GOAL = (
    "Measure NRMSE and spectral loss of the frozen, pretrained Tadpole-B autoencoder on the curated validation designs "
    "at 800x160x32 (32³ tiles) and 1600x320x64 (64³ tiles): the baseline of every fine-tuning experiment."
)
HYPOTHESIS = None

QUANTILES = (0.5, 0.9)
DEVICE = "cuda"


def reconstruct(latent_scale: int) -> pl.DataFrame:
    """Evaluate the frozen model at one resolution, write its configs and slice plot, return the per-design table."""
    config = data_config(latent_scale)
    model_config = tadpole_config(latent_scale, "frozen")
    path_run = path_results / f"latent_scale_{latent_scale}"
    path_run.mkdir(exist_ok=True)
    write_configs([config, model_config], path_run / "config.yaml")

    _, validation = train_validation_split(metadata, config)
    loader = validation_loader(validation, config)
    log.info("%d validation designs on grid %s", len(validation), config.grid.shape)
    model = Tadpole(model_config).to(DEVICE).eval()
    tables = evaluate(model, track(loader, f"Reconstructing at latent scale {latent_scale}"), stats)

    with torch.no_grad():
        reference = next(iter(loader)).to(DEVICE).normalize(stats)
        prediction = model.decode(model.encode(reference))
    plot_slices(reference, prediction, config.grid).savefig(path_run / "slices_first_validation_design.png", dpi=150)
    return per_design_frame(tables, latent_scale, "zero-shot")


with run_experiment(TITLE, GOAL, HYPOTHESIS, hub=HuggingFaceHub()) as path_results:
    setup_logging(log_file=path_results / "run.log")
    log = get_logger("coolfrog.experiment")
    stats = load_stats()
    metadata = curated_metadata()

    per_design = pl.concat([reconstruct(latent_scale) for latent_scale in LATENT_SCALES])
    per_design.write_parquet(path_results / "per_design.parquet")
    for metric, name in (("nrmse", "zero-shot-nrmse"), ("spectral_loss", "zero-shot-spectral-loss")):
        table = quantile_table(per_design, metric, QUANTILES).drop("model")
        log.info("%s\n%s", name, table)
        inject_table(path_results, name, table)
