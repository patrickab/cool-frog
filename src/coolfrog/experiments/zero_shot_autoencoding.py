"""Zero-shot Tadpole-B reconstruction of the curated validation designs at 2x and 4x on 32³ and 64³ tiles.

Results:

- `results/per_design.parquet` (Hugging Face): NRMSE, MSE and spectral loss per latent scale, design and channel
- `results/zero-shot-nrmse.csv`, `results/zero-shot-spectral-loss.csv` (README tables): median and q0.9 per channel
- `results/latent_scale_<n>/slices_first_validation_design.png`
"""

import logging

import polars as pl
import torch

from coolfrog.engine.cooldata import CoolData
from coolfrog.engine.model_evaluation import evaluate
from coolfrog.engine.tadpole import Tadpole
from coolfrog.experiments.autoencoding import (
    LATENT_SCALES,
    curated_metadata,
    data_config,
    load_stats,
    per_design_frame,
    quantile_table,
    split_metadata,
    tadpole_config,
    validation_loader,
)
from coolfrog.lib.experiment import run_experiment, write_configs
from coolfrog.lib.huggingface import HuggingFaceHub
from coolfrog.lib.log import track
from coolfrog.lib.plot import plot_slices

TITLE = "Zero-shot Tadpole reconstruction on CoolData"
GOAL = (
    "Measure NRMSE and spectral loss of the frozen, pretrained Tadpole-B autoencoder on the curated validation designs "
    "at 800x160x32 (32³ tiles) and 1600x320x64 (64³ tiles). Fine-tuning runs compare against their own update 0 on whole "
    "domains instead, a different setup."
)
HYPOTHESIS = None

QUANTILES = (0.5, 0.9)
DEVICE = "cuda"


def reconstruct(latent_scale: int, validation: CoolData) -> pl.DataFrame:
    """Evaluate the frozen model at one resolution, write its configs and slice plot, return the per-design table."""
    config = validation.config
    model_config = tadpole_config(latent_scale, "frozen")
    path_run = run.path_results / f"latent_scale_{latent_scale}"
    path_run.mkdir(exist_ok=True)
    write_configs([config, model_config], path_run / "config.yaml")

    loader = validation_loader(validation, config)
    log.info("%d validation designs on grid %s", len(validation), config.grid.shape)
    model = Tadpole(model_config).to(DEVICE).eval()
    tables = evaluate(model, track(loader, f"Reconstructing at latent scale {latent_scale}"), stats)

    with torch.no_grad():
        reference = next(iter(loader)).to(DEVICE).normalize(stats)
        prediction = model.decode(model.encode(reference))
    plot_slices(reference, prediction, config.grid).savefig(path_run / "slices_first_validation_design.png", dpi=150)
    return per_design_frame(tables, latent_scale, "zero-shot")


with run_experiment(TITLE, GOAL, HYPOTHESIS, hub=HuggingFaceHub()) as run:
    log = logging.getLogger("coolfrog.experiment")
    stats = load_stats()
    _, validation_metadata = split_metadata(curated_metadata())
    # Voxelize every scale before the first model, the forked voxelizing workers hang once CUDA is initialized
    validations = {latent_scale: CoolData(validation_metadata, data_config(latent_scale)) for latent_scale in LATENT_SCALES}

    per_design = pl.concat([reconstruct(latent_scale, validation) for latent_scale, validation in validations.items()])
    per_design.write_parquet(run.path_results / "per_design.parquet")
    for metric, name in (("nrmse", "zero-shot-nrmse"), ("spectral_loss", "zero-shot-spectral-loss")):
        table = quantile_table(per_design, metric, QUANTILES).drop("model")
        log.info("%s\n%s", name, table)
        run.publish(name, table)
