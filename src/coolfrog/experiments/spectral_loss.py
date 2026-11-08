"""LoRA-32 fine-tuning at latent scale 1 with fluid MSE alone vs. MSE plus spectral distance, same data, seed and updates.

`--resume` continues an unfinished run. One `Stage` per loss, `results/<loss>/`, with the outputs of
`finetune_autoencoder.fine_tune`, its `config.yaml` holds the stage's `TrainConfig`.

- Both stages seed alike, so they start from the same LoRA initialization and see the same designs and crops

- `SPECTRAL_WEIGHT` = 0.05: median ||∇ MSE|| / ||∇ spectral distance|| over 8 training batches at LoRA init (GigaChad,
  ratios 0.012 to 0.38), so both terms pull about equally hard at the start
- 5 epochs, meant for the 10 % data share (`_partial`), warmup and validation every 100 updates since 5 epochs there are
  about 900 updates

README tables: `results/nrmse.csv`, `results/mse.csv`, `results/spectral-loss.csv`, `results/training.csv`.
"""

from dataclasses import replace
import logging
import sys

import polars as pl
import torch

from coolfrog.engine.model_evaluation import MSE, NRMSE, SPECTRAL_LOSS
from coolfrog.experiments.autoencoding import curated_metadata, load_stats, quantile_table
from coolfrog.experiments.finetune_autoencoder import MODEL, MODEL_CONFIG, fine_tune, training_row, zero_shot_medians
from coolfrog.lib.config import SEED, TrainConfig, seed_everything
from coolfrog.lib.distributed import rank
from coolfrog.lib.experiment import run_experiment
from coolfrog.lib.huggingface import HuggingFaceHub
from coolfrog.lib.plots import METRICS_TRAINING_AGGREGATES, METRICS_VALIDATION_AGGREGATES

TITLE = "Spectral loss in LoRA fine-tuning"
GOAL = (
    "Fine-tune Tadpole-B with rank-32 LoRA at 400x80x16 once on the fluid MSE and once on the MSE plus the spectral "
    "distance, and compare both on the same validation designs."
)
HYPOTHESIS = "The spectral term lowers the validation spectral loss without raising the validation NRMSE."

LATENT_SCALE = 1
SPECTRAL_WEIGHT = 0.05
MSE_CONFIG = TrainConfig(epochs=5, warmup_updates=100, validate_every=100)
LOSSES = {  # stage name -> model label, train config
    "mse": (MODEL, MSE_CONFIG),
    "mse_spectral": (f"{MODEL} + spectral", replace(MSE_CONFIG, spectral_weight=SPECTRAL_WEIGHT)),
}
log = logging.getLogger("coolfrog.experiment")

if __name__ == "__main__":
    # one TrainConfig per class name in config.yaml, the stage folders hold each loss's own
    with run_experiment(
        TITLE, GOAL, HYPOTHESIS, [MODEL_CONFIG, MSE_CONFIG], hub=HuggingFaceHub(), resume="--resume" in sys.argv
    ) as run:
        stats = load_stats()
        metadata = curated_metadata()
        per_design_tables, training_rows, zero_shot = [], [], None
        for name, (model_label, train_config) in LOSSES.items():
            stage = run.stage(name)
            seed_everything(SEED + rank())  # same for every stage, not seeded by stage name
            if stage.is_complete():
                log.info("%s: stage complete, loading its results from %s", name, stage.folder)
            else:
                log.info("%s: training", name)
                stage.save(fine_tune(LATENT_SCALE, stage, metadata, stats, train_config, model_label))
                torch.cuda.empty_cache()
            per_design_tables.append(stage.load("per_design.parquet"))
            training = training_row(stage.load(f"{METRICS_TRAINING_AGGREGATES}.parquet"), LATENT_SCALE)
            training_rows.append({"model": model_label, **training})
            if zero_shot is None:  # update 0 is the same untrained model in every stage
                zero_shot = zero_shot_medians(stage.load(f"{METRICS_VALIDATION_AGGREGATES}.parquet"), LATENT_SCALE)

        per_design = pl.concat(per_design_tables)
        for metric, name in ((NRMSE, "nrmse"), (MSE, "mse"), (SPECTRAL_LOSS, "spectral-loss")):
            fine_tuned = quantile_table(per_design, metric, (0.5,))
            table = pl.concat([zero_shot.filter(pl.col("metric") == metric).drop("metric"), fine_tuned], how="vertical_relaxed")
            log.info("%s\n%s", name, table)
            run.publish(name, table.drop("statistic"))
        run.publish("training", pl.DataFrame(training_rows))
