"""LoRA autoencoding fine-tuning of Tadpole-B at latent_scale 2 and 4, run from the repo root."""

from dataclasses import replace

import torch

from coolfrog.engine.cooldata import CoolData, CoolMetadata, create_dataloader
from coolfrog.engine.model_evaluation import evaluate, write_metric_tables
from coolfrog.engine.tadpole import Tadpole
from coolfrog.engine.training import RandomCrops, fine_tune_autoencoder
from coolfrog.lib.config import HOST_CONFIG, DataConfig, TadpoleConfig, TrainConfig
from coolfrog.lib.datatypes import Grid, NormalizationStats
from coolfrog.lib.experiment import PATH_EXPERIMENTS, create_experiment, write_configs
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

LATENT_SCALES = (2, 4)  # 800x160x32 with 32³ crops, 1600x320x64 with 64³ crops
PATH_STATS = PATH_EXPERIMENTS / "01_min-max-normalization-stats-on-cooldata" / "results" / "normalization_stats.csv"  # 1600x320x64
TRAIN_CONFIG = TrainConfig()
DEVICE = "cuda"

path_results = create_experiment(TITLE, GOAL, HYPOTHESIS, [TRAIN_CONFIG])
setup_logging(log_file=path_results / "run.log")
log = get_logger("coolfrog.experiment")
stats = NormalizationStats.load(PATH_STATS)
stats.save(path_results / "normalization_stats.csv")
metadata = CoolMetadata.load(HOST_CONFIG.n_splits, HOST_CONFIG.idx_split)


def fine_tune(latent_scale: int) -> None:
    """Fine-tune and evaluate one resolution into `results/latent_scale_<latent_scale>/`."""
    grid = Grid(latent_scale=latent_scale)
    data_config = DataConfig(grid=grid, batch_size=2, num_workers=6)  # designs per pass
    tadpole_config = TadpoleConfig(
        size="B",
        encoder_state=32,
        decoder_state=32,
        encoder_crop_size=grid.shape[2],  # crop edge = grid height, training crops equal validation tiles
        max_internal_batchsize=16,  # validation tiles per encoder/decoder call
    )
    path_run = path_results / f"latent_scale_{latent_scale}"
    path_run.mkdir(exist_ok=True)
    write_configs([data_config, tadpole_config], path_run / "config.yaml")

    dataset = CoolData(metadata, data_config)
    train, validation = dataset.split([len(dataset) - TRAIN_CONFIG.n_validation_designs, TRAIN_CONFIG.n_validation_designs])
    log.info("%d train and %d validation designs on grid %s", len(train), len(validation), grid.shape)
    crops = RandomCrops(stats, tadpole_config.encoder_crop_size, TRAIN_CONFIG.crops_per_step)
    train_loader = create_dataloader(train, data_config, shuffle=True, collate_fn=crops)
    validation_loader = create_dataloader(validation, replace(data_config, batch_size=1), shuffle=False)

    model = Tadpole(tadpole_config).to(DEVICE)
    metrics = fine_tune_autoencoder(model, train_loader, validation_loader, stats, TRAIN_CONFIG)
    metrics.write_csv(path_run / "metrics.csv")
    model.save_adapter(path_run / "adapter.pt")

    write_metric_tables(evaluate(model, validation_loader, stats), path_run)
    model.eval()
    with torch.no_grad():
        reference = next(iter(validation_loader)).to(DEVICE).normalize(stats)
        prediction = model.decode(model.encode(reference))
    plot_slices(reference, prediction, grid).savefig(path_run / "slices_first_validation_design.png", dpi=150)


for latent_scale in LATENT_SCALES:
    fine_tune(latent_scale)
    torch.cuda.empty_cache()
