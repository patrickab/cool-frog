"""Zero-shot Tadpole reconstruction errors on all curated CoolData designs, run from the repo root."""

import polars as pl
import torch

from coolfrog.engine.cooldata import CoolData, CoolMetadata, create_dataloader
from coolfrog.engine.model_evaluation import evaluate, summarize, write_metric_tables
from coolfrog.engine.tadpole import Tadpole
from coolfrog.lib.config import HOST_CONFIG, DataConfig, TadpoleConfig
from coolfrog.lib.datatypes import NormalizationStats
from coolfrog.lib.experiment import PATH_EXPERIMENTS, run_experiment
from coolfrog.lib.log import get_logger, setup_logging, track
from coolfrog.lib.plot import plot_slices

TITLE = "Zero-shot Tadpole reconstruction on CoolData"
GOAL = (
    "Measure NRMSE and spectral error of the frozen, pretrained Tadpole-B autoencoder on every curated CoolData design, "
    "normalized with the quantile stats of experiment 02."
)
HYPOTHESIS = None

DATA_CONFIG = DataConfig(batch_size=4)
TADPOLE_CONFIG = TadpoleConfig(size="B", max_internal_batchsize=7)  # one design per encoder/decoder call, 28 fields OOM
PATH_STATS = PATH_EXPERIMENTS / "02_cooldata-value-distribution" / "results" / "normalization_stats.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with run_experiment(TITLE, GOAL, HYPOTHESIS, [DATA_CONFIG, TADPOLE_CONFIG]) as path_results:
    setup_logging(log_file=path_results / "run.log")
    log = get_logger("coolfrog.experiment")
    stats = NormalizationStats.load(PATH_STATS)
    stats.save(path_results / "normalization_stats.csv")
    excluded = pl.read_csv(PATH_STATS.with_name("excluded_designs.csv"))["design_id"]
    metadata = CoolMetadata(
        CoolMetadata.load(HOST_CONFIG.n_splits, HOST_CONFIG.idx_split).designs.filter(~pl.col("design_id").is_in(excluded))
    )
    dataset = CoolData(metadata, DATA_CONFIG)  # downloads and voxelizes every uncached design
    loader = create_dataloader(dataset, DATA_CONFIG, shuffle=False)
    log.info("%d curated designs on grid %s", len(dataset), DATA_CONFIG.grid.shape)
    model = Tadpole(TADPOLE_CONFIG).to(DEVICE).eval()

    tables = evaluate(model, track(loader, "Reconstructing"), stats)
    write_metric_tables(tables, path_results)
    for name, per_design in tables.items():
        log.info("%s\n%s", name, summarize(per_design))

    with torch.no_grad():
        reference = next(iter(loader)).to(DEVICE).normalize(stats)
        prediction = model.decode(model.encode(reference))
    plot_slices(reference, prediction, DATA_CONFIG.grid).savefig(path_results / "slices_first_design.png", dpi=150)
