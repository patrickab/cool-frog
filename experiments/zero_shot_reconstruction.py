"""Zero-shot Tadpole reconstruction errors on all CoolData designs, run from the repo root."""

import polars as pl
import torch
from tqdm import tqdm

from coolfrog.engine.cooldata import CoolData, create_dataloader
from coolfrog.engine.tadpole import Tadpole
from coolfrog.lib.config import DataConfig, TadpoleConfig
from coolfrog.lib.datatypes import CHANNELS, Grid, NormalizationStats
from coolfrog.lib.experiment import create_experiment
from coolfrog.lib.log import get_logger, setup_logging
from coolfrog.lib.metrics import nrmse, spectral_loss
from coolfrog.lib.plot import plot_slices

TITLE = "Zero-shot Tadpole reconstruction on CoolData"
GOAL = "Measure NRMSE and spectral error of the frozen, pretrained Tadpole-B autoencoder on every voxelized CoolData design."
HYPOTHESIS = None

DATA_CONFIG = DataConfig(grid=Grid(resolution=(400, 80, 16)), batch_size=4)
TADPOLE_CONFIG = TadpoleConfig(size="B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

path_results = create_experiment(TITLE, GOAL, HYPOTHESIS, [DATA_CONFIG, TADPOLE_CONFIG])
setup_logging(log_file=path_results / "run.log")
log = get_logger("coolfrog.experiment")

dataset = CoolData(DATA_CONFIG)  # downloads and voxelizes every uncached design
loader = create_dataloader(dataset, DATA_CONFIG, shuffle=False)
log.info("%d designs on grid %s", len(dataset), DATA_CONFIG.grid.resolution)

# no training split in zero-shot, so global per-channel min/max over all designs
stats = NormalizationStats.from_flowfields(tqdm(loader, desc="Normalization stats"))
model = Tadpole(TADPOLE_CONFIG).to(DEVICE).eval()

nrmse_tables, spectral_tables = [], []
with torch.no_grad():
    for batch_index, flowfield in enumerate(tqdm(loader, desc="Reconstructing")):
        reference = flowfield.to(DEVICE).normalize(stats)
        prediction = model.decode(model.encode(reference))
        nrmse_tables.append(nrmse(prediction, reference))  # scale-invariant, equals physical-unit NRMSE
        spectral_tables.append(spectral_loss(prediction, reference))  # in normalized units
        if batch_index == 0:
            plot_slices(reference, prediction, DATA_CONFIG.grid).savefig(path_results / "slices_first_design.png", dpi=150)

channel_columns = [channel.value for channel in CHANNELS]
for name, tables in (("nrmse", nrmse_tables), ("spectral_loss", spectral_tables)):
    per_design = pl.concat(tables)
    per_design.write_csv(path_results / f"{name}_per_design.csv")
    channels = pl.col(channel_columns)
    statistics = {"mean": channels.mean(), "median": channels.median(), "max": channels.max()}
    summary = pl.concat([per_design.select(pl.lit(statistic).alias("statistic"), values) for statistic, values in statistics.items()])
    summary.write_csv(path_results / f"{name}_summary.csv")
    log.info("%s\n%s", name, summary)
