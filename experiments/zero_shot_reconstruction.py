"""Zero-shot Tadpole reconstruction errors on all CoolData designs, run from the repo root."""

import torch

from coolfrog.engine.cooldata import CoolData, CoolMetadata, create_dataloader
from coolfrog.engine.model_evaluation import evaluate, summarize, write_metric_tables
from coolfrog.engine.tadpole import Tadpole
from coolfrog.lib.config import HOST_CONFIG, DataConfig, TadpoleConfig
from coolfrog.lib.datatypes import NormalizationStats
from coolfrog.lib.experiment import create_experiment
from coolfrog.lib.log import get_logger, setup_logging, track
from coolfrog.lib.plot import plot_slices

TITLE = "Zero-shot Tadpole reconstruction on CoolData"
GOAL = "Measure NRMSE and spectral error of the frozen, pretrained Tadpole-B autoencoder on every voxelized CoolData design."
HYPOTHESIS = None

DATA_CONFIG = DataConfig(batch_size=4)
TADPOLE_CONFIG = TadpoleConfig(size="B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

path_results = create_experiment(TITLE, GOAL, HYPOTHESIS, [DATA_CONFIG, TADPOLE_CONFIG])
setup_logging(log_file=path_results / "run.log")
log = get_logger("coolfrog.experiment")

metadata = CoolMetadata.load(HOST_CONFIG.n_splits, HOST_CONFIG.idx_split)
dataset = CoolData(metadata, DATA_CONFIG)  # downloads and voxelizes every uncached design
loader = create_dataloader(dataset, DATA_CONFIG, shuffle=False)
log.info("%d designs on grid %s", len(dataset), DATA_CONFIG.grid.shape)

# no training split in zero-shot, so global per-channel min/max over all designs
stats = NormalizationStats.from_flowfields(track(loader, "Normalization stats"))
model = Tadpole(TADPOLE_CONFIG).to(DEVICE).eval()

tables = evaluate(model, track(loader, "Reconstructing"), stats)
write_metric_tables(tables, path_results)
for name, per_design in tables.items():
    log.info("%s\n%s", name, summarize(per_design))

with torch.no_grad():
    reference = next(iter(loader)).to(DEVICE).normalize(stats)
    prediction = model.decode(model.encode(reference))
plot_slices(reference, prediction, DATA_CONFIG.grid).savefig(path_results / "slices_first_design.png", dpi=150)
