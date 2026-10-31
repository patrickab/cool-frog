"""Global per-channel fluid min and max over all CoolData designs at 1600x320x64, run from the repo root."""

from coolfrog.engine.cooldata import CoolMetadata
from coolfrog.lib.config import HOST_CONFIG, DataConfig
from coolfrog.lib.datatypes import Grid, NormalizationStats
from coolfrog.lib.experiment import create_experiment
from coolfrog.lib.log import get_logger, setup_logging

TITLE = "Min-max normalization stats on CoolData"
GOAL = (
    "Measure the global per-channel fluid min and max over every CoolData design at 1600x320x64, "
    "the shared normalization of all subsequent models at every resolution."
)
HYPOTHESIS = None

DATA_CONFIG = DataConfig(grid=Grid(latent_scale=4), batch_size=1, num_workers=8)  # 0.92 GB and 1.2 s read per design

path_results = create_experiment(TITLE, GOAL, HYPOTHESIS, [DATA_CONFIG])
setup_logging(log_file=path_results / "run.log")
log = get_logger("coolfrog.experiment")

metadata = CoolMetadata.load(HOST_CONFIG.n_splits, HOST_CONFIG.idx_split)
log.info("%d designs on grid %s", metadata.designs.height, DATA_CONFIG.grid.shape)
stats = NormalizationStats.compute(metadata, DATA_CONFIG)
stats.save(path_results / "normalization_stats.csv")  # read back with `NormalizationStats.load`
log.info("normalization stats\n%s", (path_results / "normalization_stats.csv").read_text())
