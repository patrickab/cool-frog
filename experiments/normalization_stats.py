"""Global per-channel fluid min and max over all CoolData designs, run from the repo root."""

import polars as pl

from coolfrog.engine.cooldata import CoolData, CoolMetadata, create_dataloader
from coolfrog.lib.config import HOST_CONFIG, DataConfig
from coolfrog.lib.datatypes import CHANNELS, NormalizationStats
from coolfrog.lib.experiment import create_experiment
from coolfrog.lib.log import get_logger, setup_logging, track

TITLE = "Min-max normalization stats on CoolData"
GOAL = "Measure the global per-channel fluid min and max over every voxelized CoolData design, reused for normalization."
HYPOTHESIS = None

DATA_CONFIG = DataConfig(batch_size=4)

path_results = create_experiment(TITLE, GOAL, HYPOTHESIS, [DATA_CONFIG])
setup_logging(log_file=path_results / "run.log")
log = get_logger("coolfrog.experiment")

metadata = CoolMetadata.load(HOST_CONFIG.n_splits, HOST_CONFIG.idx_split)
dataset = CoolData(metadata, DATA_CONFIG)  # downloads and voxelizes every uncached design
loader = create_dataloader(dataset, DATA_CONFIG, shuffle=False)
log.info("%d designs on grid %s", len(dataset), DATA_CONFIG.grid.resolution)

stats = NormalizationStats.from_flowfields(track(loader, "Normalization stats"))
table = pl.DataFrame(
    {
        "channel": [channel.value for channel in CHANNELS],
        "min": stats.channel_min.flatten().tolist(),
        "max": stats.channel_max.flatten().tolist(),
    }
)
table.write_csv(path_results / "normalization_stats.csv")
log.info("normalization stats\n%s", table)
