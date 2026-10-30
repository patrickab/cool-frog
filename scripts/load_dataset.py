"""Build this host's voxel cache, then stream it once through a DataLoader, run from the repo root."""

from coolfrog.engine.cooldata import CoolData, CoolMetadata, create_dataloader
from coolfrog.lib.config import HOST_CONFIG, DataConfig
from coolfrog.lib.log import get_logger, setup_logging, track

setup_logging()
log = get_logger("coolfrog.scripts")
config = DataConfig(batch_size=4, num_workers=4)
metadata = CoolMetadata.load(HOST_CONFIG.n_splits, HOST_CONFIG.idx_split)
dataset = CoolData(metadata, config)  # downloads and voxelizes every uncached design

for _ in track(create_dataloader(dataset, config, shuffle=True), "Loading batches"):
    pass
log.info("loaded %d designs on grid %s", len(dataset), config.grid.resolution)
