"""Build this host's voxel cache, then stream it once through a DataLoader, run from the repo root."""

import logging

from coolfrog.engine.cooldata import CoolData, CoolMetadata, create_dataloader
from coolfrog.lib.config import HOST_CONFIG, DataConfig, Grid
from coolfrog.lib.log import setup_logging, track


def main() -> None:
    setup_logging()
    log = logging.getLogger("coolfrog.scripts")
    config = DataConfig(batch_size=4, num_workers=4, grid=Grid(latent_scale=2))
    metadata = CoolMetadata.load(HOST_CONFIG.n_splits, HOST_CONFIG.idx_split)
    dataset = CoolData(metadata, config)  # downloads and voxelizes every uncached design

    for _ in track(create_dataloader(dataset, config, shuffle=True), "Loading batches"):
        pass
    log.info("loaded %d designs on grid %s", len(dataset), config.grid.shape)


if __name__ == "__main__":  # the forkserver of `build_voxel_cache` imports this file
    main()
