"""
Used to generate a store locally simuldated data based on seed.
"""
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

import stereo

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs/dataset")
def build(cfg: DictConfig) -> None:
    """Generate and save data."""
    SEED = cfg['seed']
    # Set all seeds
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    savepath = Path(cfg['savepath'])
    savepath.parents[0].mkdir(parents=True, exist_ok=True)

    dataset_kwargs = cfg.get('kwargs', {})
    dataset_class = getattr(stereo, cfg['name'])
    dataset = dataset_class(**dataset_kwargs)

    dataset.save(savepath)

    logger.info(f"Saved dataset under {savepath}")


if __name__ == '__main__':
    build()
