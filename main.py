import logging
import random
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from scipy.sparse import csr_matrix, save_npz
from torch.utils.data import DataLoader

from src import stereo

# Logging using rich
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[RichHandler(omit_repeated_times=False, rich_tracebacks=True)],
)
logging.captureWarnings(True)

logger = logging.getLogger("pytorch_lightning.core")


def get_dataset(cfg):
    """Build a dataset or load from checkpoint."""
    dataset_class = getattr(stereo, cfg['dataset']['name'])
    if cfg['dataset_ckpt'] is not None:
        logger.info(f"Loading dataset from {cfg['dataset_ckpt']}")
        return dataset_class.load_from_checkpoint(cfg['dataset_ckpt'])

    dataset_kwargs = cfg['dataset'].get('kwargs', {})
    return dataset_class(**dataset_kwargs)


def get_sources(dataset: stereo.GeneRegPseudotimeDataset, cfg: str | None = None):
    """Return a sequence of source_masks corresponding to genes that will
    be used as regulators.
    """
    def load_sources_from_ckpt(path):
        """Load sources from file

        Each line should be a list of TF sources. Number of lines should equal
        the number of time steps.

        ```sources.txt
        [1, 2, 3, 4]
        [5, 2, 3]
        [1, 2, 9, 87, 4]
        ```
        """
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [np.asarray(eval(line)) for line in lines]
            sources_mask_seq = [np.full(n, False) for n in dataset.n_nodes_seq]
            assert len(lines) == len(sources_mask_seq)
            is_tf_idx = dataset.gene_is_tf.nonzero()[0]
            for sources_mask, line in zip(sources_mask_seq, lines):
                sources_mask[is_tf_idx[line]] = True
            return sources_mask_seq

    # Sources ckpt takes priority
    if cfg['sources_ckpt'] is not None:
        return load_sources_from_ckpt(cfg['sources_ckpt'])

    ckpt = cfg['ckpt']

    if ckpt is None:  # use all known TFs as sources if no previous ckpt
        if hasattr(dataset, 'gene_is_tf'):
            return [torch.from_numpy(dataset.gene_is_tf)] * dataset.n_seq
        return None  # Use all nodes as sources if not a gene reg dataset

    # otherwise we only load the TFs selected by HierProx
    ckpt = torch.load(ckpt)
    theta_seq = ckpt['callbacks']['HierProx']['theta_seq_']
    assert len(theta_seq) == dataset.n_seq

    if not hasattr(dataset, 'gene_is_tf'):  # simulation study
        return [theta.cpu() > 0 for theta in theta_seq]

    # shape of theta's equals shape of TFs, so we restrict to TFs before indexing
    tfs = dataset.genes[dataset.gene_is_tf]  # all TF names
    selected_tfs_seq = [tfs[theta.cpu() > 0] for theta in theta_seq]  # sel. TFs
    # construct new masks of right shape
    sources_mask_seq = [np.isin(dataset.genes, selected_tfs)
                        for selected_tfs in selected_tfs_seq]

    logger.info(f'Selected sources per t: {[len(i) for i in selected_tfs_seq]}')
    return sources_mask_seq


def get_module(cfg, dataset, sources_mask_seq):
    """Return module"""
    if 'model_kwargs' not in cfg['module_conf']:
        raise ValueError("'model_kwargs' not found in module_conf.")

    kwargs = cfg['module_conf'].copy()

    kwargs['model_kwargs']['in_features'] = dataset.in_features
    kwargs['model_kwargs']['out_features'] = dataset.out_features

    if cfg['warm_init']:
        kwargs['init_A_seq'] = dataset.correlation_matrices()

    module = getattr(stereo, cfg['module'])(
        n_nodes_seq=dataset.n_nodes_seq,
        sources_mask_seq=sources_mask_seq,
        **kwargs,
    )

    if torch.cuda.is_available():
        logger.info('Using cuda')
        module = module.to('cuda')
    else:
        logger.info('Using cpu')

    return module


def get_trainer(cfg, dataset, module):
    """Get trainer"""
    # Try using wandb if installed
    try:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(
            save_dir='.',
            project=f"{dataset.__class__.__name__}_logs",
        )
        wandb_logger.watch(module, log="all")
        trainer_kwargs = {'logger': wandb_logger}
    except ImportError:
        logger.warning("'wandb' not installed. Will use default logger.")
        trainer_kwargs = {}

    trainer = pl.Trainer(
        max_epochs=cfg['max_epochs'],
        log_every_n_steps=cfg['log_every_n_steps'],
        **trainer_kwargs,
    )
    return trainer


def train_pl_module(cfg, dataset):
    """Train STEREO-GCN and Autoregressive baseline"""
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    sources_mask_seq = get_sources(dataset, cfg)
    module = get_module(cfg, dataset, sources_mask_seq)
    trainer = get_trainer(cfg, dataset, module)

    try:
        trainer.fit(model=module, train_dataloaders=dataloader)
    except InterruptedError as ie:
        logger.info(ie)
        pass

    return module


def train_tvdbn(cfg, dataset):
    """Trainer TVDBN"""
    tvdbn = stereo.TVDBN(**cfg['module_conf'])
    sources_mask_seq = get_sources(dataset, cfg)
    # only need the first mask as TFs are the same for all t
    sources_mask = sources_mask_seq[0] if sources_mask_seq is not None else None
    tvdbn.fit(dataset, sources_mask=sources_mask)
    return tvdbn


def train_tvgl(cfg, dataset):
    """Trainer TVDBN"""
    tvgl = stereo.TVGL(**cfg['module_conf'])
    sources_mask_seq = get_sources(dataset, cfg)
    tvgl.fit(dataset, sources_mask_seq=sources_mask_seq)
    return tvgl


@hydra.main(version_base=None, config_path="configs")
def train(cfg: DictConfig) -> None:
    SEED = cfg['seed']

    # Set all seeds
    logger.info(f"Setting seed={SEED}")
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    """Select module and train."""
    dataset = get_dataset(cfg)
    save_adj = False

    if cfg['module'] in ['STEREO_GCN_Module', 'TemporalDeepAutoreg_Module']:
        module = train_pl_module(cfg, dataset)
        savedir = Path(module.logger.experiment.dir)
    elif cfg['module'] in ['TVDBN', 'TVGL']:
        savedir = Path('results') / cfg['dataset']['name']
        savedir.mkdir(exist_ok=True, parents=True)
        save_adj = True
        fn = train_tvdbn if cfg['module'] == 'TVDBN' else train_tvgl
        module = fn(cfg, dataset)
    else:
        raise NotImplementedError(f"Module named {cfg['module']} not found.")

    if save_adj:
        # Dump adjacency matriecs and config used
        savedir = savedir / f"{cfg['module']}-SEED:{SEED}-{datetime.now().strftime('%m.%d.%H:%M:%S')}"
        A_dir = savedir / 'adjs'
        logger.info(f"Storing adjacency matrices in '{A_dir}'.")
        A_dir.mkdir(exist_ok=True, parents=True)
        OmegaConf.save(cfg, savedir / "config.yaml")

        for i, A in enumerate(module.A_seq_):
            if isinstance(A, torch.Tensor):
                A = A.cpu().data.numpy()
            A = csr_matrix(A)
            save_npz(A_dir / f'A_{i}.npz', A)
    else:
        OmegaConf.save(cfg, savedir / "config.yaml")


if __name__ == "__main__":
    train()
