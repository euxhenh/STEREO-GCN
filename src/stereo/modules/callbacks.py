import logging
from pathlib import Path
from typing import Any, Dict, List, TypeAlias

import numpy as np
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
from pytorch_lightning.callbacks import Callback
from torch import FloatTensor, LongTensor

from ..utils.plotting import interpolate_2dimage
from ..utils.third_party import estimate_lambda, hier_prox

logger = logging.getLogger(__name__)

THETA_SEQ: TypeAlias = List[FloatTensor]


class HierProx(Callback):
    """Hierarchical Proximal operator callback.

    Parameters
    ----------
    params: Iterable[Tensor]
        Parameters to penalize column-wise.
    min_n_nonzero: int
        Minimum number of nonzero-entries allowed. If number of src nodes
        for a given time point is about to be less than this, then the
        update for that time point will freeze.
    soft_lambda: float
        The soft-thresholding lambda for hier-prox. A higher value leads to
        a sparser theta, and consequently, a row-sparser graph. If None,
        will automatically estimate a good starting point.
    M: float
        The hierarchy multiplier. I.e., |W|_infty < M * theta.
    path_multiplier: float
        After every proximal update, will update prox_lambda *= (1 +
        prox_path_multiplier).
    warm_epochs: int
        Number of epochs to warm-up for before applying hier-prox.
    run_every_n: int
        Apply hier prox every `run_every_n` epochs (after warm start). If
        0, will compute every epoch.
    fillval: float
        Used to fill initial theta parameters.

    Attributes
    ----------
    theta_seq_: List[Tensor]
        List of thetas used by hier-prox for every adjacency matrix.
    theta_seq_history_: Dict[int, List[Tensor]]
        A dictionary mapping epoch to a list of theta's learned by
        hier-prox.
    freeze_update_: Tensor[int]
        A 1D tensor with the same length as the number of time points. If
        the i-th element is True, then params[i] has converged.
    """
    UPDATABLE: int = -1

    def __init__(
        self,
        params_key: str,
        *,
        clip_A: bool = False,
        min_n_nonzero: int = 1,
        soft_lambda: float | None = None,
        M: float = 10.0,
        path_multiplier: float = 0.02,
        warm_epochs: int = 0,
        run_every_n: int = 0,
        fillval: float = 1.0,
        device: str = 'cpu',
    ):
        assert warm_epochs >= 0
        self.params_key = params_key
        self.clip_A = clip_A
        self.min_n_nonzero = min_n_nonzero
        self.soft_lambda = soft_lambda
        self.M = M
        self.path_multiplier = path_multiplier
        self.warm_epochs = warm_epochs
        self.run_every_n = run_every_n
        self.fillval = fillval
        self.device = device

    @property
    def all_frozen(self) -> bool:
        """Return True if all A parameters stopped updating."""
        return torch.all(self._freeze_update != self.UPDATABLE)

    @property
    def freeze_update_(self) -> torch.BoolTensor:
        """Return a tensor with i-th item set to true if i-th param stopped
        converging.
        """
        return self._freeze_update != self.UPDATABLE

    def setup(self, trainer, pl_module, stage) -> None:
        """Called when training begins."""
        params = getattr(pl_module, self.params_key)

        for param in params:
            if param.ndim != 2:
                raise ValueError("Expected 2D parameter for hier-prox.")

        self.n_seq_ = len(params)

        self.theta_seq_: THETA_SEQ = [
            torch.full((param.shape[1],), self.fillval, requires_grad=False).to(self.device)
            for param in params
        ]
        self.theta_seq_history_: Dict[int, THETA_SEQ] = {}
        self._freeze_update: LongTensor = torch.full((self.n_seq_,), self.UPDATABLE)

        if self.soft_lambda is None:
            self.soft_lambda = self.init_soft_lambda(params)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Applies the Hierarchical Proximal Operator from LassoNet."""
        if not self.should_step(pl_module):
            return

        params = getattr(pl_module, self.params_key)
        n_nonzero_seq = []
        for i, (theta, W) in enumerate(zip(self.theta_seq_, params)):
            if self.freeze_update_[i]:
                n_nonzero_seq.append(len(theta.nonzero().data.ravel()))
                continue
            upd_theta, upd_W = hier_prox(
                theta, W, soft_lambda=self.soft_lambda, M=self.M)
            n_nonzero = len(upd_theta.nonzero().data.ravel())
            n_nonzero_seq.append(n_nonzero)

            if self.min_n_nonzero > n_nonzero:
                self._freeze_update[i] = pl_module.current_epoch
            else:
                theta.data = upd_theta.data
                W.data = upd_W.data

        if self.clip_A:
            pl_module._clip_A_seq()

        pl_module.roll_back_weights()  # In case we froze any
        pl_module.store_old_weights()  # Store post-update weights

        pl_module.log_dict({
            'n_A_updatable': (~self.freeze_update_).sum().float(),
            'average_n_nonzero': float(np.mean(n_nonzero_seq)),
            'min_n_nonzero': float(np.min(n_nonzero_seq)),
            'soft_lambda': float(self.soft_lambda),
        })

        self.step_soft_lambda()

        self.theta_seq_history_[pl_module.current_epoch] = [
            theta.cpu().clone() for theta in self.theta_seq_
        ]

    def should_step(self, pl_module: pl.LightningModule) -> bool:
        """return True if should take a hier-prox step."""
        if self.warm_epochs > pl_module.current_epoch:  # Not warm yet
            return False
        if self.run_every_n <= 1:
            return True
        if (pl_module.current_epoch - self.warm_epochs) % self.run_every_n == 0:
            return True
        return False

    def step_soft_lambda(self) -> None:
        """Apply step on prox lambda."""
        self.soft_lambda *= (1 + self.path_multiplier)

    def init_soft_lambda(self, params) -> float:
        # Init lambda for the first time based on the first time point
        logger.info("Estimating Hier-Prox lambda...")
        soft_lambda = estimate_lambda(self.theta_seq_[0], params[0], M=self.M)
        logger.info(f"Initializing Hier-Prox with lambda={soft_lambda}")
        return soft_lambda

    @property
    def hparams(self) -> Dict[str, Any]:
        """Return dict of hparams"""
        return {
            'min_n_nonzero': self.min_n_nonzero,
            'soft_lambda': self.soft_lambda,
            'M': self.M,
            'path_multiplier': self.path_multiplier,
            'warm_epochs': self.warm_epochs,
            'run_every_n': self.run_every_n,
            'fillval': self.fillval,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            'theta_seq_': self.theta_seq_,
            'theta_seq_history_': self.theta_seq_history_,
            '_freeze_update': self._freeze_update,
            **self.hparams,
        }


class ASeqLogger(Callback):
    """Logs A seq."""

    def __init__(self, log_A_every_k_epochs: int | None = None, save_A_seq: bool = False):
        self.log_A_every_k_epochs = log_A_every_k_epochs
        self.save_A_seq = save_A_seq

    @torch.no_grad()
    def on_train_epoch_end(self, _, pl_module) -> None:
        if pl_module._is_last_epoch or (
            self.log_A_every_k_epochs and
            pl_module.current_epoch % self.log_A_every_k_epochs == 0
        ):
            A_seq = pl_module.A_seq_  # cache
            self._log_A_seq_plot(pl_module, A_seq)
            if self.save_A_seq:
                self._save_A_seq(pl_module, A_seq)

    def _save_A_seq(self, pl_module, A_seq):
        savedir = Path(pl_module.logger.experiment.dir) / 'adjs' / f"{pl_module.current_epoch}"
        savedir.mkdir(exist_ok=True, parents=True)

        for i, A in enumerate(A_seq):
            A = sp.csr_array(A.cpu().data.numpy())
            sp.save_npz(savedir / f'A_{i}.npz', A)

    def _log_A_seq_plot(self, pl_module, A_seq):
        """Logs each adjacency matrix using the logger."""

        def posneg(A):
            pos = interpolate_2dimage(torch.clip(A, 0, None), 'Reds')
            neg = interpolate_2dimage(torch.clip(A, None, 0), 'Blues', reverse=True)
            pos[A < 0] = neg[A < 0]
            pos = (pos * 255).data
            if pos.is_cuda:
                pos = pos.cpu()
            pos = pos.numpy().astype(np.uint8)
            return pos

        # Don't save matrices if different shapes or wandb will complain
        shapes = [A.shape for A in A_seq]
        if len(set(shapes)) > 1:
            return

        A_seq = [posneg(A) for A in A_seq]
        caption = [f"A-epoch={pl_module.current_epoch}-i={i}" for i in range(len(A_seq))]
        pl_module.logger.log_image(key='A_seq', images=A_seq, caption=caption)
