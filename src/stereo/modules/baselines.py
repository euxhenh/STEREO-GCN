import logging
import os
from itertools import starmap
from typing import Any, Callable, Dict, List, TypedDict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import FloatTensor, optim

from ..models import TemporalDeepAutoreg
from ..utils.graph_ops import sparsity
from ..utils.nn_ops import soft_threshold
from ..utils.search import search_loss_fn, search_optim_fn

logger = logging.getLogger("pytorch_lightning.core")


class ModelKwargs(TypedDict):
    """Type annotations for TemporalDeepAutoreg.
    """
    n_nodes: int
    in_features: int
    out_features: int | None
    n_linear_layers: int
    nonlin: str
    init_A: str
    fillval: float
    kwargs: Dict[str, Any]


class TemporalDeepAutoreg_Module(pl.LightningModule):
    """A base module for TemporalDeepAutoreg baseline.

    Parameters
    ----------
    model_kwargs: ModelKwargs
        Parameters to pass to TemporalDeepAutoreg.
    sources_mask_seq: List[array-like]
        ID's of the source nodes for each time point. If provided, will
        only use (train) these as source nodes.
    loss_fn: str
        Loss function to use. Will search `torch.nn` and `torch`.
    loss_fn_kwargs: dict
        Kwargs for loss.
    optim_fn: str
        Optimizer to use. Will search `torch.nn` and `torch`.
    optim_fn_kwargs: dict
        Kwargs for optimizer.
    l1_lambda: float
        if non-zero, will apply L1 penalty to adjacency matrices.
    """

    def __init__(
        self,
        n_nodes_seq: List[int],
        *,
        sources_mask_seq: List[np.ndarray | torch.BoolTensor] | None = None,
        model_kwargs: ModelKwargs | None = None,
        loss_fn: Callable | str = 'MSELoss',
        loss_fn_kwargs: Dict[str, Any] | None = None,
        optim_fn: Callable | str = 'Adam',
        optim_fn_kwargs: Dict[str, Any] | None = None,
        max_sparsity: float = 0.9,
        clip_A: bool = True,
        l1_lambda: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        model_kwargs = model_kwargs or {}
        self.model = TemporalDeepAutoreg(
            n_nodes_seq=n_nodes_seq,
            sources_mask_seq=sources_mask_seq,
            **model_kwargs
        )

        self.n_nodes_seq = n_nodes_seq
        self.sources_mask_seq = sources_mask_seq
        loss_fn_kwargs = loss_fn_kwargs or {}
        self.criterion = search_loss_fn(loss_fn, **loss_fn_kwargs)
        self.optim_fn = optim_fn
        optim_fn_kwargs = optim_fn_kwargs or {'lr': 0.01}
        self.optim_fn_kwargs = optim_fn_kwargs
        self.max_sparsity = max_sparsity
        self.clip_A = clip_A
        self.l1_lambda = l1_lambda

        self._freeze_update = torch.full((self.n_seq_,), False)

    @property
    def n_seq_(self) -> int:
        """Return number of time points."""
        return self.model.n_seq_

    @property
    def A_seq_(self) -> List[FloatTensor]:
        return self.model.A_seq_

    @property
    def all_frozen(self) -> bool:
        """Return True if all A_params should stop updating."""
        return torch.all(self._freeze_update)

    def configure_optimizers(self) -> optim.Optimizer:
        return search_optim_fn(self.optim_fn)(
            self.model.parameters(), **self.optim_fn_kwargs,
        )

    def compute_loss(self, inputs, targets):
        """Computes the loss for every pair (a, b) in zip(inputs, targets)
        using criterion, then computes the mean/sum of the results.
        """
        losses = torch.stack(list(starmap(self.criterion, zip(inputs, targets))))
        return losses.mean()

    def forward(self, x_seq) -> List[FloatTensor]:
        """Predict given inputs."""
        return self.model(x_seq)

    @torch.no_grad()
    def evaluate(self, y_seq, x_seq):
        """Computes inner loss given the inputs. If A_seq is None, will use
        self.A_seq_.
        """
        out = self(x_seq)
        loss = self.compute_loss(out, y_seq)
        return loss.item()

    def on_train_start(self):
        # Print model summary
        logger.info(self)

    def training_step(self, batch, _) -> FloatTensor:
        x_seq, y_seq = batch
        self.store_old_weights()

        out = self(x_seq)
        loss = self.compute_loss(out, y_seq)

        self.log('loss', loss, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Perform ista step"""
        if self.all_frozen:
            self.stop_training("A_params stopped converging.")

        if self.l1_lambda:
            self.ista_step()
            self.roll_back_weights()

        l0_loss = self._compute_norms(0)
        l1_loss = self._compute_norms(1)
        self.log_dict({
            'l0_loss': l0_loss,
            'l1_loss': l1_loss,
        })

    def on_before_optimizer_step(self, _):
        """Clear grads for froze params"""
        for t, frozen in enumerate(self._freeze_update):
            if frozen:
                self.A_seq_[t].grad = None

    @torch.no_grad()
    def ista_step(self):
        """Proximal gradient descent."""
        for i, A in enumerate(self.A_seq_):
            if self._freeze_update[i]:
                continue

            data = soft_threshold(A.data, soft_lambda=self.l1_lambda)
            zero_rows = data.sum(1) == 0
            top_column = A.data[zero_rows].argmax(1)
            data[zero_rows][:, top_column] = 1e-4  # fill with small value

            if sparsity(data) > self.max_sparsity:
                self._freeze_update[i] = True
            else:
                A.data = data

    def on_train_epoch_end(self):
        if self.clip_A:
            self.model.clip_As()

    def on_train_end(self) -> None:
        """Print path of checkpoint."""
        self.trainer.checkpoint_callback.on_train_epoch_end(self.trainer, self)
        if not hasattr(self.trainer.checkpoint_callback, "last_model_path"):
            return
        path = self.trainer.checkpoint_callback.last_model_path
        path = os.path.abspath(path)
        print(f"Read last checkpoint from '{path}'")

    def stop_training(self, msg=''):
        """TODO: Find a graceful way to stop training."""
        # manually call this since it is not run on interrupt
        self.on_train_end()
        raise InterruptedError(f"Stopping training... {msg}")

    def _compute_norms(self, norm):
        """Computes l1 norm for each param in params and averages them."""
        return torch.stack([A.norm(p=norm) for A in self.A_seq_]).mean()

    def store_old_weights(self):
        """Save current weights."""
        self.old_A_seq_ = {}
        for t in range(self.n_seq_):
            self.old_A_seq_[t] = self.A_seq_[t].data.clone()

    def roll_back_weights(self):
        """In case weights are frozen, we roll back to pre-step frozen
        weights.
        """
        for t, frozen in enumerate(self._freeze_update):
            if frozen:
                self.A_seq_[t].data = self.old_A_seq_[t]

    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(
            monitor='loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=True,
            filename=('{{epoch:02d}}-{{loss:.2f}}'),
        )
        return checkpoint
