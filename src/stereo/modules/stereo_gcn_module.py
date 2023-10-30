import logging
import os
from collections import defaultdict
from functools import cached_property, partial
from itertools import pairwise, starmap
from typing import Any, Callable, Dict, List, Literal, Tuple, TypedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.utils.validation import check_consistent_length
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, optim

from .. import models
from ..utils.graph_ops import sparsity
from ..utils.nn_ops import init_weights, soft_threshold
from ..utils.search import search_loss_fn, search_optim_fn
from .callbacks import ASeqLogger, HierProx

logger = logging.getLogger("pytorch_lightning.core")

A_SEQ = List[FloatTensor]


class UnitKwargs(TypedDict):
    """Type annotations for Block parameters."""
    batch_norm: bool
    dropout: float
    nonlin: str
    rnn_kwargs: Dict[str, Any] | None
    gnn_kwargs: Dict[str, Any] | None
    pool_type: Literal['SAGPooling', 'TopKPooling']


class ModelKwargs(TypedDict):
    """Type annotations for model parameters."""
    in_features: int
    out_features: int
    add_linear_layer: bool
    hidden_size: int | None
    n_blocks: int
    final_block_out_features: int | None
    add_input_skip_connection: bool
    unit_kwargs: UnitKwargs


class AdjKwargs(TypedDict):
    """Type annotations for parameters used to initialize A_seq."""
    rank: int | None
    init_edge_weights: str
    fillval: float


class HierProxKwargs(TypedDict):
    """Type annotations for hier-prox callback."""
    min_n_nonzero: int
    soft_lambda: float | None
    M: float
    path_multiplier: float
    warm_epochs: int
    run_every_n: int
    fillval: float


class STEREO_GCN_Module(pl.LightningModule):
    """A base module for EGCH nets with unknown graphs.

    Inner optimization updates model weights, while outer optimization
    updates graph edge_weights. We optimize manually since pl calls both
    optimizers at the same time, but we need to run each step with a
    different number of epochs (one outer step for every `inner_loops`
    steps).

    1. Initialize model
    2. Initialize edge_weights for all time points.
    3. (If hier-prox) Initialize theta's for Hier-Prox to 1.0.
    == Training routine == For i=1, ..., n_epochs
        For inner_loops iterations
            4. Compute gradient w.r.t., EGCH weights W.
            5. Perform a gradient step on W.
        6. Compute gradient w.r.t., edge_weights.
        7. Add regularizers (L1 penalty or smoothness penalty).
        8. Perform a gradient step on edge_weights.
        9. (If hier-prox) Perform a Hier-Prox update.
        10. (Optionally) Clip edge_weights to min=0.

    Parameters
    ----------
    n_nodes_seq: List[int]
        List of integers corresponding to the number of nodes for each time
        point.
    model: str
        Name of model to use.
    model_kwargs: dict[str, Any]
        The Graph temporal model params.
    inner_loss_fn, outer_loss_fn: Callable or str
        A callable that returns the loss object, or a string used to obtain
        the loss constructor from torch.nn / torch.nn.functional. If
        `outer_loss_fn` is None, will use the same loss as `inner_loss_fn`.
    inner_loss_fn_kwargs, outer_loss_fn_kwargs: Dict[str, Any]
        Kwargs to pass to inner_loss_fn or outer_loss_fn. If
        `outer_loss_fn_kwargs` is None, will use `inner_loss_fn_kwargs`.
    global_l1_lambda: float
        The coefficient to use for L1 penalty during the outer optimization
        loop. This is applied to the entire graph. Will NOT be scaled by
        the learning rate during the Proximal step.
    max_sparsity: float in [0, 1]
        Max sparsity to allow for each A. Only used when not in hier-prox.
    smoothness_lambda: float
        The coefficient to use for the L2 penalty of consecutive temporal
        graphs. The L2 difference is averaged across all pairs.
    inner_optim_fn, outer_optim_fn: Callable or str
        A callable that returns the optimizer object, or a string used to
        obtain the optimizer constructor from torch.optim.
    inner_optim_kwargs, outer_optim_kwargs: Callable, str
        Arguments to pass to inner and outer optimizers.
    apply_meta_step: bool
        If False, will not run the outer loop.
    inner_loops: int
        The number of loops to run the inner optimization procedure before
        performing an outer optimization step.
    clip_A: bool
        If True, will clip A or U and V so that edge weights are
        non-negative.
    low_rank_mode: bool
        If True, will learn column matrices U and V such that U @ V = A,
        instead of learning A directly.
    adj_kwargs: Dict[str, Any]
        Arguments used to initialize A_seq.
    sources_mask_seq: List[array-like]
        ID's of the source nodes for each time point. If provided, will
        only use (train) these as source nodes.
    true_A_seq: List[array-like]
        The true adjacency matrices to use for evaluation metrics.
    clear_nan: bool
        If True, will replace nan's in A with zeros. NaN's can occur when a
        target node has no source nodes.
    apply_hier_prox: bool
        Whether to apply the hierarchical proximal operator to obtain
        row-sparse graphs. If True, will ignore `global_l1_lambda`.
    hier_prox_kwargs: dict
        Kwargs to pass to Hier-Prox.
    log_A_every_k_epochs: int
        Will log the adjacency matrices every k epochs. These will be
        handled by the logger.

    Attributes
    ----------
    model: nn.Module
        The temporal graph model.
    inner_criterion, outer_criterion: torch.losses
        Inner and outer losses, respectively.
    A_param_seq_: nn.ModuleDict
        Points to the adjacency matrices to be learned. If `low_rank_mode`
        is True, than this will contain two keys 'U' and 'V', otherwise,
        will contain a single key 'A'.
    n_seq_: int
        Number of time points.
    A_seq_: List[Tensor]
        The adjacency matrices. If `low_rank_mode` is True, this will
        compute U@V, otherwise return `self.A_param_seq_['A']`.
    W_seq_: List[Tensor]
        Returns the matrix that is to be penalized by hier-prox. This will
        be `A_seq_` if not in `low_rank_mode`, else will be
        `self.A_param_seq_['V']`.
    """

    def __init__(
        self,
        n_nodes_seq: List[int],
        *,
        model: str = "StackedEGCH",
        model_kwargs: ModelKwargs | None = None,
        # Losses
        inner_loss_fn: Callable | str = 'MSELoss',
        inner_loss_fn_kwargs: Dict[str, Any] | None = None,
        outer_loss_fn: Callable | str | None = None,
        outer_loss_fn_kwargs: Dict[str, Any] | None = None,
        global_l1_lambda: float = 0.0,
        max_sparsity: float = 0.95,
        smoothness_lambda: float = 0.0,
        # Optims
        inner_optim_fn: Callable | str = 'Adam',
        inner_optim_fn_kwargs: Dict[str, Any] | None,
        outer_optim_fn: Callable | str | None = None,
        outer_optim_fn_kwargs: Dict[str, Any] | None = None,
        # Meta-related
        apply_meta_step: bool = True,
        inner_loops: int = 5,
        clip_A: bool = False,
        # Graph-learning related
        low_rank_mode: bool = False,
        adj_kwargs: AdjKwargs | None = None,
        sources_mask_seq: List[np.ndarray | BoolTensor] | None = None,
        sources_names_seq: List[np.ndarray] | None = None,
        init_A_seq: List[FloatTensor] | None = None,
        true_A_seq: List[np.ndarray | FloatTensor] | None = None,
        clear_nan: bool = False,
        # Hier-prox related
        apply_hier_prox: bool = False,
        hier_prox_kwargs: HierProxKwargs | None = None,
        # Logging
        log_A_every_k_epochs: int | None = None,
        save_A_seq: bool = False,
    ):
        """Prefer class names and kwargs instead of objects so that wandb
        can save them as parameters.
        """
        super().__init__()
        # We will perform the optimization step manually since we have a
        # bi-level optimization procedure.
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["true_A_seq"])

        check_consistent_length(n_nodes_seq, sources_mask_seq, true_A_seq)
        self.n_nodes_seq = n_nodes_seq

        self.model: nn.Module = self.init_model(model, **(model_kwargs or {}))
        self._model_param_requires_grad: Dict[str, bool] = {
            name: param.requires_grad
            for name, param in self.model.named_parameters()
        }

        # losses and regularizers
        inner_loss_fn_kwargs = inner_loss_fn_kwargs or {}
        self.inner_criterion = search_loss_fn(inner_loss_fn, **inner_loss_fn_kwargs)
        self.outer_criterion = search_loss_fn(
            outer_loss_fn or inner_loss_fn,
            **(outer_loss_fn_kwargs or inner_loss_fn_kwargs)
        )
        self.global_l1_lambda = global_l1_lambda
        self.max_sparsity = max_sparsity
        self.smoothness_lambda = smoothness_lambda

        # optimizers
        self.inner_optim_fn = inner_optim_fn
        self.inner_optim_kwargs = inner_optim_fn_kwargs or {'lr': 0.01}
        self.lr = self.inner_optim_kwargs['lr']
        self.outer_optim_fn = outer_optim_fn
        self.outer_optim_kwargs = outer_optim_fn_kwargs

        # meta related
        self.apply_meta_step = apply_meta_step
        self.inner_loops = inner_loops
        self.clip_A = clip_A

        # hier-prox related
        self.apply_hier_prox = apply_hier_prox
        self.hier_prox_kwargs = hier_prox_kwargs

        # log edge weight history
        self.log_A_every_k_epochs = log_A_every_k_epochs
        self.save_A_seq = save_A_seq

        self.sources_mask_seq = sources_mask_seq
        self.sources_names_seq = sources_names_seq

        self.low_rank_mode = low_rank_mode
        adj_kwargs = adj_kwargs or {}
        self.A_param_seq_: nn.ModuleDict = self.init_A_param_seq(
            n_nodes_seq=n_nodes_seq,
            low_rank_mode=low_rank_mode,
            sources_mask_seq=sources_mask_seq,
            init_A_seq=init_A_seq,
            **adj_kwargs,
        )
        self.true_A_seq = true_A_seq
        self.clear_nan = clear_nan
        self._freeze_update = torch.full((self.n_seq_,), False)

    @property
    def n_seq_(self) -> int:
        """Returns the number of time points"""
        return len(self.n_nodes_seq)

    @property
    def A_seq_(self) -> A_SEQ:
        """Returns a list with the square adjacency matrices"""
        if self.low_rank_mode:
            return [U @ V for U, V in zip(*self.A_param_seq_.values())]
        return list(self.A_param_seq_.A)

    @property
    def W_seq_(self) -> List[FloatTensor]:
        """Returns the weights that will be col-penalized by hier prox.
        Row space of A == row space of U.
        Col space of A == col space of V.
        Penalizing columns (sources) of A means penalizing columns of V.
        """
        return getattr(self.A_param_seq_, 'V' if self.low_rank_mode else 'A')

    @property
    def n_sources_seq_(self) -> List[int]:
        """Return the number of sources for each time point."""
        if self.sources_mask_seq is None:
            return self.n_nodes_seq
        return list(map(np.sum, self.sources_mask_seq))

    @cached_property
    def hp(self) -> HierProx | None:
        """Return the HierProx callback object."""
        for callback in self.trainer.callbacks:
            if isinstance(callback, HierProx):
                return callback
        return None

    @property
    def all_frozen(self) -> bool:
        """Return True if all A_params should stop updating."""
        self_frozen = torch.all(self._freeze_update)
        return self.hp.all_frozen or self_frozen if self.hp is not None else self_frozen

    @property
    def freeze_update_(self) -> torch.BoolTensor:
        """i-th element True if i-th A_param stopped converging."""
        return (
            self.hp.freeze_update_ | self._freeze_update if self.hp is not None
            else self._freeze_update
        )

    def configure_optimizers(self) -> Tuple[optim.Optimizer, optim.Optimizer]:
        """Initialize inner optimizer. The inner optimizer updates model weights.
        """
        inner_optimizer = search_optim_fn(self.inner_optim_fn)(
            self.model.parameters(), **self.inner_optim_kwargs,
        )
        outer_optimizer = search_optim_fn(self.outer_optim_fn or self.inner_optim_fn)(
            self.A_param_seq_.parameters(),
            **(self.outer_optim_kwargs or self.inner_optim_kwargs),
        )
        return inner_optimizer, outer_optimizer

    @staticmethod
    def compute_loss(inputs, targets, criterion, reduction='mean'):
        """Computes the loss for every pair (a, b) in zip(inputs, targets)
        using criterion, then computes the mean/sum of the results.
        """
        if reduction not in ['mean', 'sum']:
            raise NotImplementedError(f"Reduction mode `{reduction}` not implemented.")
        losses = torch.stack(list(starmap(criterion, zip(inputs, targets))))
        return losses.mean() if reduction == 'mean' else losses.sum()

    def forward(self, x_seq, A_seq, **kwargs):
        """Predict given inputs."""
        return self.model(x_seq, A_seq, **kwargs)

    @torch.no_grad()
    def evaluate(self, y_seq, x_seq, A_seq=None, **kwargs):
        """Computes inner loss given the inputs. If A_seq is None, will use
        self.A_seq_.
        """
        out = self(x_seq, A_seq or self.A_seq_, **kwargs)
        inner_loss = self.compute_loss(out, y_seq, self.inner_criterion)
        return inner_loss.item()

    def on_train_start(self):
        # Print model summary
        logger.info(self)

    def training_step(self, batch, _):
        """Runs `self.inner_loops` of the inner step."""
        inner_optimizer, _ = self.optimizers()
        self.inner(True)  # Update `require_grad` params

        x_seq, y_seq = batch
        A_seq = self.A_seq_  # Cache U@V since edges are not updated here

        for _ in range(self.inner_loops):
            out = self(x_seq, A_seq, sources_mask_seq=self.sources_mask_seq)
            inner_loss = self.compute_loss(out, y_seq, criterion=self.inner_criterion)

            inner_optimizer.zero_grad()
            self.manual_backward(inner_loss)
            inner_optimizer.step()

        self.log("inner_loss", inner_loss, prog_bar=True)
        return {"inner_loss": inner_loss}

    def on_train_batch_end(self, outputs, batch, _):
        """Runs the outer step after finishing k steps of inner step. This
        is run for each batch.
        """
        if not self.apply_meta_step:
            return

        if self.all_frozen:
            self.stop_training("A_params stopped converging.")

        _, outer_optimizer = self.optimizers()
        self.inner(False)

        x_seq, y_seq = batch
        if not self.apply_hier_prox:
            self.store_old_weights()

        out = self(x_seq, self.A_seq_, sources_mask_seq=self.sources_mask_seq)
        outer_loss = self.compute_loss(out, y_seq, criterion=self.outer_criterion)
        outer_loss = self._add_regularizers(outer_loss)

        outer_optimizer.zero_grad()
        self.manual_backward(outer_loss)
        self.clear_grad_for_frozen_params()
        outer_optimizer.step()

        if self.clip_A:
            self._clip_A_seq()
        if self.clear_nan:
            self.clear_nan_in_A()

        if not self.apply_hier_prox:
            if self.global_l1_lambda:
                self.ista_step()
            self.roll_back_weights()  # roll back any frozen weights

        self.log('outer_loss', outer_loss, prog_bar=True)

    def on_train_end(self) -> None:
        """Print path of checkpoint."""
        self.trainer.checkpoint_callback.on_train_epoch_end(self.trainer, self)
        if not hasattr(self.trainer.checkpoint_callback, "last_model_path"):
            return
        path = self.trainer.checkpoint_callback.last_model_path
        path = os.path.abspath(path)
        print(f"Read last checkpoint from '{path}'")

    def _add_regularizers(self, outer_loss) -> Tensor:
        l0_loss = self._compute_norms(0)
        l1_loss = self._compute_norms(1)

        # Add smoothness loss only if using the same node set over time
        smoothness_loss = self._compute_smoothness_loss()
        if self.smoothness_lambda and np.unique(self.n_nodes_seq).size == 1:
            outer_loss = outer_loss + self.smoothness_lambda * smoothness_loss

        self.log_dict({
            'l0_loss': l0_loss,
            'l1_loss': l1_loss,
            'smoothness_loss': smoothness_loss,
        })
        return outer_loss

    def clear_nan_in_A(self):
        """Replace nan's with zeros in A_seq.

        nan's can happen when rows of A become 0, meaning there are no
        sources for that target, hence gradients are undefined. Here we sub
        these with zeros.
        """
        for W in self.W_seq_:
            W.data[torch.isnan(W)] = 0
        if self.low_rank_mode:
            for U in self.A_param_seq_.U:
                U.data[torch.isnan(U)] = 0

    @torch.no_grad()
    def ista_step(self):
        """Iterative soft-thresholding step.

        Proximal gradient descent for L1 penalty.
        """
        for i, W in enumerate(self.W_seq_):
            if self.freeze_update_[i]:
                continue

            data = soft_threshold(W.data, soft_lambda=self.global_l1_lambda)
            # Make sure to have at least one source per target
            # Pick the best source pre-step
            zero_rows = data.sum(1) == 0
            top_column = W.data[zero_rows].argmax(1)
            data[zero_rows][:, top_column] = 1e-4  # fill with small value

            if sparsity(data) > self.max_sparsity:
                self._freeze_update[i] = True
            else:
                W.data = data

    def _compute_norms(self, norm) -> Tensor:
        """Computes l1 norm for each param in params and averages them."""
        return torch.stack([A.norm(p=norm) for A in self.A_seq_]).mean()

    def _compute_smoothness_loss(self) -> float | Tensor:
        """NOTE: This assumes graphs have the same node set.
        """
        s_loss = 0.0
        if self.n_seq_ <= 1:
            return s_loss

        for a, b in pairwise(self.A_seq_):
            if a.shape != b.shape:
                if self.smoothness_lambda:
                    raise ValueError(
                        "Cannot compute smoothness loss with different node sets."
                    )
                else:
                    return 0.0
            s_loss = s_loss + torch.square(a - b).mean()
        return s_loss / (self.n_seq_ - 1)

    def store_old_weights(self):
        """Save current weights."""
        self.old_A_param_seq_ = defaultdict(dict)
        for t in range(self.n_seq_):
            if self.low_rank_mode:
                self.old_A_param_seq_['V'][t] = self.A_param_seq_.V[t].data.clone()
                self.old_A_param_seq_['U'][t] = self.A_param_seq_.U[t].data.clone()
            else:
                self.old_A_param_seq_['A'][t] = self.A_param_seq_.A[t].data.clone()

    def roll_back_weights(self):
        """In case weights are frozen, we roll back to pre-step frozen
        weights.
        """
        for t, frozen in enumerate(self.freeze_update_):
            if frozen:
                if self.low_rank_mode:
                    self.A_param_seq_.V[t].data = self.old_A_param_seq_['V'][t]
                    self.A_param_seq_.U[t].data = self.old_A_param_seq_['U'][t]
                else:
                    self.A_param_seq_.A[t].data = self.old_A_param_seq_['A'][t]

    def inner(self, mode: bool = True, /) -> None:
        """Switches between inner mode, i.e., if inner sets requires_grad_
        to True for model params and A to False.

        Only updates parameters which require grad.
        """
        for name, param in self.model.named_parameters():
            if self._model_param_requires_grad[name]:
                param.requires_grad_(mode)
        # Updates A or V
        for t, param in enumerate(self.W_seq_.parameters()):
            param.requires_grad_(not mode and not self.freeze_update_[t])
        # Also updates U if in low rank mode
        if self.low_rank_mode:
            for t, param in enumerate(self.A_param_seq_.U.parameters()):
                param.requires_grad_(not mode and not self.freeze_update_[t])

    def clear_grad_for_frozen_params(self) -> None:
        """Clears gradients for frozen params.
        """
        for t, frozen in enumerate(self.freeze_update_):
            if frozen:
                if self.low_rank_mode:
                    self.A_param_seq_.V[t].grad = None
                    self.A_param_seq_.U[t].grad = None
                else:
                    self.A_param_seq_.A[t].grad = None

    @torch.no_grad()
    def _clip_A_seq(self, min_val: float = 0.0) -> None:
        for param in self.A_param_seq_.parameters():
            param.clamp_min_(min_val)

    def stop_training(self, msg=''):
        """TODO: Find a graceful way to stop training."""
        # manually call 'on_train_end' since it is not run on interrupt
        self.on_train_end()
        raise InterruptedError(f"Stopping training... {msg}")

    @staticmethod
    def init_model(model: str, **model_params) -> nn.Module:
        """Initializes a model with the given params."""
        return getattr(models, model)(**model_params)

    @staticmethod
    def init_A_param_seq(
        n_nodes_seq: List[int],
        *,
        low_rank_mode: bool = False,
        sources_mask_seq: List[List[int] | np.ndarray | LongTensor] | None = None,
        init_A_seq: List[FloatTensor] | None = None,
        rank: int | None = None,
        init_edge_weights: str = 'glorot',
        fillval: float = 1.0,
    ) -> nn.ModuleDict:
        """Initializes the graph edge weights.

        Parameters
        ----------
        n_nodes_seq: List[int]
            Number of nodes/samples per time point. A.shape[0] ==
            A.shape[1] == n_nodes_seq[i].
        low_rank_mode: bool
            If True, will assume a decomposition of A = U @ V where U and V
            are low rank rectangular matrices.
        sources_mask_seq: List[array]
            List of bool arrays with True representing sources.
        init_A_seq: List[array]
            Initializations for A_seq. Assumes full square even if sources
            mask seq is provided.
        rank: int
            The rank estimate for A. U.shape[1] == V.shape[0] == adj_rank.
        init_edge_weights: str
            The strategy for initializing edge weights. Can be any of
            'trunc_normal', 'glorot', 'fill'.
        fillval: float
            Initial fill value of tensors. Will be overridden if
            init_edge_weights != 'fill'.
        """
        if init_A_seq is not None:
            logger.info("Warm start")
            assert not low_rank_mode
            if sources_mask_seq is None:
                A_param_seq = init_A_seq
            else:
                assert len(init_A_seq) == len(sources_mask_seq)
                A_param_seq = [
                    A[:, sources_mask]
                    for A, sources_mask in zip(init_A_seq, sources_mask_seq)
                ]
            return nn.ModuleDict({'A': nn.ParameterList(A_param_seq)})

        init = partial(
            init_weights,
            init_edge_weights=init_edge_weights,
            fillval=fillval,
        )
        # NOTE: A is constructed in a 'target_to_source' fashion. I.e.,
        # A[i, j] denotes an edge from node j to node i.

        # By default assume every node can be a source
        if sources_mask_seq is None:
            n_sources_seq = n_nodes_seq
        else:
            n_sources_seq = [sources_mask.sum() for sources_mask in sources_mask_seq]

        if low_rank_mode:
            # Assume low rank A = U @ V for rectangular U and V.
            U_seq = [init(Tensor(n_nodes, rank)) for n_nodes in n_nodes_seq]
            # Col(A) == Col(V) so sources are columns of V.
            V_seq = [init(Tensor(rank, n_sources)) for n_sources in n_sources_seq]
            A_param_seq = nn.ModuleDict({'U': nn.ParameterList(U_seq),
                                         'V': nn.ParameterList(V_seq)})
        else:
            A_param_seq = [
                init(Tensor(n_nodes, n_sources))
                for n_nodes, n_sources in zip(n_nodes_seq, n_sources_seq)
            ]
            A_param_seq = nn.ModuleDict({'A': nn.ParameterList(A_param_seq)})

        return A_param_seq

    @property
    def _is_last_epoch(self) -> bool:
        return self.current_epoch == self.trainer.max_epochs - 1  # type: ignore

    @property
    def A_has_nan(self):
        return torch.any(Tensor([torch.any(torch.isnan(A)) for A in self.A_seq_]))

    def load_best_weights(self):
        ckpt_path = self.trainer.checkpoint_callback.best_model_path
        logger.info(f"Loading best model weights from {ckpt_path}.")
        model = type(self).load_from_checkpoint(ckpt_path)
        return model

    def configure_callbacks(self):
        """Automatically setup any callbacks."""
        callbacks = []

        loss_to_monitor = "outer_loss" if self.apply_meta_step else "inner_loss"
        checkpoint = ModelCheckpoint(
            monitor=loss_to_monitor,
            mode='min',
            save_top_k=3 if not self.apply_hier_prox else 0,
            save_last=True,
            auto_insert_metric_name=True,
            filename=(f'{{epoch:02d}}-{{{loss_to_monitor}:.2f}}'),
        )
        callbacks.append(checkpoint)
        if self.apply_hier_prox:
            hier_prox = HierProx(
                params_key='W_seq_',
                clip_A=self.clip_A,
                device=self.device,
                **(self.hier_prox_kwargs or {}),
            )
            callbacks.append(hier_prox)
        callbacks.append(ASeqLogger(self.log_A_every_k_epochs, self.save_A_seq))
        return callbacks
