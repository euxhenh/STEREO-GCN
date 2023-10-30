from typing import List

import numpy as np
import torch
import torch.nn as nn
from regain.covariance import TimeGraphicalLasso
from sklearn.utils.validation import check_consistent_length
from torch import BoolTensor, FloatTensor

from ..data.datasets import DatasetMixin
from ..utils.nn_ops import init_weights
from ..utils.search import search_nonlin_fn
from ._tvdbn import tvdbn_functional
from .gnn_fixed_w import GCNConv_Fixed_W


class TVDBN:
    r"""Time-Varying Dynamic Bayesian Networks.

    Based on "Time-Varying Dynamic Bayesian Networks" by Le Song, Mladen
    Kolar, Eric Xing, Neurips proceedings 2009.

    A linear dynamics model defined as a weighted regression problem
    optimized via the shooting algorithm.

    Parameters
    ----------
    reg_lambda: float
        Regularization parameter lambda used for the L1 penalty.
    kernel_bandwidth: float, None
        The RBF kernel bandwidth h: $exp(-t^2 / h)$.
    tol: float
        Tolerance. Used to stop training when weights update by less than
        this amount.

    Attributes
    ----------
    n_nodes_: int
        Number of nodes seen during train.
    n_seq_: int
        Number of time points seen during fit. This will be one less than
        the 2nd dimension of X.
    A_seq_: list[array] of length n_seq + 1.
        List of learned adjacency matrices, of shape (n_nodes_, n_nodes_).
        The zero-th A is random.
    """

    def __init__(
        self,
        reg_lambda: float = 0.01,
        kernel_bandwidth: float | None = None,
        tol: float = 0.05,
    ):
        self.reg_lambda = reg_lambda
        self.kernel_bandwidth = kernel_bandwidth
        self.tol = tol

    def fit(self, X, *, A0=None, sources_mask=None):
        """Learn the adjacency matrices.

        Adjacency matrices are in "target_to_source" format. I.e., A[i, j]
        corresponds to the weight for the edge (j -> i).

        Parameters
        ----------
        X: array of shape (n_nodes, n_seq + 1) or torch.Dataset
            The data array. Note, each node has 1 feature per time point.
            The plus one for n_seq is present since we try to predict the
            next time point from the current once, hence, one time point is
            lost for A. If `X` is a temporal dataset, will convert it to a
            data matrix `X` by concatenating the x in x_seq and taking the
            mean of the features for each sample.
        A0: array of shape (n_nodes, n_nodes)
            The first adjacency matrix warm start. If None, will initialize
            at random from a uniform (0, 1).
        """
        if not isinstance(X, np.ndarray):
            X = self.dataset_to_x(X)
        if sources_mask is not None and isinstance(sources_mask, torch.Tensor):
            sources_mask = sources_mask.numpy()

        assert X.ndim == 2
        self.n_nodes_ = X.shape[0]
        self.n_seq_ = X.shape[1] - 1
        if self.kernel_bandwidth is None:
            self.kernel_bandwidth = self.n_nodes_ // 3

        A0 = A0 if A0 is not None else self._init_A(self.n_nodes_)
        A_seq_ = [A0]
        A_seq_.extend([np.zeros_like(A0) for _ in range(self.n_seq_)])
        A_seq_ = np.stack(A_seq_)
        self.A_seq_ = tvdbn_functional(
            X, A_seq_,
            kernel_bandwidth=self.kernel_bandwidth,
            reg_lambda=self.reg_lambda,
            tol=self.tol,
            sources_mask=sources_mask,
        )
        self.A_seq_ = list(self.A_seq_)
        _ = self.A_seq_.pop(0)

    def fit_predict(self, X, *, A0=None, sources_mask=None):
        self.fit(X, A0=A0, sources_mask=sources_mask)
        return self.A_seq_

    def dataset_to_x(self, dataset: DatasetMixin):
        """Helper function to convert a temporal dataset to X."""
        if not hasattr(dataset, 'x_seq_full'):
            raise ValueError("A temporal dataset should have the `x_seq_full` key.")
        x_seq = dataset.x_seq_full
        x_seq = [x.mean(1) for x in x_seq]  # use mean expression as feature
        x_seq = [x.numpy() if isinstance(x, torch.Tensor) else x for x in x_seq]
        return np.stack(x_seq, -1)

    @staticmethod
    def _init_A(n_nodes) -> np.ndarray:
        return np.random.random((n_nodes, n_nodes))


class TVGL:
    """Time-Varying Graphical Lasso.

    Paper
    Hallac, David, et al. Network Inference via the Time-Varying
    Graphical Lasso. arXiv, 9 June 2017. arXiv.org,
    https://doi.org/10.48550/arXiv.1703.01958.

    Based on the implementation by regain.
    https://github.com/fdtomasi/regain/

    Parameters
    ----------
    n_sources: int
        Since we are learning undirected graphs, we could pick the top
        connected `n_sources` sources and set all other sources to 0.
    """

    def __init__(self, n_sources: int | None = None, **kwargs):
        self.n_sources = n_sources
        self.tgl = TimeGraphicalLasso(**kwargs)

    def fit(self, X, y=None, *, sources_mask_seq=None):
        """Learn covariance of features."""
        if not isinstance(X, np.ndarray):
            X, y = self.dataset_to_x(X)

        self.tgl.fit(X, y)
        self.A_seq_ = list(self.tgl.precision_.copy())

        if self.n_sources is not None:
            assert sources_mask_seq is None  # Can only provide one or the other
            for A in self.A_seq_:
                top = (A != 0).sum(0).argsort()[-self.n_sources:]
                A[:, ~np.in1d(np.arange(A.shape[1]), top)] = 0

        if sources_mask_seq is not None:
            for A, sources_mask in zip(self.A_seq_, sources_mask_seq):
                A[:, ~sources_mask] = 0

    def dataset_to_x(self, dataset: DatasetMixin):
        """Convert torch dataset to array."""
        if hasattr(dataset, 'tvgl_prep'):
            x_seq, y = dataset.tvgl_prep()
        else:
            x_seq = np.hstack(dataset.x_seq).T
            y = np.concatenate([np.full(dataset.in_features, t) for t in range(dataset.n_seq)])
        return x_seq, y


class DeepAutoreg(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        in_features: int,
        out_features: int | None = None,
        *,
        sources_mask: np.ndarray | BoolTensor | None = None,
        n_linear_layers: int = 1,
        nonlin: str = 'LeakyReLU',
        init_A: str = 'trunc_normal',
        fillval: float = 1.0,
        **kwargs,
    ):
        """Simple block that performs nonlin(A @ X) followed by Linear layers.

        Parameters
        ----------
        n_nodes: int
            Number of nodes in the graph.
        in_features: int
            Number of node features.
        out_features: int
            Number of output features.
        sources_mask: array
            Boolean mask denoting source nodes.
        n_linear_layers: int
            Number of Linear layers to stack on top of A@X operation.
        nonlin: str
            The nonlineary to use. Will search `torch.nn` and `torch`.
        init_A: str
            Initialization to use for the adjacency matrix. Can be
            'glorot', 'trunc_normal' or 'fill'.
        fillval: float
            Only used if `init_A` is 'fill'.
        """
        super().__init__()

        out_features = out_features or in_features

        self.sources_mask = sources_mask
        n_sources = sources_mask.sum() if sources_mask is not None else n_nodes
        self.A = init_weights(nn.Parameter(torch.Tensor(n_nodes, n_sources)),
                              init_edge_weights=init_A,
                              fillval=fillval)
        self.W = nn.Parameter(torch.Tensor(in_features, in_features))
        self.gcn = GCNConv_Fixed_W(
            in_channels=in_features,
            out_channels=in_features if n_linear_layers > 0 else out_features,
            **kwargs,
        )

        if n_linear_layers > 0:
            self.nonlin = search_nonlin_fn(nonlin)
            self.dense = nn.ModuleList()
            for i in range(n_linear_layers):
                self.dense.add_module(
                    f"linear_{i}", nn.Linear(in_features, out_features, bias=False)
                )
        else:
            self.register_parameter('nonlin', None)
            self.register_parameter('dense', None)

        nn.init.xavier_uniform_(self.W)

    def forward(self, x: FloatTensor) -> FloatTensor:
        out = self.gcn(self.W, x, self.A, self.sources_mask)
        if self.dense is not None:
            for layer in self.dense:
                out = layer(self.nonlin(out))
        return out

    def l1(self) -> FloatTensor:
        """Return L1 norm of A."""
        return self.A.norm(p=1)

    @torch.no_grad()
    def clip_A(self, min_val: float = 0.0) -> None:
        self.A.clamp_min_(min_val)


class TemporalDeepAutoreg(nn.Module):
    """Separate DeepAutoreg for each time point.

    Parameters
    ----------
    n_nodes_seq: List[int]
        Number of nodes for each time point.
    sources_mask_seq: List[array-like]
        ID's of the source nodes for each time point. If provided, will
        only use (train) these as source nodes.
    kwargs: Dict
        Will be passed to each DeepAutoreg.
    """

    def __init__(
        self,
        n_nodes_seq: List[int],
        sources_mask_seq: List[np.ndarray | BoolTensor] | None = None,
        **kwargs,
    ):
        super().__init__()

        if sources_mask_seq is None:
            sources_mask_seq = [None] * len(n_nodes_seq)

        self.blocks = nn.ModuleList()
        for i, (n_nodes, sources_mask) in enumerate(zip(n_nodes_seq, sources_mask_seq)):
            block = DeepAutoreg(n_nodes=n_nodes, sources_mask=sources_mask, **kwargs)
            self.blocks.add_module(f'unit_{i}', block)

    @property
    def n_seq_(self) -> int:
        """Return number of time points."""
        return len(self.blocks)

    @property
    def A_seq_(self) -> List[FloatTensor]:
        """Return a list of adjacency matrices."""
        return [block.A for block in self.blocks]

    def forward(self, x_seq: List[FloatTensor]) -> List[FloatTensor]:
        check_consistent_length(x_seq, self.blocks)
        out = [block(x) for block, x in zip(self.blocks, x_seq)]
        return out

    def l1s(self, reduction: str = 'mean') -> FloatTensor:
        """Return mean L1 norm of adjacency matrices."""
        norms = torch.stack([block.l1() for block in self.blocks])
        return norms.mean() if reduction == 'mean' else norms.sum()

    @torch.no_grad()
    def clip_As(self, min_val: float = 0.0) -> None:
        for block in self.blocks:
            block.clip_A(min_val)
