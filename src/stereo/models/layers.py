from typing import Any, Dict, List, Literal

import torch
import torch.nn as nn
from sklearn.utils.validation import check_consistent_length
from torch_geometric.nn import SAGPooling, TopKPooling
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import dense_to_sparse

from .gnn_fixed_w import GCNConv_Fixed_W


class EGCHUnit(nn.Module):
    """The EvolveGCN model from https://arxiv.org/abs/1902.10191.
    We implement the H version given by the following update rules:

    forall t
    - Input: A_t, H_t^l, W_{t-1}.
    - Output:
        W_t = RNN(H_t, W_{t-1})
        H_t^{l+1} = GNN(A_t, H_t^l, W_t).

    Here t denotes the time point, and l is the l-th layer. In other
    words, we use an RNN (such as GRU) to evolve the GNN weights.

    This layer takes a sequence of A's and X's and outputs another sequence
    of tilde{X}'s of equal length. These are the high-level
    representations of the X's. Note, H plays the role of X, and A is given
    by an edge_index and edge_weight pair. Optionaly, the layer takes a
    sequence of integer lists corresponding to rows of X that will be
    masked out (zeroed).

    Parameters
    ----------
    in_features: int
        The number of features in X_t.
    out_features: int
        The number of output features in tilde{X}_t. If None, this will be
        set equal to in_features. If stacking multiple EGCHUnit's, it may
        be preferrable to leave this as None.
    rnn_kwargs: dict
        Kwargs to pass to rnn. By default, we set num_layers=1.
    gnn_kwargs: dict
        Kwargs to pass to gnn.
    pool_type: str
        Type of Pooling to use.
    """

    def __init__(
        self,
        in_features: int,
        *,
        out_features: int | None = None,
        rnn_kwargs: Dict[str, Any] | None = None,
        gnn_kwargs: Dict[str, Any] | None = None,
        pool_type: Literal['SAGPooling', 'TopKPooling'] = 'TopKPooling',
    ):
        super(EGCHUnit, self).__init__()
        # If out_features not specified, keep same dims as input
        out_features = in_features if out_features is None else out_features
        # Compress n_samples into out_features
        pool = SAGPooling if pool_type == 'SAGPooling' else TopKPooling
        self.pool = pool(in_channels=in_features, ratio=out_features)

        rnn_kwargs = rnn_kwargs or {}
        rnn_num_layers = rnn_kwargs.setdefault('num_layers', 1)

        self.rnn: nn.Module = nn.GRU(
            input_size=in_features, hidden_size=out_features, **rnn_kwargs,
        )

        self.gnn: nn.Module = GCNConv_Fixed_W(
            in_channels=in_features, out_channels=out_features,
            **(gnn_kwargs or {}),
        )

        w_shape = (rnn_num_layers, out_features, out_features)
        self.W0 = nn.Parameter(torch.Tensor(*w_shape))
        self.reset_params()

    def reset_params(self) -> None:
        glorot(self.W0)

    def forward(
        self,
        x_seq: List[torch.FloatTensor],
        A_seq: List[torch.FloatTensor],
        sources_mask_seq: List[torch.BoolTensor] | None = None,
    ) -> List[torch.FloatTensor]:
        """Forward pass

        Parameters
        ----------
        x_seq: List[FloatTensor of shape (num_nodes_t, num_features_t)]
            The node features for each time step t. Alias for H.
        A_seq: List[torch.FloatTensor of shape (num_nodes_t, num_nodes_t)]
        or shape (num_nodes_t, num_sources_t).
            A sequence of (weighted) adjacency matrices.
            NOTE: A is assumed to be in 'target_to_source' mode.
        sources_mask_seq: List[torch.BoolTensor], None
            The mask of source indices if A is not square.

        Returns
        -------
        out: List[FloatTensor of shape(num_nodes_(t+1), num_features_(t+1))]
            Node embeddings for each time step t for the next layer
        """
        W = self.W0
        x_out = []

        check_consistent_length(x_seq, A_seq, sources_mask_seq)
        if sources_mask_seq is None:
            sources_mask_seq = [None] * len(x_seq)

        for x, A, sources_mask in zip(x_seq, A_seq, sources_mask_seq):
            # transpose since A is in 'target_to_source' mode
            edge_index, edge_weight = dense_to_sparse(A.T)
            # Summarize node embeddings into k representative vectors.
            # x_hat = self.pool(x, edge_index, edge_weight)[0]
            x_hat = self.pool(x, edge_index, edge_weight)[0]
            # Evolve weights to get W_{t+1} from W_t. Expand dim 1 so that
            # we have (1 time sequence, batch_size, features).
            _, W = self.rnn(x_hat[None, :, :], W)
            # Use evolved weights to get node embeddings for next layer.
            x = self.gnn(W=W[-1], x=x, A=A, sources_mask=sources_mask)
            x_out.append(x)

        return x_out
