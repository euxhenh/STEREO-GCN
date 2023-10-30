import torch
import torch.nn as nn


class GCNConv_Fixed_W(nn.Module):
    """GCNConv Layer that works with adjacency matrices A.

    A[j, i] denotes an edge from node i to node j.
    """

    def __init__(
        self,
        normalize: bool = False,
        add_self_loops: bool = False,
        in_channels: int | None = None,  # needed for repr only
        out_channels: int | None = None,  # needed for repr only
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.add_self_loops = add_self_loops

    def forward(
        self,
        W: torch.FloatTensor,
        x: torch.FloatTensor,
        A: torch.FloatTensor,
        sources_mask: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:
        """Forward pass. Computes A @ x @ W.

        If A is not square, must also provide `sources_mask`. If
        sources_mask[i] is True, then A[i] is the source node.

        Parameters
        ----------
        W: FloatTensor
            The weight for the linear layer of shape (in_features,
            out_features).
        x: FloatTensor
            The feature matrix of shape (n_nodes, in_features).
        A: FloatTensor
            The adjacency matrix of shape (n_nodes, n_nodes) or (n_nodes,
            n_sources). In the latter case, must also provide a boolean
            tensor of sources.
        sources_mask: BoolTensor, None
            A mask of sources such that A[sources_mask] is a square matrix
            corresponding to sources nodes only. The number of souces
            should equal A.shape[1].
        """
        if A.shape[0] != A.shape[1]:
            assert sources_mask.sum() == A.shape[1]

        if self.add_self_loops:
            A = A.clone()
            if sources_mask is None:
                _n = A.shape[0]
                A[torch.arange(_n), torch.arange(_n)] += 1.0
            else:
                A.data[sources_mask, torch.arange(sources_mask.sum())] += 1.0

        if self.normalize:
            if A.min() < 0:
                raise ValueError(
                    "Found negative values in adjacency matrix. "
                    "Cannot normalize."
                )

            diag = torch.sqrt(A.sum(1))
            diag.data[diag > 0] = 1 / diag[diag > 0]
            # diag.data[diag == 0] += 1e-8  # add small eps for grad
            D = torch.diag(diag)
            if sources_mask is None:
                A = D @ A @ D  # (n_nodes, n_nodes)
            else:
                A = D @ A @ D[sources_mask][:, sources_mask]  # (n_nodes, n_nodes)

        if sources_mask is not None:
            x = x[sources_mask]
        return A @ x @ W  # (n_nodes, out_features)

    def __repr__(self) -> str:
        if self.in_channels is not None and self.out_channels is not None:
            return (f'{self.__class__.__name__}({self.in_channels}, '
                    f'{self.out_channels})')
        return f'{self.__class__.__name__}()'
