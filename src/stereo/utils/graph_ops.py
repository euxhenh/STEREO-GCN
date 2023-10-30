import numpy as np
from torch_geometric.utils import to_dense_adj


def adj_from_edge_index(n_nodes, edge_index, weights=None,
                        as_pt=False, device=None, mode='source_to_target'):
    """Given edge begin and end points, construct a square matrix.
    """
    dense_t = to_dense_adj(
        edge_index=edge_index,
        edge_attr=weights,
        max_num_nodes=n_nodes,
    ).to(device or edge_index.device)
    if not as_pt:
        dense_t = dense_t.numpy()

    if mode == 'source_to_target':
        return dense_t
    elif mode == 'target_to_source':
        return dense_t.T
    else:
        raise ValueError(f"Could not understand mode `{mode}`")


def sparsity(A):
    """Compute the fraction of zeros in A."""
    n_zeroes = (A == 0).sum()
    n_elements = np.prod(A.shape)
    return n_zeroes / n_elements
