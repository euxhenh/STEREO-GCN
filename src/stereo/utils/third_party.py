# Source: LassoNet, https://lassonet.ml

import torch
from sklearn.utils import check_consistent_length
from torch.nn import functional as F

from .nn_ops import sign_wo_zero, soft_threshold


def hier_prox(theta, W, *, soft_lambda, M=10):
    """Hierarchical Proximal Operator.
    Source: https://lassonet.ml
    Paper: https://jmlr.org/papers/volume22/20-848/20-848.pdf

    Sparsity is applied to columns (sources) of W
    """
    W = W.T  # Transpose since the code below sparsifies rows

    scalar_theta = len(theta.shape) == 0
    one_d_W = len(W.shape) == 1
    if scalar_theta:
        theta = theta.unsqueeze(0)
    elif len(theta.shape) != 1:
        raise ValueError("Hier-Prox implemented only for 1D theta.")
    if one_d_W:
        W = W.unsqueeze(0)
    check_consistent_length(theta, W)

    k = W.shape[1]
    W_sorted = torch.sort(W.abs(), dim=1, descending=True).values  # (d, k)
    W_cumsum = F.pad(W_sorted.cumsum(dim=1), (1, 0), value=0)  # (d, k + 1)
    wm = soft_threshold(theta[:, None].abs() + M * W_cumsum, soft_lambda)  # (d, k + 1)
    wm = M / (1.0 + torch.arange(k + 1).to(wm) * M**2)[None, :] * wm  # (d, k + 1)

    W_padded = F.pad(W_sorted, (1, 0), value=float('inf'))
    W_padded = F.pad(W_padded, (0, 1), value=0)  # (d, k + 2)
    m_tilde = (
        (W_padded[:, 1:] <= wm) & (wm <= W_padded[:, :-1])
    ).float().argmax(dim=1, keepdim=True)
    wm_tilde = torch.take_along_dim(wm, m_tilde, dim=1)[:, 0]

    theta_tilde = 1 / M * sign_wo_zero(theta) * wm_tilde
    W_tilde = sign_wo_zero(W) * torch.min(wm_tilde[:, None], W.abs())

    theta_tilde = theta_tilde.squeeze() if scalar_theta else theta_tilde
    W_tilde = W_tilde.squeeze(dim=0) if one_d_W else W_tilde
    return theta_tilde, W_tilde.T  # switch back to columns


def estimate_lambda(theta, W, M=1, factor=2):
    """Estimate when the model will start to sparsify."""
    W = W.T

    def is_sparse(lambda_):
        nonlocal theta, W
        for _ in range(10000):
            new_theta, W = hier_prox(theta, W, soft_lambda=lambda_, M=M)
            if torch.abs(theta - new_theta).max() < 1e-5:
                break
            theta = new_theta
        return (torch.norm(theta, p=2, dim=0) == 0).sum()

    start = 1e-6
    while not is_sparse(factor * start):
        start *= factor
    return start
