"""Time Varying Dynamical Bayesian Networks using numba.
"""

import numpy as np
from numba import njit


@njit
def soft_thresh(S: float, b: float, reg_lambda: float) -> float:
    """Weight update using thresholding as in L1-regularized LS."""
    return (reg_lambda < np.abs(S)) * (np.sign(S - reg_lambda) * reg_lambda - S) / b


@njit
def converged(x, y, tol=1e-3) -> bool:
    """Return True if max change between x and x_old is less than self.tol.
    """
    if y is None:
        return False
    return np.linalg.norm(x - y, np.inf) < tol


@njit
def rbf(x, kernel_bandwidth):
    """Gaussian RBF kernel."""
    return np.exp(-x**2 / kernel_bandwidth)


@njit
def w_rbf(t_star, t, time_range: np.ndarray, kernel_bandwidth: float):
    r"""Weighting of an observation from time $t$ when we estimate the
    network at time $t^*$.

    Defined as
    $$weighted_kernel_{t^*}(t) = \frac{K_h(t-t^*)}{\sum_{t=1}^T
    K_h(t-t^*)}$$

    Parameters
    ----------
    t_star: int
        t^*. A given time point in [1, T].
    t: int or array of int
        The time points to compute the weighted kernel for.
    time_range: array
        Range of all time points.
    """
    return (
        rbf(t - t_star, kernel_bandwidth) /
        rbf(time_range - t_star, kernel_bandwidth).sum()
    )


@njit
def tvdbn_functional(
    X: np.ndarray,
    A_seq_: np.ndarray,
    *,
    kernel_bandwidth: float,
    reg_lambda: float,
    sources_mask: np.ndarray | None = None,
    tol: float = 1e-3,
):
    """Fast functional variant using numba."""
    n_nodes_, n_seq_ = X.shape
    n_seq_ -= 1
    # A_seq_ has 1 more A than number of time points
    assert len(A_seq_) == n_seq_ + 1
    time_range = np.arange(1, n_seq_ + 1)
    sources = (
        np.argwhere(sources_mask).flatten()
        if sources_mask is not None
        else np.arange(n_nodes_)
    )
    print(f"Using {len(sources)}/{n_nodes_} sources")

    for i in range(n_nodes_):  # For every target node
        for t_star in range(1, n_seq_ + 1):  # For every time point
            # scaling_factor[t - 1] = sqrt(w_{t^*}(t)) for t = 1, ..., T
            scaling_factor = np.sqrt(w_rbf(
                t_star, time_range, time_range, kernel_bandwidth
            ))
            # xtilde_i is shifted to the right by 1 and does not hold a
            # factor for x_i[0] (at time 0).
            # xtilde_i[t - 1] = scaling_factor[t - 1] * x_i[t]
            # for t = 1, ..., T.
            xtilde_i = scaling_factor * X[i, 1:]

            # xtilde does not hold a factor for x_i[T].
            # of shape (n_nodes_, n_seq_)
            xtilde = scaling_factor * X[:, :-1]

            # A^{t^*} of shape (n_nodes_, n_nodes_)
            A_ts = A_seq_[t_star]  # A at current time point
            # Warm start i-th node for A from the previous time point.
            # A_i^{t^*} of shape (n_nodes_,)
            A_ts[i, :] = A_seq_[t_star - 1][i, :]

            b = 2 / n_seq_ * (xtilde ** 2).sum(1)

            A_tmp = None
            while not converged(A_ts[i], A_tmp, tol):
                A_tmp = A_ts[i].copy()

                # Cache for faster computation
                ins_all_j = A_ts[i][:, None] * xtilde
                inner_s_no_ins = ins_all_j + xtilde_i
                # (1, n_nodes_) @ (n_nodes, n_seq_)
                ins = (A_ts[np.array([i])] @ xtilde).ravel()  # length n_seq_

                # instead of using all nodes, use source nodes. This is
                # equivalent if we only care about sources, since all other
                # entries in A would be 0.
                for j in sources:
                    inner_s = ins - inner_s_no_ins[j]
                    Sj = 2 / n_seq_ * np.dot(inner_s, xtilde[j])
                    new_weight = soft_thresh(Sj, b[j], reg_lambda)
                    ins += (new_weight - A_ts[i, j]) * xtilde[j]
                    A_ts[i, j] = new_weight

        print(f"Finished step {i + 1}/{n_nodes_}")

    return A_seq_
