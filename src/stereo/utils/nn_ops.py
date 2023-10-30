from typing import List

import numpy as np
import torch
import torch.nn as nn
from numpy.testing import assert_allclose


def soft_threshold(x, soft_lambda=0):
    return torch.sign(x) * torch.relu(torch.abs(x) - soft_lambda)


def sign_wo_zero(x):
    return torch.where(x >= 0, 1, -1)


def reset_model_weights(model: nn.Module):
    """Reinitializes all model layers recursively.
    """
    @torch.no_grad()
    def reset_fn(m: nn.Module):
        reset_parameters = getattr(m, 'reset_parameters', None)
        if callable(reset_parameters):
            m.reset_parameters()  # type: ignore

    # This applies fn recursively to every submodule.
    model.apply(fn=reset_fn)


def add_noise_to_model_weights(model: nn.Module, mean: float = 0.0, std: float = 0.1):
    """Adds gaussian noise to model weights.
    """
    @torch.no_grad()
    def add_noise_fn(m: nn.Module):
        for param in m.parameters():
            if isinstance(param, nn.Parameter):
                param.add_(torch.randn(param.size()) * std + mean)

    model.apply(fn=add_noise_fn)


def mask_rows(x: torch.Tensor,
              rows_to_mask: List[int] | torch.Tensor | np.ndarray | None = None):
    """Zeros out the rows specified in rows_to_mask in x.
    """
    if rows_to_mask is not None:
        mask = torch.ones_like(x)
        if isinstance(rows_to_mask, np.ndarray):
            rows_to_mask = torch.from_numpy(rows_to_mask)
        mask[rows_to_mask] = 0
        x = x * mask
        assert_allclose(x[rows_to_mask].sum().detach(), 0)

    return x


def init_weights(
    x: torch.Tensor,
    init_edge_weights: str = 'trunc_normal',
    fillval: float = 1.0,
) -> torch.Tensor:
    """Initializes and returns new tensor.
    """
    assert init_edge_weights in ['trunc_normal', 'glorot', 'fill']

    if init_edge_weights == 'trunc_normal':
        nn.init.trunc_normal_(x, mean=0.75, std=0.25, a=0.4, b=1)
    elif init_edge_weights == 'glorot':
        nn.init.xavier_uniform_(x)
    else:
        nn.init.constant_(x, fillval)
    return x
