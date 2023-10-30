from typing import Callable

import torch
from torch import nn, optim
from torch.nn import functional as F


def search_loss_fn(loss_fn: Callable | str, **kwargs):
    """
    Returns a loss object from loss_fn by calling it, searching torch.nn,
    or searching torch.nn.functional.
    """
    if callable(loss_fn):
        criterion = loss_fn(**kwargs)
    elif isinstance(loss_fn, str):
        # Search for loss in "torch.nn", and "torch.nn.functional" in that
        # order.
        if hasattr(nn, loss_fn):
            criterion = getattr(nn, loss_fn)(**kwargs)
        elif hasattr(F, loss_fn):
            criterion = getattr(F, loss_fn)
        else:
            raise ValueError(
                f"Could not find loss `{loss_fn}` in "
                "'torch.nn' or 'torch.nn.functional'."
            )
    else:
        raise ValueError(
            "Could not recognize loss function "
            f"of type {loss_fn.__class__.__name__}."
        )

    return criterion


def search_optim_fn(optim_fn: Callable | str) -> Callable:
    """
    Returns an optimizer callable from optim_fn by calling it or searching
    torch.optim.
    """
    if callable(optim_fn):
        optimizer = optim_fn
    elif isinstance(optim_fn, str):
        # Search torch.optim and return None if not found.
        optimizer = getattr(optim, optim_fn, None)  # type: ignore
        if optimizer is None:
            raise ValueError(
                f"Could not find optimizer `{optim_fn}` "
                "in 'torch.optim'."
            )
    else:
        raise ValueError(
            "Could not recognize optimizer function of "
            f"type {optim_fn.__class__.__name__}."
        )

    return optimizer


def search_nonlin_fn(nonlin_fn: Callable | str, **kwargs):
    """Searches and returns nonlinearity object or function.
    """
    if callable(nonlin_fn):
        nonlin = nonlin_fn(**kwargs)
    elif isinstance(nonlin_fn, str):
        if hasattr(nn, nonlin_fn):
            nonlin = getattr(nn, nonlin_fn)(**kwargs)
        elif hasattr(torch, nonlin_fn):
            nonlin = getattr(torch, nonlin_fn)
        elif hasattr(F, nonlin_fn):
            nonlin = getattr(F, nonlin_fn)
        else:
            raise ValueError(
                f"Could not find nonlinearity `{nonlin_fn}` in "
                "'torch.nn', 'torch', or 'torch.nn.functional'."
            )
    else:
        raise ValueError(
            "Could not recognize nonlinearity function "
            f"of type {nonlin_fn.__class__.__name__}."
        )

    return nonlin
