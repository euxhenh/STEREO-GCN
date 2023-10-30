import logging
from typing import List

import torch
import torch.nn as nn
from torch import FloatTensor

from ..utils.search import search_nonlin_fn
from .layers import EGCHUnit

logger = logging.getLogger("pytorch_lightning.core")


class Block(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        *,
        batch_norm: bool = True,
        dropout: float = 0.0,
        nonlin: str = 'LeakyReLU',
        **kwargs
    ):
        """A single EGCH block that applies EGCHUnit, activation, batch norm,
        and optional[dropout].
        """
        super(Block, self).__init__()

        self.unit = EGCHUnit(in_features, out_features=out_features, **kwargs)
        self.nonlin = search_nonlin_fn(nonlin)
        # Don't track running stats as this will be used for all time
        # points with differing distributions. Alternatively, track but use
        # a separate batch norm layer for every time point.
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_features or in_features,
                                     track_running_stats=False)
        else:
            self.register_parameter('bn', None)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.register_parameter('dropout', None)

    def forward(self, x_seq: List[FloatTensor], A_seq: List[FloatTensor], **kwargs):
        out = self.unit(x_seq, A_seq, **kwargs)
        out = [self.nonlin(x) for x in out]
        for layer in [self.bn, self.dropout]:
            if layer is not None:
                out = [layer(x) for x in out]
        return out


class StackedEGCH(nn.Module):
    """Stacks multiple Block's to form multi-layer EGCH's.

    Parameters
    ----------
    in_features: int
        Number of node features.
    out_features: int, None
        Number of output node features. If None, will be the same as
        `in_features`.
    n_blocks: int
        Number of EGCH blocks to stack.
    add_input_linear_layer: bool
        If True, will add a Linear layer before EGCH blocks.
    add_output_linear_layer: bool
        If True, will add a Linear layer after EGCH blocks.
    hidden_size: int, None
        If not None, will add a second Linear Layer after EGCH blocks. The
        input dim of this layer equals `hidden_size`.
    linear_bias: bool
        Whether to use bias for all Linear layers or not.
    final_block_out_features: int, None
        Number of output features for the last EGCH block. If None, will
        use `in_features`.
    nonlin: str
        Nonlinear function to use. Will search torch.nn and torch.
    add_input_skip_connection: bool
        If True, will concatenate the input node features to the output of
        all EGCH blocks.
    unit_kwargs: dict
        Additional args to pass to EGCH blocks.
    """

    def __init__(
        self,
        in_features: int,
        *,
        out_features: int | None = None,
        n_blocks: int = 2,
        add_input_linear_layer: bool = False,
        add_output_linear_layer: bool = True,
        hidden_size: int | None = None,
        linear_bias: bool = False,
        final_block_out_features: int | None = None,
        nonlin: str = 'LeakyReLU',
        add_input_skip_connection: bool = False,
        add_intermed_skip_connections: bool = True,
        unit_kwargs: dict = {},
    ):
        super(StackedEGCH, self).__init__()

        self.add_input_skip_connection = add_input_skip_connection
        self.add_intermed_skip_connections = add_intermed_skip_connections
        # If None, assume same dim as input
        out_features = out_features or in_features

        # Add linear layer to input
        if add_input_linear_layer:
            self.lin_inp = nn.Linear(in_features, in_features, bias=linear_bias)
            self.nonlin_inp = search_nonlin_fn(nonlin)
        else:
            self.register_parameter('lin_inp', None)
            self.register_parameter('nonlin_inp', None)

        # Add EGCH blocks
        self.egch = nn.ModuleList()
        for i in range(n_blocks):
            block = Block(
                in_features=in_features,
                out_features=(
                    final_block_out_features if i == n_blocks - 1 else None
                ),
                **unit_kwargs,
            )
            self.egch.add_module(f'unit_{i}', block)

        # Add linear layers to output
        if add_output_linear_layer:
            # Determine shape of linear layer.
            lin_in_features = (
                in_features * (n_blocks - 1)
                + (final_block_out_features or in_features)
                + (in_features if add_input_skip_connection else 0)
            )
            self.lin1 = nn.Linear(lin_in_features,
                                  hidden_size or out_features,
                                  bias=linear_bias)
        else:
            self.register_parameter('lin1', None)

        # Check if we need to add a second linear layer or not.
        if add_output_linear_layer and hidden_size:
            self.nonlin_out = search_nonlin_fn(nonlin)
            self.lin_out = nn.Linear(hidden_size, out_features, bias=linear_bias)
        else:
            self.register_parameter('nonlin_out', None)
            self.register_parameter('lin_out', None)

    def forward(self, x_seq: List[FloatTensor], A_seq: List[FloatTensor], **kwargs):
        """Forward pass.

        NOTE: A is assumed to be in 'target_to_source' mode.

        Returns List[FloatTensor].
            The transformed data.
        """
        outputs = [] if not self.add_input_skip_connection else [x_seq]

        if self.lin_inp is not None:
            x_seq = [self.nonlin_inp(self.lin_inp(x)) for x in x_seq]

        for block in self.egch:
            x_seq = block(x_seq, A_seq, **kwargs)
            outputs.append(x_seq)

        if not self.add_intermed_skip_connections:
            outputs = outputs[-1:]

        # Concatenate output from all blocks
        x_seq = [torch.cat(x_t, dim=1) for x_t in zip(*outputs)]

        for layer in [self.lin1, self.nonlin_out, self.lin_out]:
            if layer is not None:
                x_seq = [layer(x) for x in x_seq]

        x_seq = [x.squeeze() for x in x_seq]
        return x_seq
