# ==============================================================================
# @file   layers.py
# @author Juanwu Lu
# @date   Sep-16-22
# ==============================================================================
"""Neural Network Layers."""
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, Tuple, Union
import warnings

import torch as th
from torch import (nn, Tensor)
from torch_geometric.nn import MessagePassing


class Activation(nn.Module):
    """A general activation function layer.
    
    Attributes:
        activation: A string name or a callable of the activation function.
        activation_kwargs: Keyword arguments for the activation function.
    """
    
    _act_lut: Dict[str, nn.Module] = {
        "elu": nn.ELU,
        "leakyrelu": nn.LeakyReLU,
        "logsigmoid": nn.LogSigmoid,
        "prelu": nn.PReLU,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "selu": nn.SELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "threashold": nn.Threshold,
    }

    def __init__(self, activation: Union[str, Callable], **kwargs) -> None:
        super().__init__()

        if isinstance(activation, Callable):
            self.activation = activation
            self.activation_kwargs = kwargs
        elif isinstance(activation, str):
            if activation.lower() in self._act_lut:
                self.activation = self._act_lut[activation.lower()]
                self.activation_kwargs = kwargs
            else:
                warnings.warn(
                    f"{activation:s} is not supported, fall back to ReLU."
                )
                self.activation = nn.ReLU
                self.activation_kwargs = {}
        else:
            raise TypeError(
                "Expect activation to be 'str' or 'Callable', "
                f"but got {type(activation)}."
            )
        
    def forward(self, x: Tensor) -> Tensor:
        x = x.float()

        return self.activation(x, **self.activation_kwargs)


class StaticEvol(nn.Module):
    """Static block state evolution predictor with time and space features.

    Without external influences, we can model the velocity on a given highway
    segment as a function depend on current temporal features and the location
    embedding of the segment :math:`f(loc, t)`.
    The static evolution network is essentially an Multi-Layer Perceptron for
    inferring the velocity of a block in an end-to-end paragdim.

    Args:
        n_loc: Number of total locations.
        time_feature: Size of time-related features.
        loc_feature: Size of location embedding dictionary.
        n_hidden: Size of linear layers.
        n_layers: Number of linear layers.
        activation: A str name or Callble activation function.
        activation_kwargs: A dictionary of activation keyword arguments.
        bias: If use bias for linear layers.
        residual: If use residual connections.

    Shapes:
        - Input: :math:`(*, 1 + F_{t})`
        - Output: :math`(*, 1)`
    """

    def __init__(
        self, n_loc: int, time_feature: int, loc_feature: int = 64,
        n_hidden: Union[int, Iterable[int]] = 64, n_layers: int = 2,
        activation: Union[str, Callable] = "relu",
        activation_kwargs: Dict[str, Any] = {},
        bias: bool = True, residual: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(n_hidden, int):
            self.n_hidden = [n_hidden] * n_layers
        elif isinstance(n_hidden, Iterable):
            assert len(n_hidden) == n_layers, ValueError(
                "'n_hidden' does not match with 'n_layers'. "
                f"Expect a size of {n_layers:d}, but got {len(n_hidden)}."
            )
            self.n_hidden = list(n_hidden)
        else:
            raise TypeError(
                "Expect 'n_hidden' to be int or Iterable of ints, "
                f"but got {type(n_hidden)}."
            )
        
        self.loc_embedding = nn.Embedding(n_loc, loc_feature)
        self.mlp = nn.Sequential() 
        self.activation = Activation(activation, **activation_kwargs)

        in_features: int = loc_feature + time_feature
        for i, hidden in enumerate(self.n_hidden):
            self.mlp.add_module(
                f"linear_{i+1:d}",
                nn.Linear(in_features, hidden, bias=bias)
            )
            self.mlp.add_module(
                f"layernorm_{i+1:d}",
                nn.LayerNorm(hidden)
            )
            self.mlp.add_module(
                f"activation_{i+1:d}",
                Activation(activation, **activation_kwargs)
            )
            in_features = hidden
        
        self.register_module("residual_linear", None)
        self.residual = residual
        if residual and loc_feature + time_feature != self.n_hidden[-1]:
            self.residual = nn.Sequential(
                OrderedDict([
                    ("res_linear", nn.Linear(
                        loc_feature + time_feature, self.n_hidden[-1]
                    ))
                    ("res_layernorm", nn.LayerNorm(self.n_hidden[-1]))
                ])
            )
        
        # Output velocity predictor.
        self.output_head = nn.Sequential(
            OrderedDict([
                ("linear_1", nn.Linear(self.n_hidden[-1], self.n_hidden[-1])),
                ("activation", Activation(activation, **activation_kwargs)),
                ("linear_2", nn.Linear(self.n_hidden[-1], 1))
            ])
        )
        
    def forward(self, loc: Tensor, time: Tensor) -> Tensor:
        loc = loc.long()
        time = time.float()

        loc = self.loc_embedding(loc)
        feat = th.cat([loc, time], dim=-1)
        x = self.mlp(feat)
        if self.residual:
            if self.residual_linear is not None:
                feat = x + self.residual_linear(feat)
            else:
                feat = x + feat
            feat = self.activation(feat)
        out = self.output_head(feat)

        return out


class TemporalBlock(nn.Module):
    """Temporal convolution layer blocks with 1D dilated Convolutions.

    The temporal block is a direct implementation of the dilated convolution
    residual blocks in the paper "WaveNet: A Generative Model for Raw Audio"
    by Oord. A. V. D., et al.

    Args:
        res_channels: Number of residual channels for recurrent output.
        skip_channels: Number of skip channels for direct output
        dilation: Size of dilation.
    
    Shapes:
        - Input: :math:`(*, C, L)`
        - RecurOutput: :math:`(*, Cr, L)`
        - SkipOutput: :math:`(*, Cs, L)`
    """

    def __init__(
        self, res_channels: int, skip_channels: int, dilation: int = 1
    ) -> None:
        super().__init__()

        self.dilated_conv = nn.Conv1d(res_channels,
                                      res_channels,
                                      kernel_size=2,
                                      stride=1,
                                      padding=0,
                                      dilation=dilation,
                                      bias=False)
        # 1 x 1 convolutions.
        self.conv_res = nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = nn.Conv1d(res_channels, skip_channels, 1)

        # Gated activation units.
        self.tanh_gate = nn.Sequential(
            OrderedDict([
                ("linear", nn.Conv1d(res_channels, res_channels, 1)),
                ("tanh_gate", nn.Tanh())
            ])
        )
        self.sigmoid_gate = nn.Sequential(
            OrderedDict([
                ("linear", nn.Conv1d(res_channels, res_channels)),
                ("sigmoid_gate", nn.Sigmoid())
            ])
        )

    def forward(self, x: Tensor, skip_size: int) -> Tuple[Tensor, Tensor]:
        x = x.float()

        # Recurrent output.
        out: Tensor = self.dilated_conv(x)
        gated = self.tanh_gate(out) * self.sigmoid_gate(out)
        out = self.conv_res(gated)
        out += x[:, :, -out.shape[-1]]

        # Skip connection.
        skip = self.conv_skip(gated)
        skip = skip[:, :, -skip_size:]

        return out, skip


class DynEvol(nn.Module):
    """Dynamic block state evolution estimator with local temporal dependencies.

    The dynamic block state evolution estimator aims to capture local velocity
    volatility given observation of historical speeds and other features within
    a temporal influential horizon `tau`.

    Args:
        in_features: Size of input features.
        in_channels: Number of input variables.
        res_channels: Number of residual channels.
        num_layers: Number of temporal convolution layers, max dialation=`2**n`.
        num_blocks: Number of temporal blocks with the same number of layers.
        activation: Activation function.
        activation_kwargs: Keyword arguments for the activation function.

    Shapes:
        - Input: :math:`[*, C, L]`
        - Output: :math:`[*, C', L]`
    """

    def __init__(
        self, in_features: int, in_channels: int, res_channels: int,
        num_layers: int, num_blocks: int,
        activation: Union[str, Callable] = "relu",
        activation_kwargs: Dict[str, Any] = {}
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.activation = activation
        self.activation_kwargs = activation_kwargs

        self._build()

    def _build(self) -> None:
        # Kick start temporal convolution with causal convolution
        self.causal_conv = th.nn.Conv1d(self.in_channels,
                                        self.res_channels,
                                        kernel_size=2,
                                        stride=1,
                                        padding=1,
                                        bias=False)

        # Building stacked temporal blocks
        self.temp_blocks = []
        dilations = self.num_blocks * [2 ** l for l in range(self.num_layers)]
        for dilation in dilations:
            self.temp_blocks.append(
                TemporalBlock(
                    self.res_channels, self.in_channels, dilation=dilation
                )
            )
        self.temp_blocks = nn.ModuleList(self.temp_blocks)

        # Building output layer
        out_features = self._get_out_features()
        self.out_layer = nn.Sequential(
            OrderedDict([
                ("activation_1", Activation(self.activation,
                                            **self.activation_kwargs))
                ("conv_1", nn.Conv1d(self.in_channels, self.in_channels, 1))
                ("activation_2", Activation(self.activation,
                                            **self.activation_kwargs))
                ("flatten", nn.Flatten(start_dim=1, end_dim=-1))
                ("linear_2", nn.Linear(out_features, 1))
            ])
        )

    def _get_out_features(self) -> int:
        receptive_field = sum(
            [2 ** l for l in range(self.num_layers)] * self.num_blocks
        ) 
        out_features = self.in_features - receptive_field

        if out_features < 1:
            raise RuntimeError(
                "Expect the input horizon to be larger than receptive field,"
                f" got an input size of {self.in_features:d},"
                f" a receptive field size of {receptive_field:d}."
            )

        return out_features 
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.float()

        res_out = self.causal_conv(x)
        skip = []
        for temporal_block in self.temp_blocks:
            res_out, skip_out = temporal_block(
                res_out, self._get_out_features()
            )
            skip.append(skip_out)
        out = th.sum(skip_out, dim=0)
        out = self.out_layer(out)

        return out


class PropagateEvol(MessagePassing):
    """"""


class Shockwave(nn.Module):
    """"""