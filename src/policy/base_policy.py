# =============================================================================
# @file   base_policy.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Base policy function module declaration"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch as th
import torch.distributions as D

from src.typing import Cfg


class BasePolicy(th.nn.Module):
    device: th.device = th.device('cpu')
    network: th.nn.Module
    target_network: th.nn.Module

    def __init__(self, config: Cfg) -> None:
        super().__init__()

        self.ob_dim = config['ob_dim']
        self.ac_dim = config['ac_dim']

        # TODO: network resolver

    def forward(self,
                obs: th.Tensor,
                target: bool = False) -> Union[D.Distribution, th.Tensor]:
        raise NotImplementedError

    def get_action(self,
                   obs: np.ndarray,
                   explore: bool = True,
                   target: bool = False) -> np.ndarray:
        raise NotImplementedError

    def sync_target(self, non_blocking: bool = True) -> None:
        raise NotImplementedError
