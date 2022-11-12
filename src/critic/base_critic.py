# =============================================================================
# @file   base_critic.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Base critic function module declaration."""
from __future__ import annotations

import torch as th


class BaseCritic(th.nn.Module):
    network: th.nn.Module
    target_network: th.nn.Module

    def forward(self,
                obs: th.Tensor,
                acs: th.Tensor,
                target: bool = False) -> th.Tensor:
        raise NotImplementedError

    def update_target_function(self, non_blocking: bool = False) -> None:
        raise NotImplementedError
