# =============================================================================
# @file   base_policy.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Base policy function module declaration."""
from __future__ import annotations

from typing import Optional

import torch as th


class BasePolicy(th.nn.Module):
    network: th.nn.Module
    target_network: th.nn.Module

    def forward(self, obs: th.Tensor, target: bool = False) -> th.Tensor:
        raise NotImplementedError

    def get_action(self,
                   obs: th.Tensor,
                   explore: bool = True,
                   target: bool = False) -> th.Tensor:
        raise NotImplementedError

    def sync_target(self, non_blocking: bool = True) -> None:
        raise NotImplementedError
