# =============================================================================
# @file   critic.py
# @author Juanwu Lu
# @date   Oct-9-22
# =============================================================================
from __future__ import annotations

from typing import Optional

from torch import Tensor, nn


class BaseCritic(nn.Module):
    critic_net: nn.Module
    target_critic_net: nn.Module

    def forward(self,
                obs: Tensor,
                action: Optional[Tensor] = None,
                target: bool = False) -> Tensor:
        raise NotImplementedError

    def sync(self, non_blocking: bool = False) -> None:
        raise NotImplementedError
