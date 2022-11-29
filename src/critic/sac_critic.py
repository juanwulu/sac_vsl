# =============================================================================
# @file   sac_critic.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Soft Actor-Critic critic function module declaration"""
from __future__ import annotations

from typing import Tuple

import torch as th

from src.critic.base_critic import BaseCritic
from src.typing import Cfg


class SACCritic(BaseCritic):

    def __init__(self, config: Cfg) -> None:
        super().__init__(config)

        self.network_1 = None
        self.network_2 = None

    def forward(self, obs, acs) -> Tuple[th.Tensor, th.Tensor]:
        q_1 = self.network_1.forward(obs, acs)
        q_2 = self.network_2.forward(obs, acs)

        return [q_1, q_2]
