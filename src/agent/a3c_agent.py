# =============================================================================
# @file   a3c_agent.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Actor-Critic agent class declaration"""
from __future__ import annotations

from typing import Optional

import torch as th

from src.agent.base_agent import BaseAgent
from src.critic.base_critic import BaseCritic
from src.policy.base_policy import BasePolicy


class A3CAgent(BaseAgent):

    def __init__(self,
                 critic: BaseCritic,
                 policy: BasePolicy,
                 critic_lr: Optional[float],
                 policy_lr: Optional[float],
                 learning_rate: Optional[float] = 1e-4) -> None:
        super().__init__()

        self.critic = critic
        self.critic_opt = None
        self.policy = policy
