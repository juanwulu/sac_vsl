# =============================================================================
# @file   base_agent.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Base agent class declaration"""
from __future__ import annotations

import os
from typing import Any, Dict, Union

import numpy as np
import torch as th
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.critic.base_critic import BaseCritic
from src.policy.base_policy import BasePolicy

# Type Aliases
Log = Dict[str, Any]
PathLike = Union[str, 'os.PathLike[str]']


class BaseAgent:
    agent_id: str = ''
    device: th.device = th.device('cpu')
    critic: BaseCritic
    critic_opt: Optimizer
    critic_lr_scheduler: _LRScheduler
    policy: BasePolicy
    policy_opt: Optimizer
    policy_lr_scheduler: _LRScheduler

    def update_policy(self, obs: th.Tensor, acs: th.Tensor) -> Log:
        raise NotImplementedError

    def update_critic(self,
                      obs: th.Tensor,
                      acs: th.Tensor,
                      next_obs: th.Tensor,
                      next_acs: th.Tensor) -> Log:
        raise NotImplementedError

    def get_action(self,
                   obs: Union[np.ndarray, th.Tensor],
                   explore: bool = True,
                   target: bool = False) -> th.Tensor:
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).to(self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        return self.policy.get_action(obs, explore, target)

    def save(self, filepath: PathLike) -> None:
        assert os.path.exists(os.path.dirname(filepath)), \
            ValueError('Checkpoint directory not exitst')
        state_dict = {
            "critic_state_dict": self.critic.state_dict(),
            "critic_opt_state_dict": self.critic_opt.state_dict(),
            "critic_lr_scheduler_state_dict":
            self.critic_lr_scheduler.state_dict()
            if self.critic_lr_scheduler is not None else None,
            "policy_state_dict": self.policy.state_dict(),
            "policy_opt_state_dict": self.policy_opt.state_dict(),
            "policy_lr_scheduler_state_dict":
            self.policy_lr_scheduler.state_dict()
            if self.policy_lr_scheduler is not None else None
        }
        th.save(state_dict, filepath)

    def train(self) -> None:
        for module in dir(self):
            if isinstance(module, th.nn.Module):
                module.train()

    def eval(self) -> None:
        for module in dir(self):
            if isinstance(module, th.nn.Module):
                module.eval()

    def sync_target(self, non_blocking: bool = True) -> None:
        self.critic.sync_target(non_blocking)
        self.policy.sync_target(non_blocking)
