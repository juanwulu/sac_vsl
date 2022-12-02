# =============================================================================
# @file   agent.py
# @author Juanwu Lu
# @date   Oct-2-22
# =============================================================================
from __future__ import annotations

import os
from typing import Optional, Union

import torch as th
from numpy import ndarray
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.algorithm.critic import BaseCritic
from src.algorithm.policy import BasePolicy

# Type Alias
_PathLike = Union[str, 'os.PathLike[str]']


class BaseAgent:
    agent_id: str = ''
    device: th.device
    critic: BaseCritic
    critic_opt: Optimizer
    critic_lr_scheduler: _LRScheduler = None
    policy: BasePolicy
    policy_opt: Optimizer
    policy_lr_scheduler: _LRScheduler = None

    def update_policy(self,
                      obs: Optional[Tensor] = None,
                      action: Optional[Tensor] = None) -> None:
        raise NotImplementedError

    def update_critic(self,
                      obs: Tensor,
                      action: Tensor,
                      next_obs: Tensor,
                      rewards: Tensor,
                      dones: Tensor) -> None:
        raise NotImplementedError

    def get_action(self,
                   obs: Union[ndarray, Tensor],
                   explore: bool = True,
                   target: bool = False) -> Tensor:
        if isinstance(obs, ndarray):
            obs = th.from_numpy(obs).to(self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        return self.policy.get_action(obs, explore, target)

    def update_target(self, non_blocking: bool = True) -> None:
        self.critic.sync(non_blocking)
        self.policy.sync(non_blocking)

    def save(self, filepath: _PathLike) -> None:
        state_dict = {
            "critic_state_dict": self.critic.state_dict(),
            "critic_opt_state_dict": self.critic_opt.state_dict(),
            "critic_lr_scheduler_state_dict": (
                self.critic_lr_scheduler.state_dict()
                if self.critic_lr_scheduler is not None else None
            ),
            "policy_state_dict": self.policy.state_dict(),
            "policy_opt_state_dict": self.policy_opt.state_dict(),
            "policy_lr_scheduler_state_dict": (
                self.policy_lr_scheduler.state_dict()
                if self.policy_lr_scheduler is not None else None
            )
        }
        th.save(state_dict, filepath)

    def set_train(self) -> None:
        for module in dir(self):
            if isinstance(module, nn.Module):
                module.train()

    def set_eval(self) -> None:
        for module in dir(self):
            if isinstance(module, nn.Module):
                module.eval()


class SACAgent(BaseAgent):

    def __init__(self) -> None:
        super().__init__()
