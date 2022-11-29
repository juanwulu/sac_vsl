# =============================================================================
# @file   base_agent.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Base agent class declaration"""
from __future__ import annotations

import os
from typing import Any, Dict, TypeVar, Union

import numpy as np
import torch as th
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.critic.base_critic import BaseCritic
from src.policy.base_policy import BasePolicy
from src.typing import Cfg, Log, PathLike
from src.utils.meter import AverageMeterGroup
from src.utils.resolver import resolver


def optimizer_resolver(optimizer_id: Union[str, Any] = 'adam',
                       *args, **kwargs) -> Optimizer:
    base_cls = th.optim.Optimizer
    base_cls_repr = 'Optimizer'
    optimizers = [optim for optim in vars(th.optim).values()
                  if isinstance(optim, type) and issubclass(optim, base_cls)]
    optimizer_dict = {}

    return resolver(optimizers, optimizer_dict, optimizer_id,
                    base_cls, base_cls_repr, *args, **kwargs)


def lr_scheduler_resolver(lr_scheduler_id: Union[str, Any] = 'step_lr',
                          *args, **kwargs) -> _LRScheduler:
    base_cls = th.optim.lr_scheduler._LRScheduler
    base_cls_repr = '_LRScheduler'
    schedulers = [sched for sched in vars(th.optim.lr_scheduler).values()
                  if isinstance(sched, type) and issubclass(sched, base_cls)]
    scheduler_dict = {}

    return resolver(schedulers, scheduler_dict, lr_scheduler_id,
                    base_cls, base_cls_repr, *args, **kwargs)


class BaseAgent:
    agent_id: str = ''
    config: Cfg
    device: th.device = th.device('cpu')
    critic: BaseCritic
    critic_opt: Optimizer
    critic_lr_scheduler: _LRScheduler
    policy: BasePolicy
    policy_opt: Optimizer
    policy_lr_scheduler: _LRScheduler
    steps: int = 0

    def update_policy(self, obs: th.Tensor, acs: th.Tensor) -> Log:
        raise NotImplementedError

    def update_critic(self,
                      obs: th.Tensor,
                      acs: th.Tensor,
                      rews: th.Tensor,
                      next_obs: th.Tensor,
                      next_acs: th.Tensor,
                      dones: th.Tensor) -> Log:
        raise NotImplementedError

    def __init__(self, config: Cfg) -> None:
        super().__init__()

        self.learning_rate = config.get('learning_rate')
        pass

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

    def step(self,
             obs: np.ndarray,
             acs: np.ndarray,
             rews: np.ndarray,
             next_obs: np.ndarray,
             next_acs: np.ndarray,
             dones: np.ndarray) -> Log:
        self.steps += 1
        meter = AverageMeterGroup()

        for _ in range(self.config['num_critic_updates_per_step']):
            critic_log = self.update_critic(
                th.from_numpy(obs).to(self.device),
                th.from_numpy(acs).to(self.device),
                th.from_numpy(rews).to(self.device),
                th.from_numpy(next_obs).to(self.device),
                th.from_numpy(next_acs).to(self.device),
                th.from_numpy(dones).to(self.device)
            )
            meter.update(critic_log)

        if self.steps % self.config['target_sync_frequency'] == 0:
            self.sync_target(self.config['non_blocking'])

        return meter.meters()

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
