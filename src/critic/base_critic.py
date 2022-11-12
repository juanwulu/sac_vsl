# =============================================================================
# @file   base_critic.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Base critic function module declaration"""
from __future__ import annotations

import torch as th

from src.typing import Cfg
from src.utils.resolver import resolver


class BaseCritic(th.nn.Module):
    discount: float
    loss_func: th.nn.Module

    def __init__(self, config: Cfg) -> None:
        super().__init__()

        self.discount = config.get('discount', 0.99)
        self.loss_func = resolver(
            classes=[loss for loss in vars(th.nn).values()
                     if isinstance(loss, type)
                     and issubclass(loss, th.nn)
                     and 'Loss' in loss.__name__],
            class_dict={},
            query=config.get('critic_loss', 'mse_loss'),
            base_cls=th.nn,
            base_cls_repr='Loss',
            **config.get('critic_loss_kwargs', {})
        )

    def forward(self,
                obs: th.Tensor,
                acs: th.Tensor,
                target: bool = False) -> th.Tensor:
        raise NotImplementedError

    def sync_target(self, non_blocking: bool = False) -> None:
        raise NotImplementedError
