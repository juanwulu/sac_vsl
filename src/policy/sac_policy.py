# =============================================================================
# @file   sac_policy.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
from __future__ import annotations

import math

import torch as th
import torch.distributions as D

from src.policy.base_policy import BasePolicy
from src.typing import Cfg


# https://github.com/ChocolateDave/cs285_homework_fall2022/blob/main/hw3/cs285/infrastructure/sac_utils.py
class SquashedNormal(D.transformed_distribution.TransformedDistribution):

    def __init__(self, loc: th.Tensor, scale: th.Tensor) -> None:
        self.loc = loc
        self.scale = scale

        self.base_dist = D.Normal(loc, scale)
        transforms = [D.transforms.TanhTransform]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tsfm in self.transforms:
            mu = tsfm(mu)

        return mu


class SACPolicy(BasePolicy):

    def __init__(self, config: Cfg) -> None:
        super().__init__()

        self.action_range = config.get('action_range', [-1, 1])
        self.init_temperature = config.get('init_temperature', 1.0)
        self.log_std_bounds = config.get('log_std_bounds', [-20, 2])
        self.tar_entropy = config.get('target_entropy', -self.ac_dim)

        self.log_std = th.nn.Parameter()
        self.log_alpha = th.nn.Parameter(
            math.log(self.init_temperature)
        ).to(self.device)

    @property
    def alpha(self) -> th.Tensor:
        return self.log_alpha.exp()

    def forward(self, obs, target=False):
        if target:
            loc = self.target_network.forward(obs)
        else:
            loc = self.network.forward(obs)
        scale = self.log_std.clamp(*self.log_std_bounds).exp()

        ac_dist = SquashedNormal(loc=loc, scale=scale)
        return ac_dist

    def get_action(self, obs, explore=True, target=False):
        if len(obs.shape) == 1:
            obs = obs[None, ...]

        with th.no_grad():
            dist = self.forward(th.from_numpy(obs).to(self.device))
            if explore:
                acs = dist.rsample()
            else:
                acs = dist.mean()
        acs = acs.clamp(*self.action_range).detach().cpu().numpy()

        return acs
