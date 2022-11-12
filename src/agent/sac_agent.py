# =============================================================================
# @file   sac_agent.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Soft Actor-Critic agent class declaration"""
from __future__ import annotations

import torch as th

from src.agent.base_agent import BaseAgent, optimizer_resolver
from src.policy.sac_policy import SACPolicy


class SACAgent(BaseAgent):
    policy: SACPolicy
    log_alpha_opt: th.optim.Optimizer

    def __init__(self, config) -> None:
        super().__init__(config)

        self.log_alpha_opt = optimizer_resolver(
            config.get('log_alpha_opt', 'adam'),
            params=[self.policy.log_alpha],
            lr=config.get('log_alpha_lr', 3e-4)
        )

    def update_critic(self, obs, acs, rews, next_obs, next_acs, dones):
        if rews.dim() == 1:
            rews = rews.view(-1, 1)
        if dones.dim() == 1:
            dones = dones.view(-1, 1)

        if next_acs is None:
            next_acs = self.policy.get_action(next_obs, False, True)
        bellman_tar = rews + self.critic.discount * (1.0 - dones) * \
            self.critic.forward(next_obs, next_acs, True).detach()
        q_vals = self.critic.forward(obs, acs, target=False)
        critic_loss: th.Tensor = self.critic.loss_func(q_vals, bellman_tar)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return {'critic_loss': critic_loss.item()}

    def update_policy(self, obs, acs):
        act_dist = self.policy.forward(obs, False)
        acs = act_dist.rsample()
        log_prob = act_dist.log_prob(acs)
        q_1, q_2 = -self.critic.forward(obs, acs)
        actor_q = th.min(q_1, q_2)
        policy_loss = (actor_q - self.policy.alpha.detach() * log_prob).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        alpha_tar = (-log_prob - self.policy.tar_entropy).detach()
        alpha_loss = (self.policy.alpha * alpha_tar).mean()
        self.log_alpha_opt.zero_grad()
        alpha_loss.backward()
        self.log_alpha_opt.step()

        return {'policy_loss': policy_loss.item(),
                'alpha_loss': alpha_loss.item()}
