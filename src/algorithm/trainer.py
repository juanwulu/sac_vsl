# =============================================================================
# @file   trainer.py
# @author Juanwu Lu
# @date   Oct-23-22
# =============================================================================
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Mapping, Optional, Tuple, Union

import torch as th
from gymnasium.core import Env
from gymnasium.spaces import Discrete
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.algorithm.agent import BaseAgent, SACAgent
from src.memory import BaseBuffer, ReplayBuffer
from src.meter import AverageMeterGroup
from src.typing import PathLike


class BaseTrainer:
    agent: BaseAgent
    buffer: BaseBuffer
    env: Env

    def train_one_episode(self, epoch: int) -> Mapping[str, Any]:
        raise NotImplementedError

    def exec_one_episode(self, epoch: int = -1) -> Mapping[str, Any]:
        raise NotImplementedError

    def __init__(self,
                 log_dir: PathLike,
                 num_episodes: int,
                 exp_name: str = 'default',
                 max_episode_steps: Optional[int] = None,
                 eval_frequency: Optional[int] = 10) -> None:

        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.eval_frequency = eval_frequency
        self.steps_so_far = 0

        now = datetime.now().strftime('%m-%d-%d_%H-%M-%S')
        log_dir = os.path.join(log_dir, f'{exp_name:s}_{now:s}')
        self.writer = SummaryWriter(log_dir)

    def train(self, execution: bool = False) -> Any:
        meter = AverageMeterGroup()
        for episode in tqdm(range(1, self.num_episodes + 1),
                            desc='Training Progress',
                            position=0,
                            leave=False):
            self.set_train()

            log = self.train_one_episode(episode)
            # Update episodic tracker
            meter.update(log)
            for key, val in meter.items():
                key = 'Train/' + key
                self.writer.add_scalar(key, val, episode)
            if episode % self.eval_frequency == 0 and execution:
                self.set_eval()
                mean_reward = 0.
                for i in range(20):
                    log = self.exec_one_epoch(episode)
                    mean_reward += log['eval_returns']
                log = {'mean_reward': mean_reward / 20}
                for key, val in log.items():
                    key = 'Execution/' + key
                    self.writer.add_scalar(key, val, episode)

    def set_train(self) -> None:
        self.agent.set_train()

    def set_eval(self) -> None:
        self.agent.set_eval()


class SACTrainer:

    def __init__(self,
                 env: Env,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 device: th.device = th.device('cpu'),
                 log_dir: str = 'logs/',
                 num_episodes: int = 20000,
                 name: str = '',
                 policy_net: Optional[Union[str, BaseNN]] = 'MLP',
                 policy_net_kwargs: Optional[Mapping[str, Any]] = None,
                 critic_net: Optional[Union[str, BaseNN]] = 'MLP',
                 critic_net_kwargs: Optional[Mapping[str, Any]] = None,
                 learning_rate: Optional[float] = None,
                 policy_lr: Optional[float] = None,
                 critic_lr: Optional[float] = None,
                 discount: Optional[float] = 0.99,
                 grad_clip: Optional[Tuple[float, float]] = None,
                 soft_update_tau: Optional[float] = 0.9,
                 max_episode_steps: Optional[int] = None,
                 num_timesteps_before_training: Optional[int] = 100,
                 seed: int = -1) -> None:
        super().__init__(log_dir, num_episodes, name, max_episode_steps)

        # Retreive observation and action size
        if len(env.observation_space.shape) > 2:
            observation_size = env.observation_space
        else:
            observation_size = env.observation_space.shape[0]
        if isinstance(env.action_space, Discrete):
            action_size = env.action_space.n
            self.discrete_action = True
        else:
            action_size = env.action_space.shape[0]
            self.discrete_action = False

        self.agent: SACAgent = SACAgent(observation_size=observation_size,
                                        action_size=action_size,
                                        discrete_action=self.discrete_action,
                                        device=device,
                                        policy_net=policy_net,
                                        policy_net_kwargs=policy_net_kwargs,
                                        policy_lr=policy_lr,
                                        critic_net=critic_net,
                                        critic_net_kwargs=critic_net_kwargs,
                                        learning_rate=learning_rate,
                                        critic_lr=critic_lr,
                                        discount=discount,
                                        grad_clip=grad_clip,
                                        soft_update_tau=soft_update_tau)
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.env = env
        self.seed = seed

        self.num_timesteps_before_training = num_timesteps_before_training

        if seed > 0:
            self.env.reset(seed=seed)

    def train_one_episode(self,
                          episode: int,
                          seed: Optional[int] = None) -> Any:
        seed = seed or self.seed
        log = defaultdict(list)

        # Initialize random process
        self.agent.reset_noise()

        # Receive the initial state
        steps: int = 0
        ob = self.env.reset()

        while True:
            with th.no_grad():
                self.agent.eval_mode()

                if self.steps_so_far < self.num_timesteps_before_training:
                    ac = self.env.action_space.sample()
                    ac_loc = [ac]
                else:
                    ac = self.agent.get_action(ob, explore=True, target=False)
                    if self.discrete_action:
                        # convert one-hot to integer
                        ac_loc = ac.max(1, keepdim=False)[1].cpu().numpy()
                    else:
                        ac_loc = ac.float().cpu().numpy()
                    ac = ac.cpu().numpy()[0]

                next_ob, rew, done, truncated, _ = self.env.step(ac_loc[0])
                self.buffer.add_transition(
                    ob, ac, next_ob, rew, done
                )
                log['episode_returns'].append(rew)
                steps += 1
                self.steps_so_far += 1
                ob = next_ob
                done = done or truncated

            if self.steps_so_far > self.num_timesteps_before_training and \
                    len(self.buffer) > self.batch_size:
                self.agent.train_mode()

                obs, acs, next_obs, rews, dones = self.buffer.sample(
                    self.batch_size, random=True
                )
                obs = th.from_numpy(obs).to(self.agent.device)
                acs = th.from_numpy(acs).to(self.agent.device)
                next_obs = th.from_numpy(next_obs).to(self.agent.device)
                rews = th.from_numpy(rews).to(self.agent.device)
                dones = th.from_numpy(dones).to(self.agent.device)

                # Update critic network
                critic_loss = self.agent.update_critic(
                    obs, acs, next_obs, rews, dones)
                log['critic_loss'].append(critic_loss)

                # Update policy network
                policy_loss = self.agent.update_policy(obs)
                log['policy_loss'].append(policy_loss)

                # Update target networks
                self.agent.update_target()

            if self.max_episode_steps:
                done = done or steps > self.max_episode_steps

            if done:
                return {key: sum(value)
                        if key == 'episode_returns'
                        else sum(value) / steps
                        for key, value in log.items()}

    def exec_one_episode(self,
                         episode: int = -1,
                         seed: Optional[int] = None) -> Any:
        seed = seed or self.seed
        log = defaultdict(list)

        # Receive the initial state
        steps: int = 0
        ob = self.env.reset()

        self.agent.eval_mode()

        while True:
            with th.no_grad():
                ac = self.agent.get_action(ob, explore=False, target=False)
                if self.discrete_action:
                    # convert one-hot to integer
                    ac_loc = ac.max(1, keepdim=False)[1].cpu().numpy()
                else:
                    ac_loc = ac.float().cpu().numpy()
                ob, rew, done, truncated, _ = self.env.step(ac_loc[0])
                done = done or truncated

                log['eval_returns'].append(rew)
                steps += 1
                # print(ac_loc[0])

            if self.max_episode_steps:
                done = steps > self.max_episode_steps or done

            if done:
                return {key: sum(value)
                        if key == 'eval_returns'
                        else sum(value) / steps
                        for key, value in log.items()}
