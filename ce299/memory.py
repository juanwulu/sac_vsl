# =============================================================================
# @file   memory.py
# @author Juanwu Lu
# @date   Nov-28-22
# =============================================================================
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


def add_noise(data: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
    data = deepcopy(data)
    data_mean = np.mean(data, axis=0)
    data_mean[data_mean == 0] = 1e-6
    data_std = data_mean * noise_scale
    for i, _ in enumerate(data_mean):
        data[:, i] = np.copy(
            data[:, i] + np.random.normal(
                loc=0.0,
                scale=np.absolute(data_std[i]),
                size=(data.shape[0], )
            )
        )


def convert_sequence_of_paths(paths: Sequence[Path]) -> Tuple:
    states = np.concatenate([path.observation for path in paths])
    actions = np.concatenate([path.action for path in paths])
    next_states = np.concatenate([path.next_observation for path in paths])
    rewards = np.concatenate([path.reward for path in paths])
    dones = np.concatenate([path.done for path in paths])

    return (states, actions, next_states, rewards, dones)


# Trajectory type
@dataclass
class Path:
    observation: np.ndarray = None
    action: np.ndarray = None
    next_observation: np.ndarray = None
    reward: np.ndarray = None
    done: np.ndarray = None

    @property
    def length(self) -> int:
        return len(self.reward)


@dataclass
class BaseBuffer:
    max_size: int = 100000
    paths: List[Path] = field(default_factory=lambda: [])
    observations: Union[Sequence[np.ndarray], np.ndarray] = None
    actions: Union[Sequence[np.ndarray], np.ndarray] = None
    next_observations: Union[Sequence[np.ndarray], np.ndarray] = None
    rewards: Union[Sequence[np.ndarray], np.ndarray] = None
    dones: Union[Sequence[np.ndarray], np.ndarray] = None

    def __len__(self) -> int:
        if self.observations is None:
            return 0
        else:
            return self.observations.shape[0]

    def add_paths(self, paths: Sequence[Path], noised: bool = False) -> None:
        raise NotImplementedError

    def add_transition(self,
                       ob: np.ndarray,
                       ac: np.ndarray,
                       next_ob: np.ndarray,
                       rew: np.ndarray,
                       done: np.ndarray,
                       noised: bool = False) -> None:

        if isinstance(ac, int):
            action = np.asarray([ac], dtype='float32')
        elif isinstance(ac, np.ndarray) and len(ac.shape) == 1:
            action = ac[None, ...]
        else:
            action = ac

        path = Path(observation=np.asarray([ob], dtype='float32'),
                    action=action,
                    next_observation=np.asarray([next_ob], dtype='float32'),
                    reward=np.asarray([rew], dtype='float32'),
                    done=np.asarray([done], dtype='int64'))
        self.add_paths([path], noised=noised)

    def sample(self, batch_size: int, random: bool = False) -> Tuple:
        raise NotImplementedError


# Vanilla Replay Buffer
# =====================
class ReplayBuffer(BaseBuffer):

    def add_paths(self, paths: Sequence[Path], noised: bool = False) -> None:
        self.paths += list(paths)

        # Convert a sequence of rollouts
        obs, act, next_obs, r, d = convert_sequence_of_paths(paths)

        # Add noise to observations
        if noised:
            obs = add_noise(obs)
            next_obs = add_noise(next_obs)

        # Update data pointer
        if self.observations is None:
            self.observations = obs[-self.max_size:]
            self.actions = act[-self.max_size:]
            self.next_observations = next_obs[-self.max_size:]
            self.rewards = r[-self.max_size:]
            self.dones = d[-self.max_size:]
        else:
            self.observations = np.concatenate(
                [self.observations, obs])[-self.max_size:]
            self.actions = np.concatenate([self.actions, act])[-self.max_size:]
            self.next_observations = np.concatenate(
                [self.next_observations, next_obs])[-self.max_size:]
            self.rewards = np.concatenate([self.rewards, r])[-self.max_size:]
            self.dones = np.concatenate([self.dones, d])[-self.max_size:]

    def sample(self,
               batch_size: int,
               random: bool = False,
               rand_idcs: Optional[Sequence[int]] = None
               ) -> Tuple[np.ndarray, ...]:
        assert (
            self.observations.shape[0] ==
            self.actions.shape[0] ==
            self.next_observations.shape[0] ==
            self.rewards.shape[0] ==
            self.dones.shape[0]
        ), RuntimeError("Unmatched size MDP tuple elements found!")

        if random:
            if rand_idcs is None:
                # Randomly sample data from buffer
                rand_idcs = np.random.permutation(len(self))[:batch_size]
            else:
                assert len(rand_idcs) >= batch_size
                rand_idcs = rand_idcs[:batch_size]

            return (
                self.observations[rand_idcs],
                self.actions[rand_idcs],
                self.next_observations[rand_idcs],
                self.rewards[rand_idcs],
                self.dones[rand_idcs]
            )
        else:
            # Sample from recent paths
            cntr: int = 0
            idx: int = -1
            while cntr <= batch_size:
                recent_sample = self.paths[idx]
                cntr += recent_sample.length
                idx -= 1
            return convert_sequence_of_paths(self.paths[idx:])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} ({len(self):d})'
