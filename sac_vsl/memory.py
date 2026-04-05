# =============================================================================
# @file   memory.py
# @author Juanwu Lu
# @date   Nov-28-22
# =============================================================================
from __future__ import annotations

import copy
import dataclasses
import typing

import numpy as np


def add_noise(data: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
    data = copy.deepcopy(data)
    data_mean = np.mean(data, axis=0)
    data_mean[data_mean == 0] = 1e-6
    data_std = data_mean * noise_scale
    for i, _ in enumerate(data_mean):
        data[:, i] = np.copy(
            data[:, i]
            + np.random.normal(
                loc=0.0, scale=np.absolute(data_std[i]), size=(data.shape[0],)
            )
        )

    return data


def convert_sequence_of_paths(paths: typing.Sequence[Path]) -> tuple:
    states = np.concatenate([path.observation for path in paths])
    actions = np.concatenate([path.action for path in paths])
    next_states = np.concatenate([path.next_observation for path in paths])
    rewards = np.concatenate([path.reward for path in paths])
    dones = np.concatenate([path.done for path in paths])

    return (states, actions, next_states, rewards, dones)


# Trajectory type
@dataclasses.dataclass
class Path:
    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    reward: np.ndarray
    done: np.ndarray

    @property
    def length(self) -> int:
        if self.reward is None:
            return 0
        return len(self.reward)


@dataclasses.dataclass
class BaseBuffer:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    max_size: int = 100000
    paths: list[Path] = dataclasses.field(default_factory=lambda: [])

    def __len__(self) -> int:
        if self.observations is None:
            return 0
        return self.observations.shape[0]

    def add_paths(
        self,
        paths: typing.Sequence[Path],
        noised: bool = False,
    ) -> None:
        raise NotImplementedError

    def add_transition(
        self,
        ob: np.ndarray,
        ac: np.ndarray,
        next_ob: np.ndarray,
        rew: np.ndarray,
        done: np.ndarray,
        noised: bool = False,
    ) -> None:
        if isinstance(ac, int):
            action = np.asarray([ac], dtype="float32")
        elif isinstance(ac, np.ndarray) and len(ac.shape) == 1:
            action = ac[None, ...]
        else:
            action = ac

        path = Path(
            observation=np.asarray([ob], dtype="float32"),
            action=action,
            next_observation=np.asarray([next_ob], dtype="float32"),
            reward=np.asarray([rew], dtype="float32"),
            done=np.asarray([done], dtype="int64"),
        )
        self.add_paths([path], noised=noised)

    def sample(self, batch_size: int, random: bool = False) -> tuple:
        raise NotImplementedError


# Vanilla Replay Buffer
# =====================
class ReplayBuffer(BaseBuffer):
    def add_paths(
        self,
        paths: typing.Sequence[Path],
        noised: bool = False,
    ) -> None:
        self.paths += list(paths)

        # Convert a sequence of rollouts
        obs, act, next_obs, r, d = convert_sequence_of_paths(paths)

        # Add noise to observations
        if noised:
            obs = add_noise(obs)
            next_obs = add_noise(next_obs)

        # Update data pointer
        if self.observations is None:
            self.observations = obs[-self.max_size :]
            self.actions = act[-self.max_size :]
            self.next_observations = next_obs[-self.max_size :]
            self.rewards = r[-self.max_size :]
            self.dones = d[-self.max_size :]
        else:
            self.observations = np.concatenate([self.observations, obs])[
                -self.max_size :
            ]
            self.actions = np.concatenate([self.actions, act])[
                -self.max_size :
            ]
            self.next_observations = np.concatenate(
                [self.next_observations, next_obs]
            )[-self.max_size :]
            self.rewards = np.concatenate([self.rewards, r])[-self.max_size :]
            self.dones = np.concatenate([self.dones, d])[-self.max_size :]

    def sample(
        self,
        batch_size: int,
        random: bool = False,
        rand_idcs: typing.Sequence[int] | None = None,
    ) -> tuple[np.ndarray, ...]:
        assert (
            self.observations.shape[0]
            == self.actions.shape[0]
            == self.next_observations.shape[0]
            == self.rewards.shape[0]
            == self.dones.shape[0]
        ), RuntimeError("Unmatched size MDP tuple elements found!")

        if random:
            if rand_idcs is None:
                # Randomly sample data from buffer
                _idcs = np.random.permutation(len(self)).tolist()
                _idcs = _idcs[:batch_size]
            else:
                assert len(rand_idcs) >= batch_size
                _idcs = rand_idcs[:batch_size]

            return (
                self.observations[_idcs],
                self.actions[_idcs],
                self.next_observations[_idcs],
                self.rewards[_idcs],
                self.dones[_idcs],
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
        return f"{self.__class__.__name__} ({len(self):d})"
