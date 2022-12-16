# =============================================================================
# @file   env.py
# @author Juanwu Lu
# @date   Dec-2-22
# =============================================================================
from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gym.core import ActType, Env, ObsType
from gym.spaces import Box, Discrete
from ray.rllib.env.env_context import EnvContext

# SUMO Traci
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import sumolib
    import traci
else:
    sys.exit('Please declare envrionment variable "SUMO_HOME".')

ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


class CAVI80VSLEnv(Env):
    """I80 Emeryville connected vehicle variable speed limit environment.

    This environment is associated with the variable speed limit control
    problem on Interstate I80, Emeryville, CA. Scenario is generated using
    NGSIM I80 Emeryville Dataset.

    - Action Space

        The action is a float array of shape `(1, )` with elements in range
        `[15.64, 29.06]`, corresponding to 35~65 mph.

    - Observation Space

        The observation is an array of shape `[2, H, W]`, where the first
        dimension is a heatmap of vehicle position given specific time and
        the second dimension is a timestep encoding.

    - Rewards

        The reward is the negative sum of mainline vehicle speed variation
        and mean time loss of all the vehicles at current time step.

    - Starting State

        The starting state is captured after warming up for 300 seconds.

    - Episode Termination

        The episode terminates after 3600 simulation seconds.

    - Arguments
    """
    metadata: Dict[str, Any] = {'render_modes': ['human']}

    def __init__(self, config: EnvContext) -> None:
        super().__init__()

        self.penetration_rate = config.get('penetration_rate', 0.0)
        assert self.penetration_rate in [0.0, 0.1, 0.2, 0.5, 1.0], ValueError(
            'Expect penetration rate to be within [0.0, 0.1, 0.2, 0.5, 1.0], ',
            f'but got {self.penetration_rate}.'
        )
        self.step_length = config.get('step_length', 1.0)
        self.exp_name = config.get('exp_name', 'default')
        self.raster_length = config.get('raster_length', 20.0)
        self.sumo_binary = 'sumo-gui' if config.get('gui', False) else 'sumo'
        self.sumo_cfg = os.path.join(ASSET_DIR, 'I80', 'i80.sumo.cfg')
        self.route_file = os.path.join(
            ASSET_DIR, 'I80',
            f'i80.rou_pr_{int(self.penetration_rate * 100)}.xml'
        )
        if not os.path.isdir(os.path.join(ASSET_DIR, 'I80', 'output')):
            # Output directory for detectors
            os.makedirs(os.path.join(ASSET_DIR, 'I80', 'output'))

        # Observation parameters
        self._net = sumolib.net.readNet(
            os.path.join(ASSET_DIR, 'I80', 'i80.net.xml'))
        self._obs_edges = [
            # Mainline edges
            'i80_upstream_n',
            'i80_weaving_n',
            'i80_weaving_ext_n',
            'i80_shrink_n',
            # Ramps
            'powell_on_ramp',
            'ashby_off_ramp'
        ]
        self._vsl_edges = [
            'i80_load_n',
            'i80_upstream_n'
        ]
        self._rew_edges = [
            'i80_weaving_n'
        ]
        self._edge_left_grid_map = {}
        self._edge_right_grid_map = {}
        _grid = 0
        for edge_id in self._obs_edges:
            edge = self._net.getEdge(edge_id)
            if 'i80' in edge_id:
                self._edge_left_grid_map[edge_id] = _grid
                _grid += math.ceil(edge.getLength() / self.raster_length)
                self._edge_right_grid_map[edge_id] = _grid
            if edge_id == 'powell_on_ramp':
                self._edge_left_grid_map[edge_id] = \
                    self._edge_left_grid_map['i80_weaving_n'] - \
                    math.ceil(edge.getLength() / self.raster_length)
                self._edge_right_grid_map[edge_id] = \
                    self._edge_left_grid_map['i80_weaving_n']
            if edge_id == 'ashby_off_ramp':
                self._edge_left_grid_map[edge_id] = \
                    self._edge_right_grid_map['i80_weaving_ext_n']
                self._edge_right_grid_map[edge_id] = \
                    self._edge_right_grid_map['i80_weaving_ext_n'] + \
                    math.ceil(edge.getLength() / self.raster_length)
        self.obs_width = max(self._edge_right_grid_map.values())
        self.obs_height = max(
            self._net.getEdge(edge_id).getLaneNumber()
            for edge_id in self._obs_edges if 'i80' in edge_id
        ) + 1
        self.observation_space = Box(
            low=0.0, high=float('inf'),
            shape=(self.obs_height * 6, self.obs_width, 3)
        )

        # Action and Reward parameters
        if config.get('discrete', True):
            self.action_list = [
                15.64, 17.88, 20.12, 22.35, 24.59, 26.82, 29.06
            ]
            self.action_space = Discrete(n=len(self.action_list))
        else:
            self.action_list = None
            self.action_space = Box(low=15.64, high=29.06, shape=(1, ))

    def close(self) -> None:
        traci.close(False)

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> ObsType:
        super().reset(seed=seed, options=options)

        if seed is not None:
            sumo_cmd = [
                self.sumo_binary,
                '-c', self.sumo_cfg,
                '--route-files', self.route_file,
                '--start',
                '--seed', str(seed),
                '--step-length', str(float(self.step_length)),
                '--quit-on-end'
            ]
        else:
            sumo_cmd = [
                self.sumo_binary,
                '-c', self.sumo_cfg,
                '--route-files', self.route_file,
                '--step-length', str(float(self.step_length)),
                '--start',
                '--quit-on-end'
            ]

        traci.start(sumo_cmd, label='sim_' + str(time.time()))
        self.warm_up()

        curr_time = traci.simulation.getTime()
        obs = []
        for timestep in range(1, 7, 1):
            while traci.simulation.getTime() < curr_time + 30.0:
                traci.simulationStep()

            curr_obs = self.get_observation(timestep=timestep)
            obs.append(curr_obs)
            curr_time += 30.0

        obs = np.concatenate(obs, axis=0)
        self.curr_vsl = 24.59

        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, Dict]:
        """Take action to apply speed limit to all the connected vehicles."""
        if self.action_list is None:
            assert action.shape == self.action_space.shape, ValueError(
                f'Incosistent action shape, expect {self.action_space.shape}, '
                f'but got {action.shape}.'
            )
            # NOTE: Continuous VSL
        else:
            action = self.action_list[action]

        if isinstance(action, float):
            vsl = action
        elif isinstance(action, np.ndarray) and len(action.shape) == 1:
            vsl = action[0]
        else:
            raise ValueError(f'Invalid action value {action}!')

        # penalty inconsistent vsl
        penalty = -min(abs(vsl - self.curr_vsl) / 4.17, 1.0)
        self.set_vsl(vsl)

        curr_time = traci.simulation.getTime()
        obs = []
        # reward = []
        for timestep in range(1, 7, 1):
            while traci.simulation.getTime() < curr_time + 10.0:
                traci.simulationStep()
                # reward.append(self.get_reward())

            curr_obs = self.get_observation(timestep=timestep)
            obs.append(curr_obs)
            curr_time += 10.0

        obs = np.concatenate(obs, axis=0)
        # reward = np.mean(reward)  # Return the average reward in the interval
        reward = self.get_reward()
        reward = reward + 0.2 * penalty
        done = traci.simulation.getTime() >= 5700

        if done:
            self.close()

        return obs, reward, done, {}

    def get_observation(self, timestep: int = 0) -> np.ndarray:
        obs = np.zeros([self.obs_height, self.obs_width, 3], 'float32')
        # Last dimension: time stamp encoding
        obs[:, :, 2] = np.ones_like(obs[:, :, 2]) * timestep

        for edge_id in self._obs_edges:
            edge = self._net.getEdge(edge_id)
            edge_len = edge.getLength()
            num_lanes = edge.getLaneNumber()
            _left = self._edge_left_grid_map[edge_id]
            _right = self._edge_right_grid_map[edge_id]

            for idx in range(num_lanes):
                row_obs = np.zeros([1, _right - _left], 'float32')
                row_cav_obs = np.zeros([1, _right - _left], 'float32')
                lane_id = '_'.join([edge_id, str(idx)])
                veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                veh_pos = list(
                    map(
                        lambda x, eid=edge_id, elen=edge_len:
                        traci.vehicle.getDrivingDistance(x, eid, elen),
                        veh_ids
                    )
                )
                for v_id, v_pos in zip(veh_ids, veh_pos):
                    veh_len = traci.vehicle.getLength(v_id)
                    rear_grid = math.floor(v_pos / self.raster_length)
                    front_grid = math.ceil(v_pos / self.raster_length)
                    if rear_grid + 1 == front_grid:
                        # Case 1: vehicle is within a grid
                        row_obs[0, rear_grid] += 1
                        if 'cav' in v_id:
                            row_cav_obs[0, rear_grid] += 1
                    if rear_grid + 2 == front_grid:
                        # Case 2: vehicle crosses consecutive grids
                        ref = self.raster_length * (rear_grid + 1)
                        if rear_grid >= 0:
                            # Clip left grid
                            ratio = (ref - v_pos + veh_len / 2) / veh_len
                            row_obs[0, rear_grid] += ratio
                            if 'cav' in v_id:
                                row_cav_obs[0, rear_grid] += ratio
                        if rear_grid + 1 <= _right:
                            # Clip right grid
                            ratio = (v_pos - ref + veh_len / 2) / veh_len
                            row_obs[0, rear_grid + 1] += ratio
                            if 'cav' in v_id:
                                row_cav_obs[0, rear_grid + 1] += ratio

                if 'ramp' in edge_id:
                    obs[-1, _left:_right, 0] = row_obs
                    obs[-1, _left:_right, 1] = row_cav_obs
                else:
                    obs[idx, _left:_right, 0] = row_obs
                    obs[idx, _left:_right, 1] = row_cav_obs

        return obs

    def get_reward(self) -> float:
        # Harmonization: Minimize variation of mainline vehicles' speed
        # var_reward = -np.var(
        #     [traci.vehicle.getSpeed(v) for v in self.mainline_vehicles]
        # )

        # Efficiency: Maximize the approximate throughput flow
        tt_reward = -traci.multientryexit.getLastIntervalMeanTravelTime(
            'weaving_e3'
        )
        # speed_reward = np.mean([
        #     np.quantile([
        #         traci.lane.getLastStepMeanSpeed(lane.getID()) /
        #         traci.lane.getMaxSpeed(lane.getID())
        #         for lane in self._net.getEdge(edge).getLanes()
        #     ], q=0.15)
        #     for edge in self._rew_edges
        # ])

        # Congestion reward: Minimize the jam distance in lanearea detectors
        # jam_reward = np.sum([
        #     traci.lanearea.getJamLengthVehicle(f'e2_{i}')
        #     for i in range(6)
        # ])

        reward = 0.05 * tt_reward

        return reward

    def set_vsl(self, vsl: float) -> None:
        # for vehicle in self.vehicles:
        #     traci.vehicle.setSpeed(vehicle, -1)

        # for vehicle in self.action_vehicles:
        #     if 'cav' in vehicle:
        #         # traci.vehicle.setMaxSpeed(vehicle, vsl)
        #         traci.vehicle.setSpeed(vehicle, vsl)
        for edge in self._vsl_edges:
            for lane in self._net.getEdge(edge).getLanes():
                traci.lane.setMaxSpeed(lane.getID(), vsl)
        self.curr_vsl = vsl

    def warm_up(self) -> None:
        """Warm up simulation before getting the starting state."""
        while traci.simulation.getTime() < 120.0:
            traci.simulationStep()

    @property
    def action_vehicles(self) -> List[str]:
        ml_veh_ids = []
        for edge_id in self._vsl_edges:
            edge = self._net.getEdge(edge_id)
            for lane in edge.getLanes():
                ml_veh_ids += traci.lane.getLastStepVehicleIDs(lane.getID())

        return ml_veh_ids

    @property
    def vehicles(self) -> List[str]:
        veh_ids = []
        for edge in self._net.getEdges():
            for lane in edge.getLanes():
                veh_ids += traci.lane.getLastStepVehicleIDs(lane.getID())

        return veh_ids


if __name__ == '__main__':
    env = CAVI80VSLEnv(config=dict(
        penetration_rate=0.1,
        gui=True
    ))
    obs = env.reset(seed=42)
    done = False
    while not done:
        next_obs, rew, done, _ = env.step(env.action_space.sample())
        obs = next_obs
    env.close()
