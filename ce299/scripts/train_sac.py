# =============================================================================
# @file   train_sac.py
# @author Juanwu Lu
# @date   Dec-2-22
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import ray
import torch as th
from ray.rllib.algorithms.sac import SAC
from ray.tune.logger import pretty_print

from ce299.env import CAVI80VSLEnv


# Global Variable
LOG_DIR = Path(os.path.abspath(__file__)).parents[2].joinpath('run_logs')


def main() -> None:
    parser = argparse.ArgumentParser()

    # Environment Arguments
    parser.add_argument('-pr', '--penetration-rate', type=float, default=0.0,
                        help='Penetration rate of connected vehicles.')
    parser.add_argument('--exp-name', type=str, default='CVI80VSL_default',
                        help='Experiment name.')
    parser.add_argument('--gui', action='store_true', default=False,
                        help='Enable SUMO GUI for visualization.')
    parser.add_argument('--continuous', action='store_true', default=False,
                        help='Enable continous action space.')

    # SAC Configurations
    parser.add_argument('--buffer-size', type=int, default=int(1e5),
                        help='Replay buffer capacity.')
    parser.add_argument('--num-episode', type=int, default=int(5e3),
                        help='Total number of episode to run.')
    parser.add_argument('--conv-activation', type=str, default='relu',
                        help='Convolutional layer activation function.')
    parser.add_argument('--fc-hiddens', nargs='+', type=int,
                        default=[256, 256], help='FC layer hidden sizes.')
    parser.add_argument('--fc-activation', type=str, default='relu',
                        help='FC layer activation function descriptor.')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Enable GPU acceleration.')
    parser.add_argument('--gpu-id', nargs='+', type=int, default=[0,],
                        help='GPU device ids to train on.')
    parser.add_argument('--save-frequency', type=int, default=100,
                        help='Number of steps to save model checkpoint.')
    parser.add_argument('--evaluation-interval', type=int, default=None,
                        help='Evaluate frequency in number of iterations.')
    args = vars(parser.parse_args())

    if args['gpu'] and th.cuda.is_available():
        num_gpus: int = len(args.get('gpu_id', [0, ]))
        ray.init(num_gpus=num_gpus)
    else:
        num_gpus: int = 0
        ray.init()

    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    os.makedirs(os.path.join(LOG_DIR, args['exp_name']), exist_ok=True)
    os.makedirs(
        os.path.join(LOG_DIR, args['exp_name'], 'checkpoint'), exist_ok=True
    )

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='# %(asctime)s %(name)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout,
                        datefmt='%Y%m%d %H:%M:%S %p')

    algo = SAC(config={
        # ===Environment Settings===
        'env': CAVI80VSLEnv,
        'env_config': {
            'penetration_rate': args['penetration_rate'],
            'exp_name': args['exp_name'],
            'gui': args['gui'],
            'discrete': not args['continuous']
        },
        'horizon': 24,

        # ===Model Settings===
        'num_gpus': num_gpus,
        'framework': 'torch',
        'q_model_config': {
            'conv_filters': None,
            'conv_activation': args['conv_activation'],
            'fcnet_hiddens': args['fc_hiddens'],
            'fcnet_activation': args['fc_activation'],
            'custom_model': None,
            'custom_model_config': {}
        },
        'policy_model_config': {
            'conv_filters': None,
            'conv_activation': args['conv_activation'],
            'fcnet_hiddens': args['fc_hiddens'],
            'fcnet_activation': args['fc_activation'],
            'custom_model': None,
            'custom_model_config': {}
        },
        'replay_buffer_config': {
            'capacity': args['buffer_size']
        },

        # ===Evaluation Settings===
        'evaluation_interval': args['evaluation_interval'],
        'evaluation_duration': 3,

    })

    for episode in range(1, args['num_episode'] + 1):
        result = algo.train()
        logger.info(pretty_print(result))

        if episode % args['save_frequency'] == 0:
            checkpoint = algo.save(
                os.path.join(LOG_DIR, args['exp_name'], 'checkpoint'),
                prevent_upload=True
            )
            logger.info('Checkpoint saved at %s', checkpoint)


if __name__ == '__main__':
    main()
