# ==============================================================================
# @file   optimizer.py
# @author Juanwu Lu
# @date   Sep-14-22
# ==============================================================================
"""Building Gradient Descent Optimization Solvers"""
import torch as th
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Union
from yacs.config import CfgNode


def build_optimizer(model: nn.Module, opt_cfg: CfgNode) -> Optimizer:
    r"""Build gradient descent optimization solver.

    Args:
        model:  A `nn.Module` object the model to train.
        opt_cfg: A `CfgNode` object optimizer configuration node.
    
    Returns:
        An `Optimizer` object gradient descent optimization solver.
    """
    opt_name = opt_cfg.NAME
    lr = opt_cfg.INIT_LR
    if opt_name == "adam":
        optimizer = th.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        momentum = opt_cfg.MOMENTUM
        nesterov = opt_cfg.NESTEROV
        optimizer = th.optim.Adam(
            model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov
        )
    else:
        raise RuntimeError(f"Optimizer {opt_name} is not currently supported.")
    
    return optimizer

def build_scheduler(
    optimizer: Optimizer, sched_cfg: CfgNode
) -> Optional[_LRScheduler]:
    r"""Build learning rate scheduler.
    
    Args:
        optmizer: An `Optimizer` object to apply scheduler on.
        sched_cfg: A `CfgNode` object learning rate scheduler configuration node.
    
    Returns:
        A `_LRScheduler` object learning rate scheduler.
    """
    scheduler_name = sched_cfg.NAME
    if scheduler_name == "unchange":
        return None
    elif scheduler_name == "step":
        gamma = sched_cfg.STEP_GAMMA
        step_size = sched_cfg.STEP_SIZE
        return th.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    elif scheduler_name == "multi_steps":
        gamma = sched_cfg.STEP_GAMMA
        milestones = sched_cfg.MULTI_STEP_MILESTONE
        return th.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones, gamma
        )
    elif scheduler_name == "reduce_on_plateau":
        # TODO (Juanwu): additional keyword arguments
        gamma = sched_cfg.STEP_GAMMA
        patience = sched_cfg.ROP_PATIENCE
        return th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=gamma, patience=patience
        )
    elif scheduler_name == "cyclic_lr":
        scheduler = th.optim.lr_scheduler.CyclicLR(
            optimizer, max_lr=sched_cfg.CYCLIC_MAX_LR,
            step_size_up=sched_cfg.CYCLIC_STEP_SIZE_UP,
            step_size_down=sched_cfg.CYCLIC_STEP_SIZE_DOWN,
        )
        return scheduler
    else:
        raise RuntimeError(
            f"Scheduler {scheduler_name} is currently not supported."
        )