# ==============================================================================
# @file   model.py
# @author Juanwu Lu
# @date   Sep-16-22
# ==============================================================================
"""Block Velocity Prediction Model."""
from typing import Any, Dict

import torch as th
from torch import nn, Tensor
from i24_vsl.model.layers import StaticEvol, DynEvol


class Model(nn.Module):
    """Prediction framework for short-term block velocity prediction.
    
    Args:
    """

    def __init__(self, config: Dict[str, Dict[str, Any]]) -> None:
        super().__init__()

        # Speed estimators
        self.static_estimator = StaticEvol(**config["static_estimator"])
        self.dynamic_estimator = DynEvol(**config["dynamic_estimator"])

        # Shockwave propagation model
        if config["enable_shockwave"]:
            raise NotImplementedError("Shockwave model is under building...")
            

    def forward(self, x: Tensor, loc: Tensor, time: Tensor) -> Tensor:
        return

    def _loss(self) -> Tensor:
        return