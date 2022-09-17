# ==============================================================================
# @file   model.py
# @author Juanwu Lu
# @date   Sep-16-22
# ==============================================================================
"""Block Velocity Prediction Model."""
from typing import Any, Dict, Tuple, Union

import torch as th
from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
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
            

    def forward(self, data: Union[Batch, Data]) -> Tuple[Tensor, Tensor]:
        x = data["x"].float()
        loc = data["loc"].long()
        time = data["time"].float()
        y = data["y"].float()

        assert len(x.shape) == 3, ValueError(
            f"Required speed input to be (B, C, L), but got {x.shape}."
        )
        assert len(loc.shape) == 2, ValueError(
            f"Required location to be (B, 1), but got {loc.shape}."
        )
        assert len(time.shape) == 2, ValueError(
            f"Required time feature to be (B, Ft), but got {time.shape}."
        )

        static_v = self.static_estimator(loc, time)
        dyn_v = self.dynamic_estimator(x)
        out = static_v + dyn_v

        loss = self._loss(out, y)

        return out, loss

    def _loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.mse_loss(y_pred, y_true, reduction="mean")