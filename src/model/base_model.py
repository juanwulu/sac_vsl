# =============================================================================
# @file   base_model.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Base model class declaration"""
from __future__ import annotations

import torch as th

from src.typing import Cfg


class BaseModel(th.nn.Module):

    def forward(self) -> th.Tensor:
        raise NotImplementedError

    def __init__(self, config: Cfg) -> None:
        super().__init__()
