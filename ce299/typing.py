# =============================================================================
# @file   typing.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Type variable collection"""
from __future__ import annotations

import os
from typing import Any, Dict, TypeVar, Union

Cfg = TypeVar('Cfg', bound=Dict[str, Any])
Log = TypeVar('Log', bound=Dict[str, Any])
PathLike = TypeVar('PathLike', bound=Union[str, 'os.PathLike[str]'])
