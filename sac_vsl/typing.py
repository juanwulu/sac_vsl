# =============================================================================
# @file   typing.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Type variable collection."""
from __future__ import annotations

import os
import typing

Cfg = typing.TypeVar("Cfg", bound=typing.Dict[str, typing.Any])
Log = typing.TypeVar("Log", bound=typing.Dict[str, typing.Any])
PathLike = typing.TypeVar(
    "PathLike",
    bound=typing.Union[str, "os.PathLike[str]"],
)
