# =============================================================================
# @file   error.py
# @author Juanwu Lu
# @date   Nov-28-22
# =============================================================================
from __future__ import annotations

import warnings


class UndeclaredModule(Exception):
    """Raised when try to import missing package"""
    pass


class UnregisteredEnv(Exception):
    """Raised when try to initialize unregistered environment."""
    pass
