# ==============================================================================
# @file   config.py
# @author Juanwu Lu
# @date   Sep-14-22
# ==============================================================================
"""Default Configurations for Model Training"""
import os
import warnings
from os import path as osp
from yacs.config import CfgNode 

# Global configuration handle
# -----------------------------------------
__C = CfgNode()

# TODO (Juanwu): Add configurations