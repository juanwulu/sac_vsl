# ==============================================================================
# @file   utils.py
# @author Juanwu Lu
# @date   Sep-08-22
# ==============================================================================
"""Common utilities for datasets"""

import numpy as np

# ===========
# INRIX DATA
# ===========
INRIX_DTYPE = {
    "xd_id": np.uint32,
    "measurement_tstamp": str,
    "speed": np.float32,
    "average_speed": np.float32,
    "reference_speed": np.float32,
    "travel_time_seconds": np.float32,
    "confidence_score": np.float32,
    "cvalue": np.float32
}
INRIX_FILENAME = ["inrix_am.csv"]
XD_SEQUENCE = [1524495872, 1524270742, 1524323951, 1524323973, 1524323903,
            1524584569, 1524475747, 1524475699, 1524318097, 1524553047,
            1524552976, 1524552999, 461634577, 461626140, 461629140,
            396160294, 1524313670, 1524313621, 1524313548, 1524313572,
            429359943]