# ==============================================================================
# @file   inrix_data.py
# @author Juanwu Lu
# @date   Sep-15-22
# ==============================================================================
"""INRIX Traffic Forecasting Dataset API"""

import os
import pickle
from os import path as osp
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch as th
from numpy import ndarray
from pandas.tseries.holiday import USFederalHolidayCalendar
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .utils import INRIX_DTYPE, INRIX_FILENAME, XD_SEQUENCE

# Type alias
# =========================================
_PathLike = Union[str, "os.PathLike[str]"]


# TODO(Juanwu) Graph-based INRIX Data
# =========================================
class _InrixData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "x":
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

# Graph-based INRIX Dataset
# =========================================
class InrixDataset(Dataset):
    """The graph-based INRIX speed prediction dataset.
    
    Attributes:
        sigma: The spatial influence horizon in blocks.
        tau: The temporal influence horizon in minutes.
        delta: The prediction horizon in minutes.
    """

    def __init__(self,
                 root: _PathLike,
                 sigma: int = 5,
                 tau: int = 5,
                 delta: int = 1) -> None:
        """Inits InrixDataset with data filepath and attributes."""
        super().__init__()

        self.root = root
        self.sigma = sigma
        self.tau = tau
        self.delta = delta
        if len(self.processed_paths) != 0 and all(
            [osp.exists(f) for f in self.processed_paths]
        ):
            pass
        else:
            osp.makedirs(self.processed_dir)
            print("Processing...")
            self.process()

        # Load processed data
        self._file_list: List[Tuple[int, int]] = [] 
        cntr: int = 0
        for i, fp in enumerate(self.processed_paths):
            with open(fp, mode="rb") as file:
                data = pickle.load(file)
                self._file_list += [(i, cntr + j) for j in range(data["len"])]
                cntr += data["len"]
    
    @property
    def raw_file_names(self) -> List[str]:
        return INRIX_FILENAME
    
    @property
    def processed_file_names(self) -> str:
        proc_filename = []
        for name in self.raw_file_names:
            proc_filename.append(name.split(".")[0] + ".src")
        return proc_filename

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")
    
    @property
    def raw_paths(self) -> List[str]:
        return [osp.join(self.raw_dir, f) for f in self.raw_file_names]
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed")
    
    @property
    def processed_paths(self) -> List[str]:
        return [
            osp.join(self.processed_dir, f) for f in self.processed_file_names
        ]
    
    def download(self) -> None:
        raise RuntimeError("Raw INRIX data not found!")
    
    def process(self):
        calendar = USFederalHolidayCalendar()
        for fp in self.raw_paths:
            df = pd.read_csv(fp, sep=",", encoding="utf-8", dtype=INRIX_DTYPE)
            df["measurement_tstamp"] = pd.to_datetime(df["measurement_tstamp"])
            ref_dt = df["measurement_tstamp"] - df["measurement_tstamp"].min()
            holidays = calendar.holidays(
                start=df["measurement_tstamp"].min(),
                end=df["measurement_tstamp"].max()
            )

            # Preprocess time related features.
            df["days"] = ref_dt.dt.days  # Cumulative number of days.
            df["minutes"] = ref_dt.dt.seconds // 60  # Minutes in a day.
            df["is_holiday"] = df["measurement_tstamp"].isin(holidays)
            df["is_holiday"] = df["is_holiday"].map({True: 1, False: -1})

            loc_map: Dict[int, int] = {}
            for i, xd_id in enumerate(XD_SEQUENCE[::-1]):
                loc_map[xd_id] = i
            df["block_no"] = df["xd_id"].map(loc_map)
            df.sort_values(by=["block_no", "measurement_tstamp"], inplace=True)
            df = df[[
                "days", "minutes", "block_no",
                "speed", "travel_time_seconds", "is_holiday"
            ]]
            df.dropna(axis=0, how="any", inplace=True)
            df.reset_index(drop=True, inplace=True)

        # Dump data
        filename = osp.basename(fp).split(".")[0]
        data_dict = {
            "len": len(df),
            "days": df["days"].values.astype("int32"),
            "minutes": df["minutes"].values.astype("int32"),
            "block_no": df["block_no"].values.astype("int32"),
            "speed": df["speed"].values.astype("float32"),
            "travel_time": df["travel_time_seconds"].values.astype("float32"),
            "is_holiday": df["is_holiday"].values.astype("int8")
        }

        with open(
            osp.join(self.processed_dir, f"{filename:s}.src"), mode="wb"
        ) as file:
            pickle.dump(data_dict, file, protocol=3) 

    def __len__(self) -> int:
        return len(self._file_list)
    
    def __getitem__(self, idx: int) -> Any:
        _file_idx, _dat_idx = self._file_list[idx]
        with open(self.processed_paths[_file_idx], mode="rb") as file:
            data: Dict[str, ndarray] = pickle.load(file)
        
        day = data["days"][_dat_idx]
        minute = data["minutes"][_dat_idx]
        block = data["block_no"][_dat_idx]
        holiday = data["is_holiday"][_dat_idx]

        # Construct filters
        temp_filter = ((data["days"] == day) &
                       (data["minutes"] > minute - self.tau) &
                       (data["minutes"] <= minute))
        fwd_filter = (temp_filter &
                      (data["block_no"] >= block - self.sigma) &
                      (data["block_no"] < block))
        bwd_filter = (temp_filter &
                      (data["block_no"] <= block + self.sigma) &
                      (data["block_no"] > block))
        
        # Construct prediction model features
        x = np.zeros(shape=[3, self.tau], dtype="float32")
        speed = data["speed"][(data["block_no"] == block) & temp_filter]
        t_time = data["travel_time"][(data["block_no"] == block) & temp_filter]
        assert speed.shape == t_time.shape, RuntimeError(
            "Mismatched `speed` and `travel time` feature. "
            "Possible missing data found."
        )
        x[0, -len(speed):] = speed
        x[1, -len(t_time):] = t_time
        x[2, -len(speed):] = 1.0
        # TODO (Juanwu): Fix edge conditions
        y = data["speed"][(data["block_no"] == block) &
                          (data["days"] == day) &
                          (data["minutes"] == minute + self.delta)]
        if len(y) == 0:
            y = np.zeros([1, ], dtype="float32")

        # Construct shockwave model features 
        b_swf = np.zeros(shape=[self.sigma, self.tau], dtype="float32")
        b_tt = np.zeros(shape=[self.sigma, ], dtype="float32")
        speed = data["speed"][bwd_filter]
        t_time = data["travel_time"][bwd_filter & (data["minutes"] == minute)]
        if len(speed) > 0:
            # Only when observation exists
            block_idcs = data["block_no"][bwd_filter]
            idcs = np.zeros_like(block_idcs, dtype="bool")
            idcs[1:] = block_idcs[1:] != block_idcs[:-1]
            speed = np.vstack(np.split(speed, idcs.nonzero()[0]))
            b_swf[-speed.shape[0]:, -speed.shape[1]:] = speed
            b_tt[-t_time.shape[0]:] = t_time

        f_swf = np.zeros(shape=[self.sigma, self.tau], dtype="float32")
        f_tt = np.zeros(shape=[self.sigma, ], dtype="float32")
        speed = data["speed"][fwd_filter]
        t_time = data["travel_time"][fwd_filter & (data["minutes"] == minute)]
        if len(speed) > 0:
            # Only when observation exists
            block_idcs = data["block_no"][fwd_filter]
            idcs = np.zeros_like(block_idcs, dtype="bool")
            idcs[1:] = block_idcs[1:] != block_idcs[:-1]
            speed = np.vstack(np.split(speed, idcs.nonzero()[0]))
            f_swf[:speed.shape[0], :speed.shape[1]] = speed
            f_tt[:t_time.shape[0]] = t_time

        return (th.from_numpy(x).float(),
                th.tensor([minute]).float(),
                th.tensor([block]).float(),
                th.tensor([holiday]).float(),
                th.from_numpy(y).float())


if __name__ == "__main__":
    import argparse
    from torch_geometric.loader import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True,
                        help="Root directory of the dataset.")
    parser.add_argument("--sigma", type=int, default=5, required=False,
                        help="Spatial influence horizon in blocks.")
    parser.add_argument("--tau", type=int, default=5, required=False,
                        help="Temporal influence horizon in minutes.")
    parser.add_argument("--delta", type=int, default=1, required=False,
                        help="Prediction horizon in minutes.")
    
    args = vars(parser.parse_args())

    dataset = InrixDataset(**args)
    print(f"Data loaded! Data size is {len(dataset)}")
    print("=" * 79)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    try:
        sample = next(iter(data_loader))
        print(sample)
    except Exception as e:
        raise RuntimeError(e)
