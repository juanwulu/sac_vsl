# ==============================================================================
# @file   inrix_plot.py
# @author Juanwu Lu
# @date   Sep-8-22
# ==============================================================================
"""Plot raw INRIX data and generate """

from __future__ import print_function

import os
from datetime import datetime
from os import path as osp
from pathlib import Path
from re import I
from typing import Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from tqdm import tqdm

from common import INRIX_DTYPE, INRIX_TIME_REF, XD_SEQUENCE


# Matplotlib rc Configurations
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600 

def _plot_day(pivot: DataFrame,
              date: datetime,
              totd: str,
              ax: Optional[plt.Axes] = None,
              **kwargs) -> Tuple[plt.Axes, str]:

    # Attributes
    try:
        date_str = date.strftime('%Y-%m-%d %A')
    except Exception as e:
        raise RuntimeError(e)

    ax: plt.Axes = sns.heatmap(data=pivot, ax=ax,
                                    cmap=plt.get_cmap(
                                        kwargs.get('cmap', 'RdYlGn')),
                                    vmax=kwargs.get('vmax', 80),
                                    vmin=kwargs.get('vmin', 10))
    ax.invert_yaxis()
    ax.set_title(
        f'Velocity Contour of I24-W: {date_str} {totd}\n'
        'Unit: mph                           Step: 1min'
    )
    name: str = f'{date_str}_{totd}'
    
    return ax, name

def plot(**kwargs) -> None:
    r"""Plot raw INRIX data."""

    df: DataFrame = pd.read_csv(kwargs['data'], sep=',',
                                encoding='utf-8', dtype=INRIX_DTYPE)
    df['measurement_tstamp'] = pd.to_datetime(df['measurement_tstamp'])
    df['date'] = df['measurement_tstamp'].dt.date
    ref_time_delta = df['measurement_tstamp'] - datetime(**INRIX_TIME_REF)
    df['seconds_in_day'] = ref_time_delta.dt.seconds

    for date, g in tqdm(df.groupby(by='date'),
                     desc="Traversing days",
                     position=0,
                     leave=False):
        g.rename(columns=dict(xd_id='Segments'), inplace=True)
        g['Time'] = pd.to_datetime(
            g['measurement_tstamp'], format='%H:%M:%S').dt.time
        totd: str = 'AM' if (g['seconds_in_day'] < 43200).all() else 'PM'
        speed_pivot = pd.pivot_table(
            data=g, columns='Time', index='Segments', values='speed'
        )
        speed_pivot = speed_pivot.loc[XD_SEQUENCE]

        plot_dir = kwargs.get('plot_dir') or osp.join(
            Path(kwargs['data']).parents[2], 'img', 'raw_daily_speed')
        if not osp.isdir(plot_dir):
            os.makedirs(plot_dir)
        fig, ax = plt.subplots(1, 1)
        ax, name = _plot_day(speed_pivot, date, totd, ax)
        fig.tight_layout()
        fig.savefig(osp.join(plot_dir, f'{name}_1min.png'))
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Data file path.')
    parser.add_argument('--day', action='store_true',
                        help='Split data on daily basis')
    parser.add_argument('--plot-dir', type=str, default=None,
                        help='Plot saving directory.')
    args = parser.parse_args()
    params = vars(args)

    plot(**params)