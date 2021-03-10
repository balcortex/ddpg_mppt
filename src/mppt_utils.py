# https://matplotlib.org/3.3.3/gallery/ticks_and_spines/date_concise_formatter.html

import os
from typing import Sequence, Union, Optional

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src import utils


# AOI = "#69D2E7"
# CLP = "#A7DBD8"
# BST = "#E0E4CC"
# GGF = "#F38630"
# UFP = "#FA6900"
# DKO = "#9f571d"
# COLOR_LIST = [GGF, DKO, AOI, BST, CLP, UFP]

COLOR_LIST = [
    "#1F77B4",  # muted blue
    "#FF7F0E",  # safety orange
    "#E377C2",  # raspberry yogurt pink
    "#2CA02C",  # cooked asparagus green
    "#D62728",  # brick red
    "#9467BD",  # muted purple
    "#8C564B",  # chestnut brown
    "#7F7F7F",  # middle gray
    "#BCBD22",  # curry yellow-green
    "#17BECF",  # blue-teal
]


plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(color=COLOR_LIST),
        "text.usetex": True,
        # "legend.loc": "upper right",
        # "font.family": "CMU Serif",
    }
)

LABELS = {
    "p": {
        "ylabel": "Power (W)",
        "legend": "$P_{pv}$",
    },
    "v": {
        "ylabel": "Voltage (V)",
        "legend": "$V_{pv}$",
    },
    "d": {
        "ylabel": "Value",
        "legend": "Duty Cycle",
    },
    "pmax": {
        "ylabel": "Power (W)",
        "legend": "$P_{pv, \mathrm{max}}$",
    },
    "vmpp": {
        "ylabel": "Voltage (V)",
        "legend": "$V_{pv, \mathrm{max}}$",
    },
    "dmpp": {
        "ylabel": "Value",
        "legend": "Optimum Duty Cycle",
    },
    "g": {
        "ylabel": "Irradiance ($\mathrm{W/m^2}$)",
        "legend": None,
    },
    "ta": {
        "ylabel": "Ambient Temperature ($^\mathrm{o}$C)",
        "legend": None,
    },
    "tc": {
        "ylabel": "Cell Temperature ($^\mathrm{o}$C)",
        "legend": None,
    },
}

PAIRS = {
    "p": ["PV Maximum Power", "PV Power"],
    "d": ["Optimum Duty Cycle", "Duty Cycle"],
}


def plot_all_columns(
    df: pd.DataFrame,
    y: str = "p",
    zorder: Optional[Sequence[int]] = None,
    x_axis_date: bool = True,
    # save: str = "png",
) -> matplotlib.figure.Figure:
    """
    Plot the results of the MPPT tracking

    Parameters:
        df: DataFrame containing the series
        y: Variable to be ploted (for labeling)
        zorder: Specify the drawing priority of a line (greater = front)
        x_axis_date: Whether to label de x-ticks as date


    """
    if zorder:
        assert len(zorder) == len(df.columns)
    else:
        zorder = range(len(df.columns))

    f = plt.figure()
    ax = f.add_subplot(111)

    if x_axis_date:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    else:
        plt.xlabel("Sample", usetex=True)

    for zo, col in zip(zorder, df.columns):
        plt.plot(df.index, df[col], zorder=zo, label=str(col))

    plt.ylabel(LABELS[y]["ylabel"], usetex=True)
    plt.legend()

    return f


def dataframe_comparison_efficiency(
    df: pd.DataFrame, max_power_column: int = 0
) -> None:
    """
    Calculate the efficiency of a DataFrame with respect to one column

    Parameters:
        df: the DataFrame
        max_power_column: index of the column on which the efficiency is calculated
    """
    for idx, col in enumerate(df.columns):
        if idx == max_power_column:
            continue
        eff = utils.efficiency(df.iloc[:, max_power_column], df[col])
        print(f"{str(col)} efficiency = {eff:.2f}")


def dataframe_efficiency(
    df: pd.DataFrame,
    max_power_name: str = "PV Maximum Power",
    power_name: str = "PV Power",
) -> float:
    """
    Calculate the efficiency of a DataFrame

    Parameters:
        df: the DataFrame
        max_power_column: index of the column on which the efficiency is calculated
    """
    return utils.efficiency(df[max_power_name], df[power_name])


def plot_pair_column(
    df: pd.DataFrame,
    y: str = "p",
    reverse_zorder: bool = False,
) -> matplotlib.figure.Figure:
    """
    Plot the results of the MPPT tracking

    Parameters:
        df: DataFrame containing the series
        zorder: Specify the drawing priority of a line (greater = front)
    """
    assert y in PAIRS.keys()

    zorder = list(range(2))
    if reverse_zorder:
        zorder = list(reversed(zorder))

    f = plt.figure()
    ax = f.add_subplot(111)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.plot(df.index, df[PAIRS[y][0]], zorder=zorder[1], label=PAIRS[y][0])
    ax.plot(df.index, df[PAIRS[y][1]], zorder=zorder[0], label=PAIRS[y][1])

    ax.set_ylabel(LABELS[y]["ylabel"], usetex=True)
    ax.legend()

    return f