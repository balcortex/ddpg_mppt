from typing import Optional
from src.utils import read_weather_csv
import pandas as pd
import matplotlib.figure as figure
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os

Figure = figure.Figure

plt.rcParams.update({"text.usetex": True})

PATHS = {
    "train": os.path.join("data", "weather_real_train.csv"),
    "val": os.path.join("data", "weather_real_val.csv"),
    "test": os.path.join("data", "weather_real_test.csv"),
}

COLOR_LIST = [
    "#1F77B4",  # muted blue
    "#0f5485",  # muted blue darker
    "#FF7F0E",  # safety orange
    "#db6700",  # safety orange darker
]


def df_day_from_csv(dataset: str, day_idx: int = 0) -> pd.DataFrame:
    """
    dataset: ['train', 'val', 'test']
    """
    assert dataset in PATHS.keys(), "No dataset matching"

    df = read_weather_csv(PATHS[dataset], format=None)
    df_day = [g[1] for g in df.groupby(df.index.date)][day_idx]

    return df_day


def plot_weather_df(df: pd.DataFrame, g_top_lim: Optional[float] = None) -> Figure:
    assert "['Irradiance' 'Temperature']" == str(
        df.columns.values
    ), "The dataframe must contain columns `Irradiance` and `Temperature`"

    f = plt.figure()
    ax = f.add_subplot(111)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.plot(df.index, df["Irradiance"], color=COLOR_LIST[0])
    ax.set_ylabel("Irradiance ($\mathrm{W/m^2}$)", color=COLOR_LIST[1])
    if g_top_lim:
        ax.set_ylim(top=g_top_lim)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(df.index, df["Temperature"], color=COLOR_LIST[2])
    ax2.set_ylabel("Ambient Temperature ($\mathrm{^o}$C)", color=COLOR_LIST[3])
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    f.tight_layout()

    return f


if __name__ == "__main__":
    df = df_day_from_csv("train", day_idx=0)
    f = plot_weather_df(df, g_top_lim=1080)
    f.savefig("weather_plot_sample.pdf", bbox_inches="tight")
