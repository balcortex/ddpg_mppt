# https://matplotlib.org/3.3.3/gallery/ticks_and_spines/date_concise_formatter.html

import os
from typing import Sequence, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.utils import efficiency, load_dict

# CB91_Blue = "#2CBDFE"
# CB91_Green = "#47DBCD"
# CB91_Pink = "#F3A0F2"
# CB91_Purple = "#9D2EC5"
# CB91_Violet = "#661D98"
# CB91_Amber = "#F5B14C"

# COLOR_LIST = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Purple, CB91_Violet]


AOI = "#69D2E7"
CLP = "#A7DBD8"
BST = "#E0E4CC"
GGF = "#F38630"
UFP = "#FA6900"
DKO = "#9f571d"
COLOR_LIST = [GGF, DKO, AOI, BST, CLP, UFP]


plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(color=COLOR_LIST),
        "text.usetex": True,
        # "legend.loc": "upper right",
        # "font.family": "CMU Serif",
    }
)

LABELS = {
    "PV Power": {
        "ylabel": "Power (W)",
        "legend": "$P_{pv}$",
    },
    "PV Voltage": {
        "ylabel": "Voltage (V)",
        "legend": "$V_{pv}$",
    },
    "Duty Cycle": {
        "ylabel": "Value",
        "legend": "Duty Cycle",
    },
    "PV Maximum Power": {
        "ylabel": "Power (W)",
        "legend": "$P_{pv, \mathrm{max}}$",
    },
    "PV Optimum Voltage": {
        "ylabel": "Voltage (V)",
        "legend": "$V_{pv, \mathrm{max}}$",
    },
    "Optimum Duty Cycle": {
        "ylabel": "Value",
        "legend": "Optimum Duty Cycle",
    },
    "Irradiance": {
        "ylabel": "Irradiance ($\mathrm{W/m^2}$)",
        "legend": None,
    },
    "Ambient Temperature": {
        "ylabel": "Ambient Temperature ($^\mathrm{o}$C)",
        "legend": None,
    },
    "Cell Temperature": {
        "ylabel": "Cell Temperature ($^\mathrm{o}$C)",
        "legend": None,
    },
}

PAIRS = {
    "p": ("PV Power", "PV Maximum Power"),
    "v": ("PV Voltage", "PV Optimum Voltage"),
    "dc": ("Duty Cycle", "Optimum Duty Cycle"),
    "g": ("Irradiance",),
    "t": ("Ambient Temperature",),
    "ct": ("Cell Temperature",),
}


class PlotHandler:
    def __init__(self, filepath: str):
        self.df = self.read_csv(filepath)
        self.path = os.path.split(filepath)[0]
        self.name = os.path.splitext(os.path.split(filepath)[1])[0]

        self.locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
        self.formatter = mdates.ConciseDateFormatter(self.locator)

    def plot(
        self,
        y: Union[str, Sequence[str]],
        save: str = "png",
    ) -> None:
        if isinstance(y, str):
            y = [y]

        for y_ in y:
            f = plt.figure()
            ax = plt.gca()
            ax.xaxis.set_major_locator(self.locator)
            ax.xaxis.set_major_formatter(self.formatter)
            name = os.path.join(self.path, self.name + f"_{y_}.{save}")

            for pair in PAIRS[y_]:
                plt.plot(
                    self.df.index,
                    self.df[pair],
                    label=LABELS[pair]["legend"],
                )
                plt.ylabel(LABELS[pair]["ylabel"], usetex=True)
            # plt.legend(frameon=False)
            plt.legend()
            if save:
                f.savefig(name, bbox_inches="tight")
            plt.close()

    def calc_efficiency(self) -> float:
        p_max = self.df["PV Maximum Power"]
        p = self.df["PV Power"]

        return efficiency(p_max, p)

    @staticmethod
    def read_csv(path: str) -> pd.DataFrame:
        "Read a csv file and returns a DataFrame object"
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d %H:%M:%S")
        df = df.set_index("Date")
        return df


def plot_folder(y: Sequence[str] = ["p", "dc"], index: int = -1) -> str:
    "Plot the results of the folder (zero-based index)"
    if isinstance(y, str):
        y = [y]

    basepath = os.path.join("data", "dataframes")
    path = os.path.join("data", "dataframes", os.listdir(basepath)[index])
    files = [f for f in os.listdir(path) if f.endswith(".csv")]

    eff_str = ""
    for f in tqdm(files, desc=f"Plotting {y[0]}"):
        ph = PlotHandler(os.path.join(path, f))
        eff = ph.calc_efficiency()
        eff_str += f"{f}_eff_{eff}\n"
        ph.plot(y=y)

    with open(os.path.join(path, "eff_results.txt"), "w") as f:
        f.write(eff_str)

    return path


def find_test_eff(path: str, agent_name: str = "agent-test") -> float:
    eff_path = os.path.join(path, "eff_results.txt")

    eff = 0
    days = 0
    with open(eff_path) as f:
        for line in f.read().split():
            if agent_name in line:
                eff += round(float(line.split("_")[-1]), 2)
                days += 1

    assert days != 0
    return eff / days


def find_val_eff(path: str, agent_name: str = "agent-val") -> float:
    eff_path = os.path.join(path, "eff_results.txt")

    with open(eff_path) as f:
        for line in f.read().split():
            if agent_name in line:
                return round(float(line.split("_")[-1]), 2)


    return None


def sort_eff(
    path: str = "results.txt", save_path: str = "results_sorted.txt", average: int = 10
) -> None:

    with open(path) as f:
        lines = f.read().split()

    sorted_lines = sorted(lines, key=lambda s: float(s.split("_")[-1]), reverse=True)

    with open(save_path, "w") as f:
        f.write("\n".join(sorted_lines))

    # Write average of first 10
    dir_ = sorted_lines[0].split("_")[0]
    conf_path = os.path.join(dir_, "exp_conf.json")
    dic = load_dict(conf_path)
    dic_num = {}
    for k, v in dic.items():
        try:
            v = float(v)
            dic_num[k] = 0
        except:
            pass
    for line in sorted_lines[:average]:
        dir_ = line.split("_")[0]
        conf_path = os.path.join(dir_, "exp_conf.json")
        dic = load_dict(conf_path)
        for k in dic_num.keys():
            dic_num[k] += float(dic[k])
    for k in dic_num.keys():
        dic_num[k] /= average
    with open(save_path, "a") as f:
        f.write("\n")
        for key, val in dic_num.items():
            f.write(f"\n{key}:{val}")

    # Write every result
    for line in sorted_lines:
        dir_ = line.split("_")[0]
        conf_path = os.path.join(dir_, "exp_conf.json")
        dic = load_dict(conf_path)

        with open(save_path, "a") as f:
            f.write("\n")
            for key, val in dic.items():
                f.write(f"\n{key}:{val}")


if __name__ == "__main__":
    # path = plot_folder(index=-2)
    # find_test_eff(path)
    sort_eff()
