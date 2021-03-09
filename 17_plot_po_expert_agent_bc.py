import os
from itertools import islice
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import read_weather_csv
from src.utils import flat_list
from src import mppt_utils


def name_pattern_glob(string: str, suffix: str = ".csv") -> str:
    "Return a name pattern for matching glob"
    return "*" + ("*").join(string.split()) + "*" + suffix


def get_dataframe(
    basepath: str,
    name_search: str,
    dir_index: Union[int, str],
    day_index: int = 1,
) -> pd.DataFrame:
    "Return a dataframe from the folder"

    basepath = Path(basepath)
    name_pattern = name_pattern_glob(name_search)

    if isinstance(dir_index, int):
        assert dir_index > 0, "dir_index must be greater than 0"
        path: Path = list(islice(basepath.iterdir(), dir_index - 1, dir_index))[0]
    else:
        path: Path = Path.joinpath(basepath, dir_index)

    print(path)

    csv_files = list(path.glob(name_pattern))
    assert len(csv_files) >= day_index, "Day index is out of bounds"

    return read_weather_csv(csv_files[day_index - 1], format=None)


def get_dataframes(
    basepath: str,
    name_search: str,
    dir_index: Sequence[Union[int, str]],
    day_index: Sequence[int] = (1,),
) -> Sequence[pd.DataFrame]:

    assert isinstance(dir_index, Sequence), "dir_index must be a Sequence"
    assert isinstance(day_index, Sequence), "dir_index must be a Sequence"

    dfs = [
        [get_dataframe(basepath, name_search, dir_idx, day_idx)]
        for dir_idx in dir_index
        for day_idx in day_index
    ]

    return dfs


def combine_dataframes(
    dfs: Sequence[pd.DataFrame],
    colum: str,
    column_names: Sequence[str],
    shared_column: Optional[str] = None,
    shared_col_name: Optional[str] = None,
    reset_index: bool = False,
) -> pd.DataFrame:
    assert len(dfs) == len(
        column_names
    ), "dataframes and column_names must be the same length"

    df = pd.DataFrame()

    if shared_column:
        if not shared_col_name:
            shared_col_name = shared_column
        df[shared_col_name] = dfs[0][shared_column]

    for label, df_ in zip(column_names, dfs):
        df[label] = df_[colum]

    if reset_index:
        df = df.reset_index()
        df = df.drop(columns=["Date"])

    return df


def plot_po_expert():
    basepath = r"C:\Users\Balco\Downloads\mppt_results\bc\real"
    name_pattern = "po expert"
    dir_index = ["20210306-114357"]
    day_index = [1]

    res = get_dataframes(basepath, name_pattern, dir_index, day_index)
    res = flat_list(res)

    res1 = combine_dataframes(
        dfs=res,
        colum="PV Power",
        column_names=["$\mathrm{P\&O}$"],
        shared_column="PV Maximum Power",
        shared_col_name="$P_{max}$",
        reset_index=False,
    )

    f = mppt_utils.plot_dataframe(res1, "p", zorder=[1, 0])
    f.savefig("po_expert_train.pdf", bbox_inches="tight")

    mppt_utils.dataframe_efficiency(res1)


def plot_bc_agent():
    basepath = r"C:\Users\Balco\Downloads\mppt_results\bc\real"
    name_pattern = "bc agent test"
    dir_index = ["20210308-094850"]
    day_index = [4]

    res = get_dataframes(basepath, name_pattern, dir_index, day_index)
    res = flat_list(res)

    res1 = combine_dataframes(
        dfs=res,
        colum="PV Power",
        column_names=["$\mathrm{BC}$ (ours)"],
        shared_column="PV Maximum Power",
        shared_col_name="$P_{max}$",
        reset_index=False,
    )

    f = mppt_utils.plot_dataframe(res1, "p", zorder=[1, 0])
    f.savefig("bc_agent_test.pdf", bbox_inches="tight")

    mppt_utils.dataframe_efficiency(res1)


def comparison_test(day: int):
    basepath = r"C:\Users\Balco\Downloads\mppt_results\bc\real"
    dir_index = ["20210308-094850"]
    day_index = [day]

    agent = get_dataframes(basepath, "bc agent test", dir_index, day_index)
    agent = flat_list(agent)

    po_expert = get_dataframes(basepath, "po expert test", dir_index, day_index)
    po_expert = flat_list(po_expert)

    res = [agent[0], po_expert[0]]
    res1 = combine_dataframes(
        dfs=res,
        colum="PV Power",
        column_names=["$\mathrm{BC}$ (ours)", "$\mathrm{P\&O}$"],
        shared_column="PV Maximum Power",
        shared_col_name="$P_{max}$",
        reset_index=False,
    )

    f = mppt_utils.plot_dataframe(res1, "p", zorder=[3, 2, 1])
    f.tight_layout()
    f.savefig(f"mppt_tracking_comparison_day{day}.pdf", bbox_inches="tight")

    mppt_utils.dataframe_efficiency(res1)


if __name__ == "__main__":
    # plot_po_expert()
    plot_bc_agent()
    comparison_test(day=1)
    comparison_test(day=2)
    comparison_test(day=3)
    comparison_test(day=4)
