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

# def dataframe_mean(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
#     df_cat = pd.concat((dfs))
#     df_mean = df_cat.groupby([df_cat.index.date, df_cat.index.time]).mean()
#     df_mean = df_mean.reset_index()
#     df_mean["date"] = (
#         df_mean["level_0"].astype(str) + " " + df_mean["level_1"].astype(str)
#     )
#     df_mean["date"] = pd.to_datetime(df_mean["date"])
#     df_mean = df_mean.drop(columns=["level_0", "level_1"])
#     df_mean = df_mean.set_index("date")

#     return df_mean


# def get_dataframes(
#     basepath: str,
#     name_pattern: str,
#     slice: Sequence[Optional[int]] = (None, None),
# ):
#     assert len(slice) == 2

#     basepath = Path(basepath)
#     dirs = islice(basepath.iterdir(), slice[0], slice[1])

#     paths: Sequence[Path] = [dir_ for dir_ in dirs]
#     all_csv = [list(path.glob(name_pattern)) for path in paths]

#     dfs = []
#     for csv_paths in all_csv:
#         df = pd.concat(read_weather_csv(path, format=None) for path in csv_paths)
#         dfs.append(df)

#     df_mean = dataframe_mean(dfs)

#     # dfs_day = []
#     # for df in dfs:
#     #     dfs_day.append(df.groupby(df.index.date))

#     dfs_day = [g[1] for df in dfs for g in df.groupby(df.index.date)]
#     df_m = [g[1] for g in df_mean.groupby(df_mean.index.date)]

#     return (dfs_day, df_m)


# def get_dataframes(
#     basepath: str,
#     name_pattern: str,
#     slices: Sequence[Optional[int]] = (None, None),
# ):
#     assert len(slices) == 2

#     basepath = Path(basepath)
#     dirs = islice(basepath.iterdir(), slices[0], slices[1])

#     paths: Sequence[Path] = [dir_ for dir_ in dirs]
#     csvs_by_day = [list(path.glob(name_pattern)) for path in paths]

#     df_by_day = {
#         f"day_{index}": [read_weather_csv(f, format=None) for f in day]
#         for (index, day) in enumerate(csvs_by_day, start=1)
#     }

#     return df_by_day


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


if __name__ == "__main__":
    basepath = r"C:\Users\Balco\Downloads\mppt_results\bc\real"
    # name_pattern = "agent test"
    name_pattern = "po expert"
    # dir_index = [1, 2, 3, 4, 5]
    # dir_index = range(16, 21)
    dir_index = [16, 11, 6, 1]
    # dir_index = "20210213-142807"
    day_index = [1]

    res = get_dataframes(basepath, name_pattern, dir_index, day_index)
    res = flat_list(res)

    res1 = combine_dataframes(
        dfs=res,
        colum="PV Power",
        column_names=["Step=0.01", "Step=0.02", "Step=0.03", "Step=0.04"],
        shared_column="PV Maximum Power",
        shared_col_name="$P_{max}$",
        reset_index=True,
    )

    f = mppt_utils.plot_dataframe(res1, "p", zorder=[5, 1, 2, 3, 4], x_axis_date=False)

    mppt_utils.dataframe_efficiency(res1)
