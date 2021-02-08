import datetime
import itertools
import json
import os
from typing import Any, Dict, Generator, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.logger import logger


def read_weather_csv(
    path: str, format: Optional[str] = "%d/%m/%Y %H:%M"
) -> pd.DataFrame:
    "Read a csv weather file and returns a DataFrame object"
    logger.info(f"Reading {path} . . .")
    df = pd.read_csv(path)
    if format:
        df["Date"] = pd.to_datetime(df["Date"], format=format)
    else:
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


def save_dict(dic: Dict, path: str, verbose: bool = True) -> None:
    "Save a dictionary as a JSON file"
    with open(path, "w") as f:
        f.write(json.dumps(dic))
    if verbose:
        logger.info(f"Dictionary saved to {path}")


def load_dict(path: str) -> Dict:
    "Read a JSON file as a dictionary"
    with open(path) as f:
        dic = json.loads(f.read())
    logger.info(f"Dictionary readed from {path}")
    return dic


def mse(a: Sequence[Union[int, float]], b: Sequence[Union[int, float]]) -> float:
    "Calculate the mean square error of `a` and `b`"
    a = np.array(a)
    b = np.array(b)
    return ((a - b) ** 2).mean()


def efficiency(a: Sequence[Union[int, float]], b: Sequence[Union[int, float]]) -> float:
    "Calculate the efficiency of `b` with respect to `a`"
    a = np.array(a)
    b = np.array(b)
    return (b / a).mean() * 100


def grid_generator(
    dic: Dict[Any, Sequence[Any]]
) -> Generator[Dict[Any, Any], None, None]:
    "Perform permutation on the values (sequence) of a dictionary"
    keys, values = zip(*dic.items())
    return (dict(zip(keys, v)) for v in itertools.product(*values))


def make_datetime_folder(basepath: str) -> str:
    path = os.path.join(basepath, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(path, exist_ok=True)

    return path
