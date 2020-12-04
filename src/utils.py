import json
from typing import Dict

import numpy as np
import pandas as pd

from src.logger import logger


def read_weather_csv(path: str) -> pd.DataFrame:
    "Read a csv file and returns a DataFrame object"
    logger.info(f"Reading {path} . . .")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
    df = df.set_index("Date")
    return df


# def clip_num(value: float, minimum: float = -np.inf, maximum: float = np.inf) -> float:
#     "Clip the value between minimum and maximum parameters"
#     return min(max(value, minimum), maximum)


def save_dict(dic: Dict, path: str, verbose: bool = True) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(dic))
    if verbose:
        logger.info(f"Dictionary saved to {path}")


def load_dict(path: str) -> Dict:
    with open(path) as f:
        dic = json.loads(f.read())
    logger.info(f"Dictionary readed from {path}")
    return dic
