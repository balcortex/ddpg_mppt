import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
import random
import os
from src.utils import read_weather_csv
import time


def clean_weather_file(filename: str):
    plt.ion()
    plt.show()

    df = read_weather_csv(f"{filename}.csv")

    # Cleaning
    df = df[df["Irradiance"] > 0]
    df["Temperature"] = df["Temperature"].round(2)

    df_list = [group[1] for group in df.groupby(df.index.date)]

    clean_day_index = []
    for df_day in df_list:
        if len(df_day) > 700:
            # df_day.plot(subplots=True, layout=(1, 2))
            plt.plot(df_day["Irradiance"])
            # plt.show()
            plt.draw()
            plt.pause(0.01)
            # time.sleep(3)
            # plt.close()
            clean_day_index.append(input() or "0")
            plt.clf()
        else:
            clean_day_index.append("0")

    # with open(f"clean_indexes_{filename}.txt", "w") as f:
    #     f.write(",".join(clean_day_index))

    clean_day_index_ = list(map(bool, list(map(int, clean_day_index))))
    df_list_clean = [
        df_ for (df_, idx) in zip(df_list, clean_day_index_) if idx == True
    ]

    df_clean = pd.concat(df_list_clean)
    df_clean.to_csv(f"{filename}_clean.csv")


def split_train_val_test(
    val_pct: float = 0.1,
    test_pct: float = 0.1,
    filenames: Sequence[str] = ("weather2017_clean", "weather2018_clean"),
    shuffle: bool = True,
    seed: int = 44,
):
    df = pd.concat([read_weather_csv(f"{f}.csv", format=None) for f in filenames])
    df_day = [group[1] for group in df.groupby(df.index.date)]

    if shuffle:
        random.seed(seed)
        random.shuffle(df_day)

    split_idx = int((1 - val_pct - test_pct) * len(df_day))
    df_train = pd.concat(df_day[:split_idx])

    df_ = df_day[split_idx:]
    split_idx = int((1 - test_pct / (val_pct + test_pct)) * len(df_))
    df_val = pd.concat(df_[:split_idx])
    df_test = pd.concat(df_[split_idx:])

    df_train.to_csv(os.path.join("data", "weather_real_train.csv"))
    df_val.to_csv(os.path.join("data", "weather_real_val.csv"))
    df_test.to_csv(os.path.join("data", "weather_real_test.csv"))

    print(
        f"Days in training: {len([g[1] for g in df_train.groupby(df_train.index.date)])}"
    )
    print(
        f"Days in validation: {len([g[1] for g in df_val.groupby(df_val.index.date)])}"
    )
    print(f"Days in test: {len([g[1] for g in df_test.groupby(df_test.index.date)])}")


if __name__ == "__main__":
    # clean_weather_file("weather2017")
    # clean_weather_file("weather2018")
    split_train_val_test(val_pct=0.08, test_pct=0.05)
    # clean_weather_file("weather2019")