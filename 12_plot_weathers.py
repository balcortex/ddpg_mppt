from src.utils import read_weather_csv
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot(df: pd.DataFrame) -> None:
    df_days = [g[1] for g in df.groupby(df.index.date)]
    for day in df_days:
        day.plot(subplots=True, layout=(2, 1))
        plt.show()


if __name__ == "__main__":
    # print("Training")
    # df_train = read_weather_csv(
    #     os.path.join("data", "weather_real_train.csv"),
    #     format=None,
    # )
    # plot(df_train)
    print("Validation")
    df_val = read_weather_csv(
        os.path.join("data", "weather_real_val.csv"),
        format=None,
    )
    plot(df_val)
    print("Test")
    df_test = read_weather_csv(
        os.path.join("data", "weather_real_test.csv"),
        format=None,
    )
    plot(df_test)
