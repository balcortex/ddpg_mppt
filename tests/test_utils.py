import os
from src.utils import read_weather_csv, load_dict, save_dict


def test_read_weather():
    df = read_weather_csv(os.path.join(".", "data", "weather_sim.csv"))[79:]
    ir = df.loc["2019-06-29 07:20:00"]["Irradiance"]
    temp = df.iloc[0]["Temperature"]

    assert ir == 200
    assert temp == 20


# def test_clip_num():
#     assert clip_num(0) == 0
#     assert clip_num(-1.0) == -1
#     assert clip_num(1) == 1
#     assert clip_num(-3, minimum=0) == 0
#     assert clip_num(12, maximum=10) == 10
#     assert clip_num(8, 0, 10) == 8


def test_load_save():
    dic = {
        "a": 1,
        "b": "2",
        "c": [1, 2, 3],
        "d": [1.0, 2.0, 3.0],
        "e": {"1": "Uno", "2": "Dos"},
    }
    save_dict(dic, "temp.txt")
    loaded_dic = load_dict("temp.txt")
    assert dic == loaded_dic
    os.remove("temp.txt")
