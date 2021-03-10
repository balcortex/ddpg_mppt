import os
from pathlib import Path

import matlab.engine

from src.pv_array_dcdc import PVArray
from src.pv_env_dcdc import PVEnv
from src.reward import RewardPowerDeltaPower
from src.policies import MPPPolicyDCDC
from src import utils, mppt_utils
from src.experience import ExperienceSourceDiscountedSteps


PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
WEATHER_TRAIN_PATH = os.path.join("data", "weather_real_train.csv")
WEATHER_VAL_PATH = os.path.join("data", "weather_real_val.csv")
WEATHER_TEST_PATH = os.path.join("data", "weather_real_test.csv")
WEATHER_SIM_PATH = os.path.join("data", "weather_sim.csv")
STATES = ["v_norm", "p_norm", "dv_norm"]
MODEL_NAME = "pv_boost_avg_rload"
DUTY_CYCLE_INITIAL = 0.0


try:
    ENGINE.quit()  # type: ignore
except NameError:
    pass
ENGINE = matlab.engine.connect_matlab()


path = utils.make_datetime_folder(os.path.join("data", "dataframes"))
reward_fn = RewardPowerDeltaPower(norm=True)
pvarray = PVArray.from_json(
    path=PV_PARAMS_PATH,
    engine=ENGINE,
    model_name=MODEL_NAME,
)
weather_df = utils.read_weather_csv(WEATHER_TRAIN_PATH, format=None)

env = PVEnv(
    pvarray=pvarray,
    weather_df=weather_df,
    states=STATES,
    reward_fn=reward_fn,
    dc0=DUTY_CYCLE_INITIAL,
)

policy = MPPPolicyDCDC(
    env=env,
    g_index="n_g",
    t_index="n_amb_t",
    dc_index="duty_cycle",
)
exp_source = ExperienceSourceDiscountedSteps(
    policy=policy,
    gamma=0.99,
    n_steps=1,
    steps=1,
    pvenv_kwargs={
        "save_dataframe": True,
        "include_true_mpp": True,
        "policy_name": "mpp_policy",
        "basepath": path,
    },
)

exp_source.play_episode()
basepath = list(Path(r"C:\git_code\ddpg_mppt\data\dataframes").iterdir())[-1]
path = list(basepath.iterdir())[-1]

df = utils.read_weather_csv(
    path,
    format=None,
)

mppt_utils.dataframe_efficiency(df)
f = mppt_utils.plot_pair_column(df)
f1 = mppt_utils.plot_pair_column(df, y="d")

pvarray.save()