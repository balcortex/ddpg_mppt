import matlab.engine
import numpy as np
import os

from src.experience import ExperienceSourceDiscountedSteps
from src.noise import GaussianNoise
from src.policies import RandomPolicy
from src.pv_env_dcdc import PVEnv
from src.reward import RewardPowerDeltaPower
from src.schedule import ConstantSchedule
from src.pv_array_dcdc import PVArray
from src.utils import read_weather_csv

PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
WEATHER_REAL_TRAIN_PATH = os.path.join("data", "weather_real_train.csv")
WEATHER_REAL_TEST_PATH = os.path.join("data", "weather_real_test.csv")
WEATHER_SIM_PATH = os.path.join("data", "weather_sim.csv")
PV_CKP_PATH = os.path.join("data", "02_pvarray_dcdc.json")
AGENT_CKP_PATH = None

GAMMA = 0.8
N_STEPS = 1
BATCH_SIZE = 64
STATES = ["v", "p", "dv", "dp", "duty_cycle"]

try:
    engine.quit()
except NameError:
    pass
engine = matlab.engine.connect_matlab()

reward_fn = RewardPowerDeltaPower(norm=True)
noise = GaussianNoise(0.0, 0.1)
schedule = ConstantSchedule(1.0)
pvarray = PVArray.from_json(
    path=PV_PARAMS_PATH,
    engine=engine,
)
weather_train_df = read_weather_csv(WEATHER_REAL_TRAIN_PATH)
weather_test_df = read_weather_csv(WEATHER_REAL_TEST_PATH)
weather_sim_df = read_weather_csv(WEATHER_SIM_PATH)
env = PVEnv(
    pvarray=pvarray,
    weather_df=weather_test_df,
    states=STATES,
    reward_fn=reward_fn,
    dc0=0.0,
    day_index=0,
)

random_policy = RandomPolicy(env)
po_source = ExperienceSourceDiscountedSteps(
    policy=random_policy,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
    pvenv_kwargs={
        "save_dataframe": True,
        "include_true_mpp": True,
        "policy_name": "PO",
    },
)

po_source.play_episode()
env.render_vs_true(label="Random")
# env.render(["p", "v", "duty_cycle"])
