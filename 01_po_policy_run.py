import matlab.engine
import numpy as np
import os

from src.experience import ExperienceSourceDiscountedSteps
from src.noise import GaussianNoise
from src.policies import PerturbObservePolicy
from src.pv_env import PVEnv
from src.reward import RewardPowerDeltaPower
from src.schedule import ConstantSchedule
from src.pv_array import PVArray
from src.utils import read_weather_csv

PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
WEATHER_REAL_TRAIN_PATH = os.path.join("data", "weather_real_train.csv")
WEATHER_REAL_TEST_PATH = os.path.join("data", "weather_real_test.csv")
PV_CKP_PATH = os.path.join("data", "01_pvarray_iv.json")
AGENT_CKP_PATH = os.path.join("models", "08_mppt.tar")

GAMMA = 0.8
N_STEPS = 1
BATCH_SIZE = 64
STATES = ["v_norm", "p_norm", "dv_norm", "dp_norm"]

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
    ckp_path=PV_CKP_PATH,
    f_precision=1,
)
weather_train_df = read_weather_csv(WEATHER_REAL_TRAIN_PATH)
weather_test_df = read_weather_csv(WEATHER_REAL_TEST_PATH)
env = PVEnv(
    pvarray=pvarray,
    weather_df=weather_train_df,
    states=STATES,
    reward_fn=reward_fn,
    max_steps=500,
)
test_env = PVEnv(
    pvarray=pvarray,
    weather_df=weather_test_df,
    states=STATES,
    reward_fn=reward_fn,
)

po_policy = PerturbObservePolicy(
    env=env,
    v_step=0.1,
    dv_index="dv",
    dp_index="dp",
    noise=None,
)
po_source = ExperienceSourceDiscountedSteps(
    policy=po_policy,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
)
po_source.play_episode()
env.render_vs_true(label="PO")

po_policy = PerturbObservePolicy(
    env=env,
    v_step=0.1,
    dv_index="dv",
    dp_index="dp",
    noise=noise,
    schedule=schedule,
)
po_source = ExperienceSourceDiscountedSteps(
    policy=po_policy,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
)
po_source.play_episode()
env.render_vs_true(label="PO + Noise")

po_policy_test = PerturbObservePolicy(
    env=test_env,
    v_step=0.1,
    dv_index="dv",
    dp_index="dp",
    noise=None,
)
po_source_test = ExperienceSourceDiscountedSteps(
    policy=po_policy_test,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
)
po_source_test.play_episode()
test_env.render_vs_true(label="PO")

po_policy_test = PerturbObservePolicy(
    env=test_env,
    v_step=0.1,
    dv_index="dv",
    dp_index="dp",
    noise=noise,
    schedule=schedule,
)
po_source_test = ExperienceSourceDiscountedSteps(
    policy=po_policy_test,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
)
po_source_test.play_episode()
test_env.render_vs_true(label="PO + Noise")
