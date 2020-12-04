import matlab.engine
import numpy as np

from src.experience import ExperienceSourceDiscountedSteps
from src.noise import GaussianNoise
from src.policies import PerturbObservePolicy
from src.pv_env import PVEnv
from src.reward import RewardPowerDeltaPower
from src.schedule import LinearSchedule

PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
WEATHER_TRAIN_PATH = os.path.join("data", "weather_sim.csv")
WEATHER_TEST_PATH = os.path.join("data", "weather_real.csv")
PVARRAY_CKP_PATH = os.path.join("data", "01_pvarray_iv.json")
AGENT_CKP_PATH = os.path.join("models", "08_mppt.tar")

GAMMA = 0.8
N_STEPS = 1
BATCH_SIZE = 64
STATES = ["v_norm", "p_norm", "dv_norm", "dp_norm"]

engine = matlab.engine.connect_matlab()
reward_fn = RewardPowerDeltaPower(norm=True)
noise = GaussianNoise(0.0, 0.1)
env = PVEnv.from_file(
    pv_params_path=PV_PARAMS_PATH,
    weather_path=WEATHER_TRAIN_PATH,
    pvarray_ckp_path=PVARRAY_CKP_PATH,
    engine=engine,
    states=STATES,
    reward_fn=reward_fn,
    v0=12,
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
    epsilon=1.0,
)
po_source = ExperienceSourceDiscountedSteps(
    policy=po_policy,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
)
po_source.play_episode()
env.render_vs_true(label="PO + Noise")
