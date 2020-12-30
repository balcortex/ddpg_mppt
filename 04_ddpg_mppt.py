import os

import gym
import matlab.engine

from src.ddpg import DDPGAgent, create_ddpg_actor_critic
from src.experience import ExperienceSourceDiscountedSteps
from src.noise import GaussianNoise, OUNoise
from src.policies import DDPGPolicy, PerturbObservePolicy
from src.pv_env import PVEnv
from src.pv_array import PVArray
from src.schedule import LinearSchedule, ConstantSchedule
from src.reward import RewardPowerDeltaPower
from src.logger import logger
from src.utils import read_weather_csv

PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
PV_CKP_PATH = os.path.join("data", "01_pvarray_iv.json")
WEATHER_TRAIN_PATH = os.path.join("data", "weather_real_train.csv")
WEATHER_TEST_PATH = os.path.join("data", "weather_real_test.csv")
WEATHER_SIM_PATH = os.path.join("data", "weather_sim.csv")
STATES = ["v_norm", "p_norm", "dv_norm"]

GAMMA = 0.5
N_STEPS = 1
BATCH_SIZE = 256
BUFFER_SIZE = 5_000
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
CRITIC_L2 = 1e-2
ACTOR_L2 = 1e-3
TAU = 1e-2
DECREASE_NOISE = True
NOISE_STEPS = 8_000
NORM_REWARDS = True

try:
    engine.quit()
except NameError:
    pass
engine = matlab.engine.connect_matlab()

reward_fn = RewardPowerDeltaPower(norm=True)
pvarray = PVArray.from_json(
    path=PV_PARAMS_PATH,
    engine=engine,
    ckp_path=PV_CKP_PATH,
    f_precision=1,
)
weather_df = read_weather_csv(WEATHER_SIM_PATH)
weather_test_df = read_weather_csv(WEATHER_SIM_PATH)
env = PVEnv(
    pvarray=pvarray,
    weather_df=weather_df,
    states=STATES,
    reward_fn=reward_fn,
    max_steps=500,
)
test_env = PVEnv(
    pvarray=pvarray,
    weather_df=weather_test_df,
    states=STATES,
    reward_fn=reward_fn,
    v0=22,
)
actor, critic = create_ddpg_actor_critic(env=env)
noise = GaussianNoise(mean=0.0, std=5.0)
# noise = OUNoise(mean=0.0, std=2, theta=0.0, dt=1)
# schedule = LinearSchedule(max_steps=10000, eps_final=0.1)
schedule = ConstantSchedule(1.0)
collect_policy = DDPGPolicy(
    env=env,
    net=actor,
    noise=noise,
    schedule=schedule,
    # decrease_noise=True,
)
test_policy = DDPGPolicy(env=test_env, net=actor, noise=None, schedule=None)
po_policy = PerturbObservePolicy(
    env=test_env, v_step=0.2, dv_index="dv", dp_index="dp", noise=None
)
train_exp_source = ExperienceSourceDiscountedSteps(
    policy=collect_policy, gamma=GAMMA, n_steps=N_STEPS, steps=1
)
test_exp_source = ExperienceSourceDiscountedSteps(
    policy=test_policy, gamma=GAMMA, n_steps=N_STEPS, steps=1
)
po_exp_source = ExperienceSourceDiscountedSteps(
    policy=po_policy, gamma=GAMMA, n_steps=N_STEPS, steps=1
)
agent = DDPGAgent(
    train_exp_source=train_exp_source,
    test_exp_sorce=test_exp_source,
    actor=actor,
    critic=critic,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    actor_l2=ACTOR_L2,
    critic_l2=CRITIC_L2,
    tau=TAU,
    norm_rewards=NORM_REWARDS,
)
for _ in range(10):
    # train_exp_source.play_episode()
    # env.render_vs_true()
    agent.learn(epochs=1000, train_steps=1, log_every=1000)
    test_exp_source.play_episode()
    test_env.render_vs_true()
po_exp_source.play_episode()
test_env.render_vs_true(label="PO")
