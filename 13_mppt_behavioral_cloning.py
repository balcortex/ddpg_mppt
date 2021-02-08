import os

import matlab.engine

from src import rl
from src.rl import BCAgent, init_weights_xavier_uniform
from src.experience import ExperienceSourceDiscountedSteps
from src.policies import DDPGPolicy, PerturbObservePolicyDCDC
from src.pv_array_dcdc import PVArray
from src.pv_env_dcdc import PVEnv
from src.reward import RewardPowerDeltaPower
from src import utils

PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
WEATHER_TRAIN_PATH = os.path.join("data", "weather_real_train.csv")
WEATHER_VAL_PATH = os.path.join("data", "weather_real_val.csv")
WEATHER_TEST_PATH = os.path.join("data", "weather_real_test.csv")
WEATHER_SIM_PATH = os.path.join("data", "weather_sim.csv")
STATES = ["v_norm", "p_norm", "dv_norm"]
DUTY_CYCLE_INITIAL = 0.0
DC_STEP = 0.02

GAMMA = 0.5
N_STEPS = 1
DEMO_BATCH_SIZE = 512
DEMO_BUFFER_SIZE = 5000
EPOCHS = 100_000
ACTOR_LR = 1e-3
ACTOR_L2 = 1e-4

try:
    engine.quit()  # type: ignore
except NameError:
    pass
engine = matlab.engine.connect_matlab()

path = utils.make_datetime_folder(os.path.join("data", "dataframes"))

reward_fn = RewardPowerDeltaPower(norm=True)
pvarray = PVArray.from_json(
    path=PV_PARAMS_PATH,
    engine=engine,
    model_name="pv_boost_avg_rload",
)
weather_df = utils.read_weather_csv(WEATHER_TRAIN_PATH, format=None)
weather_val_df = utils.read_weather_csv(WEATHER_VAL_PATH, format=None)
weather_test_df = utils.read_weather_csv(WEATHER_TEST_PATH, format=None)
# weather_sim_df = utils.read_weather_csv(WEATHER_SIM_PATH)
demo_env = PVEnv(
    pvarray=pvarray,
    weather_df=weather_df,
    # weather_df=weather_sim_df,
    states=STATES,
    reward_fn=reward_fn,
    dc0=DUTY_CYCLE_INITIAL,
)
val_env = PVEnv(
    pvarray=pvarray,
    weather_df=weather_val_df,
    # weather_df=weather_sim_df,
    states=STATES,
    reward_fn=reward_fn,
    dc0=DUTY_CYCLE_INITIAL,
)
test_env = PVEnv(
    pvarray=pvarray,
    weather_df=weather_test_df,
    # weather_df=weather_sim_df,
    states=STATES,
    reward_fn=reward_fn,
    dc0=DUTY_CYCLE_INITIAL,
)

actor, _ = rl.create_ddpg_actor_critic(demo_env, weight_init_fn=None)

val_policy = DDPGPolicy(env=val_env, net=actor, noise=None, schedule=None)
test_policy = DDPGPolicy(env=test_env, net=actor, noise=None, schedule=None)

po_demo_policy = PerturbObservePolicyDCDC(
    env=demo_env, v_step=DC_STEP, dv_index="dv", dp_index="dp"
)
po_val_policy = PerturbObservePolicyDCDC(
    env=val_env, v_step=DC_STEP, dv_index="dv", dp_index="dp"
)
po_test_policy = PerturbObservePolicyDCDC(
    env=test_env, v_step=DC_STEP, dv_index="dv", dp_index="dp"
)

demo_val_source = ExperienceSourceDiscountedSteps(
    policy=po_val_policy,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
    pvenv_kwargs={
        "save_dataframe": True,
        "include_true_mpp": True,
        "policy_name": "po-val",
        "basepath": path,
    },
)
agent_val_source = ExperienceSourceDiscountedSteps(
    policy=val_policy,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
    pvenv_kwargs={
        "save_dataframe": True,
        "include_true_mpp": True,
        "policy_name": "ddpg-val",
        "basepath": path,
    },
)
demo_test_source = ExperienceSourceDiscountedSteps(
    policy=po_test_policy,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
    pvenv_kwargs={
        "save_dataframe": True,
        "include_true_mpp": True,
        "policy_name": "po-test",
        "basepath": path,
    },
)
agent_test_source = ExperienceSourceDiscountedSteps(
    policy=test_policy,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
    pvenv_kwargs={
        "save_dataframe": True,
        "include_true_mpp": True,
        "policy_name": "ddpg_test",
        "basepath": path,
    },
)
demo_train_source = ExperienceSourceDiscountedSteps(
    policy=po_demo_policy,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
    pvenv_kwargs={
        "save_dataframe": True,
        "include_true_mpp": True,
        "policy_name": "po-demo",
        "basepath": path,
    },
)

bc = BCAgent(
    demo_train_source=demo_train_source,
    demo_val_source=demo_val_source,
    demo_test_source=demo_test_source,
    agent_val_source=agent_val_source,
    agent_test_source=agent_test_source,
    actor=actor,
    demo_buffer_size=DEMO_BUFFER_SIZE,
    demo_batch_size=DEMO_BATCH_SIZE,
    actor_lr=ACTOR_LR,
    actor_l2=ACTOR_L2,
)

bc.learn(epochs=EPOCHS, val_every=500)
agent_test_source.play_episode()
# bc.run_test()
test_env.render_vs_true()