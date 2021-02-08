import os

import matlab.engine

from src.behavioral_cloning import BehavioralCloning
from src.experience import ExperienceSourceDiscountedSteps
from src.policies import DDPGPolicy, PerturbObservePolicyDCDC
from src.pv_array_dcdc import PVArray
from src.pv_env_dcdc import PVEnv
from src.reward import RewardPowerDeltaPower
from src.utils import mse, read_weather_csv, save_dict, make_datetime_folder

PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
PV_CKP_PATH = os.path.join("data", "02_pvarray_dcdc.json")
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
EPOCHS = 10_000
ACTOR_LR = 1e-3
ACTOR_L2 = 1e-4

try:
    engine.quit()
except NameError:
    pass
engine = matlab.engine.connect_matlab()

path = make_datetime_folder(os.path.join("data", "dataframes"))


reward_fn = RewardPowerDeltaPower(norm=True)
pvarray = PVArray.from_json(
    path=PV_PARAMS_PATH,
    engine=engine,
    model_name="pv_boost_avg_rload",
)
# weather_df = read_weather_csv(WEATHER_TRAIN_PATH)
# weather_val_df = read_weather_csv(WEATHER_VAL_PATH)
# weather_test_df = read_weather_csv(WEATHER_TEST_PATH)
weather_sim_df = read_weather_csv(WEATHER_SIM_PATH)
demo_env = PVEnv(
    pvarray=pvarray,
    # weather_df=weather_df,
    weather_df=weather_sim_df,
    states=STATES,
    reward_fn=reward_fn,
    dc0=DUTY_CYCLE_INITIAL,
)
val_env = PVEnv(
    pvarray=pvarray,
    # weather_df=weather_val_df,
    weather_df=weather_sim_df,
    states=STATES,
    reward_fn=reward_fn,
    dc0=DUTY_CYCLE_INITIAL,
)
test_env = PVEnv(
    pvarray=pvarray,
    # weather_df=weather_test_df,
    weather_df=weather_sim_df,
    states=STATES,
    reward_fn=reward_fn,
    dc0=DUTY_CYCLE_INITIAL,
)

actor = BehavioralCloning.create_actor(env=demo_env)

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

val_po_exp_source = ExperienceSourceDiscountedSteps(
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
val_ddpg_exp_source = ExperienceSourceDiscountedSteps(
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
test_po_exp_source = ExperienceSourceDiscountedSteps(
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
test_exp_source = ExperienceSourceDiscountedSteps(
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
demo_exp_source = ExperienceSourceDiscountedSteps(
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

bc = BehavioralCloning(
    demo_exp_source=demo_exp_source,
    val_po_exp_source=val_po_exp_source,
    val_ddpg_exp_source=val_ddpg_exp_source,
    test_po_exp_source=test_po_exp_source,
    test_ddpg_exp_source=test_exp_source,
    actor=actor,
    demo_buffer_size=DEMO_BUFFER_SIZE,
    demo_batch_size=DEMO_BATCH_SIZE,
    actor_lr=ACTOR_LR,
    actor_l2=ACTOR_L2,
)

bc.learn(epochs=EPOCHS)
# bc.run_test()
test_env.render_vs_true()