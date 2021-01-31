import os
import datetime
import time

import matlab.engine

from src.behavioral_cloning import BehavioralCloning
from src.experience import ExperienceSourceDiscountedSteps
from src.policies import DDPGPolicy, PerturbObservePolicyDCDC
from src.pv_array_dcdc import PVArray
from src.pv_env_dcdc import PVEnv
from src.reward import RewardPowerDeltaPower
from src.utils import mse, read_weather_csv, save_dict
from src.ddpg import create_ddpg_actor_critic, DDPGAgent
from src.noise import GaussianNoise
from src.schedule import LinearSchedule
from src.utils import save_dict
from src.plot import plot_folder, find_test_eff
from src.utils import grid_generator


PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
PV_CKP_PATH = os.path.join("data", "02_pvarray_dcdc.json")
WEATHER_TRAIN_PATH = os.path.join("data", "weather_real_train.csv")
WEATHER_VAL_PATH = os.path.join("data", "weather_real_val.csv")
WEATHER_TEST_PATH = os.path.join("data", "weather_real_test.csv")
WEATHER_SIM_PATH = os.path.join("data", "weather_sim.csv")
STATES = ["v_norm", "p_norm", "dv_norm"]
DUTY_CYCLE_INITIAL = 0.0
DC_STEP = 0.02
# GAMMA = 0.99
# N_STEPS = 1
# BATCH_SIZE = 128
# BUFFER_SIZE = 50_000
# ACTOR_LR = 1e-4
# ACTOR_L2 = 1e-3
# CRITIC_LR = 1e-3
# CRITIC_L2 = 1e-5
# TAU = 1e-2
# NOISE_STEPS = 5_000
# DECREASE_NOISE = True
# EPS_FINAL = 0.01
# NORM_REWARDS = False
MODEL_NAME = "pv_boost_avg_rload"

try:
    ENGINE.quit()  # type: ignore
except NameError:
    pass
ENGINE = matlab.engine.connect_matlab()


class DDPGMPPT:
    def __init__(
        self,
        gamma,
        n_steps,
        batch_size,
        buffer_size,
        actor_lr,
        actor_l2,
        critic_lr,
        critic_l2,
        tau,
        noise_steps,
        decrease_noise,
        eps_final,
        norm_rewards,
        epochs,
    ):
        self.exp_conf = locals()
        self.epochs = epochs

        self.basepath = os.path.join(
            "data", "dataframes", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        os.makedirs(self.basepath)

        time.sleep(3)

        reward_fn = RewardPowerDeltaPower(norm=True)
        pvarray = PVArray.from_json(
            path=PV_PARAMS_PATH,
            engine=ENGINE,
            model_name=MODEL_NAME,
        )
        weather_df = read_weather_csv(WEATHER_TRAIN_PATH, format=None)
        weather_test_df = read_weather_csv(WEATHER_TEST_PATH, format=None)
        weather_sim_df = read_weather_csv(WEATHER_SIM_PATH)
        env = PVEnv(
            pvarray=pvarray,
            weather_df=weather_df,
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
        actor, critic = create_ddpg_actor_critic(env=env)
        noise = GaussianNoise(mean=0.0, std=0.4)
        # noise = OUNoise(mean=0.0, std=2, theta=0.0, dt=1)
        schedule = LinearSchedule(
            max_steps=noise_steps,
            eps_final=eps_final,
        )
        # schedule = ConstantSchedule(1.0)
        collect_policy = DDPGPolicy(
            env=env,
            net=actor,
            noise=noise,
            schedule=schedule,
            decrease_noise=decrease_noise,
        )
        test_policy = DDPGPolicy(env=test_env, net=actor, noise=None, schedule=None)
        po_policy = PerturbObservePolicyDCDC(
            env=test_env, v_step=DC_STEP, dv_index="dv", dp_index="dp"
        )
        train_exp_source = ExperienceSourceDiscountedSteps(
            policy=collect_policy,
            gamma=gamma,
            n_steps=n_steps,
            steps=1,
            pvenv_kwargs={
                "save_dataframe": True,
                "include_true_mpp": True,
                "policy_name": "ddpg-train",
                "basepath": self.basepath,
            },
        )
        test_exp_source = ExperienceSourceDiscountedSteps(
            policy=test_policy,
            gamma=gamma,
            n_steps=n_steps,
            steps=1,
            pvenv_kwargs={
                "save_dataframe": True,
                "include_true_mpp": True,
                "policy_name": "ddpg-test",
                "basepath": self.basepath,
            },
        )
        self.po_exp_source = ExperienceSourceDiscountedSteps(
            policy=po_policy,
            gamma=gamma,
            n_steps=n_steps,
            steps=1,
            pvenv_kwargs={
                "save_dataframe": True,
                "include_true_mpp": True,
                "policy_name": "po-test",
                "basepath": self.basepath,
            },
        )

        self.agent = DDPGAgent(
            train_exp_source=train_exp_source,
            test_exp_sorce=test_exp_source,
            actor=actor,
            critic=critic,
            buffer_size=buffer_size,
            batch_size=batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_l2=actor_l2,
            critic_l2=critic_l2,
            tau=tau,
            norm_rewards=norm_rewards,
        )

    def learn(self) -> None:
        self.agent.learn(epochs=self.epochs, train_steps=1, collect_steps=1)

    def make_results(self) -> None:
        self.po_exp_source.play_episode()
        self.agent.test_exp_sorce.play_episode()

        path = plot_folder(y=["p"])
        path = plot_folder(y=["dc"])
        test_eff = find_test_eff(path)

        self.exp_conf.update(
            {
                "test_eff": test_eff,
                "states": STATES,
                "duty_cycle_inital": DUTY_CYCLE_INITIAL,
                "duty_cycle_step": DC_STEP,
                "schedule": str(self.agent.train_exp_source.policy.schedule),
                "noise": str(self.agent.train_exp_source.policy.noise),
                "reward": str(self.agent.train_exp_source.policy.env.reward_fn),
                "model_name": MODEL_NAME,
            }
        )
        if "self" in self.exp_conf.keys():
            self.exp_conf.pop("self")

        save_dict(self.exp_conf, os.path.join(self.basepath, "exp_conf.json"))

        result = f"{self.basepath}_test_eff_{test_eff}\n"

        with open("results.txt", "a") as f:
            f.write(result)


hparams = {
    "gamma": [0.1],
    "n_steps": [1],
    "batch_size": [64, 128],
    "buffer_size": [50_000],
    "actor_lr": [1e-3],
    "actor_l2": [1e-3],
    "critic_lr": [1e-3],
    "critic_l2": [1e-5],
    "tau": [1e-3],
    "noise_steps": [5000],
    "decrease_noise": [True],
    "eps_final": [0.0],
    "norm_rewards": [False],
    "epochs": [20_000],
    # "epochs": [20_000],
}

gg = grid_generator(hparams)

for dic_hparams in gg:
    for _ in range(5):
        mppt = DDPGMPPT(**dic_hparams)
        mppt.learn()
        mppt.make_results()
