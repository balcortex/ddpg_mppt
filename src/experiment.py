from __future__ import annotations
import os
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union, Generator

import matlab.engine
import pandas as pd

from src import rl, utils
from src.ddpg import DDPGAgent, create_ddpg_actor_critic
from src.experience import ExperienceSource, ExperienceSourceDiscountedSteps
from src.noise import GaussianNoise
from src.plot import find_test_eff, plot_folder, sort_eff
from src.policies import BasePolicy, DDPGPolicy, PerturbObservePolicyDCDC
from src.pv_array_dcdc import PVArray
from src.pv_env_dcdc import PVEnv
from src.reward import RewardPowerDeltaPower
from src.schedule import LinearSchedule
from src.utils import grid_generator, mse, read_weather_csv, save_dict

PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
WEATHER_TRAIN_PATH = os.path.join("data", "weather_real_train.csv")
WEATHER_VAL_PATH = os.path.join("data", "weather_real_val.csv")
WEATHER_TEST_PATH = os.path.join("data", "weather_real_test.csv")
WEATHER_SIM_PATH = os.path.join("data", "weather_sim.csv")
STATES = ["v_norm", "p_norm", "dv_norm"]
DUTY_CYCLE_INITIAL = 0.0
DC_STEP = 0.02


class Envs(NamedTuple):
    trn: Optional[PVEnv]
    val: Optional[PVEnv]
    tst: Optional[PVEnv]


# class Policies(NamedTuple):


class RLMPPT(ABC):
    def __init__(
        self,
        epochs: int = 10_000,
        gamma: float = 0.99,
        n_steps: int = 1,
        use_real_weather: bool = True,
        model_name: str = "pv_boost_avg_rload",
    ):
        self.exp_conf = locals()
        self.epochs = epochs
        self.gamma = gamma
        self.n_steps = n_steps
        self.use_real_weather = use_real_weather

        self.path = utils.make_datetime_folder(os.path.join("data", "dataframes"))

        time.sleep(3)

        self.reward_fn = RewardPowerDeltaPower(norm=True)
        self.pvarray = PVArray.from_json(
            path=PV_PARAMS_PATH,
            engine=ENGINE,
            model_name=model_name,
        )

        self.agent = self._get_agent()

    def learn(self, **kwargs) -> None:
        "Perform the agent learning"
        self.agent.learn(epochs=self.epochs, **kwargs)

    def export_results(self) -> None:
        "Save the plots of the results and a summary"
        self.play_expert_test_episode()
        self.play_agent_test_episode()

        path = plot_folder(y=["p"])
        path = plot_folder(y=["dc"])
        agent_test_eff = find_test_eff(path)

        self.exp_conf.update(self._get_conf_dict())
        self.exp_conf.update({"agent": self.__class__.__name__})
        if "self" in self.exp_conf.keys():
            self.exp_conf.pop("self")
        if "__class__" in self.exp_conf.keys():
            self.exp_conf.pop("__class__")

        save_dict(self.exp_conf, os.path.join(self.path, "exp_conf.json"))

        result = f"{self.path}_agent_test_eff_{agent_test_eff}\n"

        with open("results.txt", "a") as f:
            f.write(result)

    def play_expert_test_episode(self) -> None:
        "Use the expert to play an episode in the test environment"
        expert_source = self._get_expert_test_source()
        expert_source.play_all_episodes()

    def play_agent_test_episode(self) -> None:
        "Use the agent to play an episode in the test environment"
        agent_source = self._get_agent_test_source()
        agent_source.play_all_episodes()

    def _create_exp_source(self, policy: BasePolicy, name: str) -> BasePolicy:
        "Create a experience source from a policy"
        exp_source = ExperienceSourceDiscountedSteps(
            policy=policy,
            gamma=self.gamma,
            n_steps=self.n_steps,
            steps=1,
            pvenv_kwargs={
                "save_dataframe": True,
                "include_true_mpp": True,
                "policy_name": name,
                "basepath": self.path,
            },
        )

        return exp_source

    def _get_weather_dataframes(self) -> Tuple[pd.DataFrame]:
        if self.use_real_weather:
            return (
                utils.read_weather_csv(WEATHER_TRAIN_PATH, format=None),
                utils.read_weather_csv(WEATHER_VAL_PATH, format=None),
                utils.read_weather_csv(WEATHER_TEST_PATH, format=None),
            )
        return (
            utils.read_weather_csv(WEATHER_SIM_PATH),
            utils.read_weather_csv(WEATHER_SIM_PATH),
            utils.read_weather_csv(WEATHER_SIM_PATH),
        )

    def _get_envs(self) -> Envs:
        "Return the environments to be used by the agent"
        trn_df, val_df, tst_df = self._get_weather_dataframes()
        trn_env = PVEnv(
            pvarray=self.pvarray,
            weather_df=trn_df,
            states=STATES,
            reward_fn=self.reward_fn,
            dc0=DUTY_CYCLE_INITIAL,
        )
        val_env = PVEnv(
            pvarray=self.pvarray,
            weather_df=val_df,
            states=STATES,
            reward_fn=self.reward_fn,
            dc0=DUTY_CYCLE_INITIAL,
        )
        tst_env = PVEnv(
            pvarray=self.pvarray,
            weather_df=tst_df,
            states=STATES,
            reward_fn=self.reward_fn,
            dc0=DUTY_CYCLE_INITIAL,
        )

        return Envs(trn_env, val_env, tst_env)

    @abstractmethod
    def _get_agent() -> Union[rl.Agent, rl.BCAgent]:
        "Return the agent to be used"

    @abstractmethod
    def _get_conf_dict() -> Dict[str, Any]:
        "Return the experiment configuration (to save to a file later)"

    @abstractmethod
    def _get_expert_test_source() -> ExperienceSource:
        "Return the expert source for the test environment"

    @abstractmethod
    def _get_agent_test_source() -> ExperienceSource:
        "Return the agent source for the test environment"

    @classmethod
    def from_grid(
        cls, grid: Dict[str, Any], repeat: int = 1
    ) -> Generator[RLMPPT, None, None]:
        gg = grid_generator(grid)
        return (cls(**params) for params in gg for _ in range(repeat))


class BCMPPT(RLMPPT):
    def __init__(
        self,
        epochs: int,
        n_steps: int,
        use_real_weather: bool,
        demo_buffer_size: int,
        demo_batch_size: int,
        actor_lr: int,
        actor_l2: int,
        model_name: str = "pv_boost_avg_rload",
    ):
        self._locals = locals()
        self.demo_buffer_size = demo_buffer_size
        self.demo_batch_size = demo_batch_size
        self.actor_lr = actor_lr
        self.actor_l2 = actor_l2
        super().__init__(
            epochs=epochs,
            gamma=1.0,
            n_steps=n_steps,
            use_real_weather=use_real_weather,
            model_name=model_name,
        )

    def _get_conf_dict(self) -> Dict[str, Any]:
        "Return the experiment configuration (to save to a file later)"
        return self._locals

    def _get_expert_test_source(self) -> ExperienceSource:
        "Return the expert source for the test environment"
        return self.agent.demo_test_source

    def _get_agent_test_source(self) -> ExperienceSource:
        "Return the agent source for the test environment"
        return self.agent.agent_test_source

    def _get_agent(self) -> rl.BCAgent:
        "Return the agent to be used"
        trn_env, val_env, tst_env = self._get_envs()
        actor, _ = rl.create_ddpg_actor_critic(trn_env)
        agent_val_policy = DDPGPolicy(
            env=val_env,
            net=actor,
            noise=None,
            schedule=None,
        )
        agent_tst_policy = DDPGPolicy(
            env=tst_env,
            net=actor,
            noise=None,
            schedule=None,
        )
        demo_trn_policy = PerturbObservePolicyDCDC(
            env=trn_env,
            v_step=DC_STEP,
            dv_index="dv",
            dp_index="dp",
        )
        demo_val_policy = PerturbObservePolicyDCDC(
            env=val_env,
            v_step=DC_STEP,
            dv_index="dv",
            dp_index="dp",
        )
        demo_tst_policy = PerturbObservePolicyDCDC(
            env=tst_env,
            v_step=DC_STEP,
            dv_index="dv",
            dp_index="dp",
        )
        agent_val_source = self._create_exp_source(agent_val_policy, "bc-agent-val")
        agent_tst_source = self._create_exp_source(agent_tst_policy, "bc-agent-test")
        demo_trn_source = self._create_exp_source(demo_trn_policy, "po-expert-train")
        demo_val_source = self._create_exp_source(demo_val_policy, "po-expert-val")
        demo_tst_source = self._create_exp_source(demo_tst_policy, "po-expert-test")
        agent = rl.BCAgent(
            demo_train_source=demo_trn_source,
            demo_val_source=demo_val_source,
            demo_test_source=demo_tst_source,
            agent_val_source=agent_val_source,
            agent_test_source=agent_tst_source,
            actor=actor,
            demo_buffer_size=self.demo_buffer_size,
            demo_batch_size=self.demo_batch_size,
            actor_lr=self.actor_lr,
            actor_l2=self.actor_l2,
        )

        return agent


class DDPGMPPT(RLMPPT):
    def __init__(
        self,
        epochs: int,
        n_steps: int,
        use_real_weather: bool,
        demo_buffer_size: int,
        demo_batch_size: int,
        actor_lr: int,
        actor_l2: int,
        model_name: str = "pv_boost_avg_rload",
    ):
        self._locals = locals()
        self.demo_buffer_size = demo_buffer_size
        self.demo_batch_size = demo_batch_size
        self.actor_lr = actor_lr
        self.actor_l2 = actor_l2
        super().__init__(
            epochs=epochs,
            gamma=1.0,
            n_steps=n_steps,
            use_real_weather=use_real_weather,
            model_name=model_name,
        )

    def _get_conf_dict(self) -> Dict[str, Any]:
        "Return the experiment configuration (to save to a file later)"
        return self._locals

    def _get_expert_test_source(self) -> ExperienceSource:
        "Return the expert source for the test environment"
        return self.agent.demo_test_source

    def _get_agent_test_source(self) -> ExperienceSource:
        "Return the agent source for the test environment"
        return self.agent.agent_test_source

    def _get_agent(self) -> rl.BCAgent:
        "Return the agent to be used"
        trn_env, val_env, tst_env = self._get_envs()
        actor, _ = rl.create_ddpg_actor_critic(trn_env)
        agent_val_policy = DDPGPolicy(
            env=val_env,
            net=actor,
            noise=None,
            schedule=None,
        )
        agent_tst_policy = DDPGPolicy(
            env=tst_env,
            net=actor,
            noise=None,
            schedule=None,
        )
        demo_trn_policy = PerturbObservePolicyDCDC(
            env=trn_env,
            v_step=DC_STEP,
            dv_index="dv",
            dp_index="dp",
        )
        demo_val_policy = PerturbObservePolicyDCDC(
            env=val_env,
            v_step=DC_STEP,
            dv_index="dv",
            dp_index="dp",
        )
        demo_tst_policy = PerturbObservePolicyDCDC(
            env=tst_env,
            v_step=DC_STEP,
            dv_index="dv",
            dp_index="dp",
        )
        agent_val_source = self._create_exp_source(agent_val_policy, "bc-agent-val")
        agent_tst_source = self._create_exp_source(agent_tst_policy, "bc-agent-test")
        demo_trn_source = self._create_exp_source(demo_trn_policy, "po-expert-train")
        demo_val_source = self._create_exp_source(demo_val_policy, "po-expert-val")
        demo_tst_source = self._create_exp_source(demo_tst_policy, "po-expert-test")
        agent = rl.BCAgent(
            demo_train_source=demo_trn_source,
            demo_val_source=demo_val_source,
            demo_test_source=demo_tst_source,
            agent_val_source=agent_val_source,
            agent_test_source=agent_tst_source,
            actor=actor,
            demo_buffer_size=self.demo_buffer_size,
            demo_batch_size=self.demo_batch_size,
            actor_lr=self.actor_lr,
            actor_l2=self.actor_l2,
        )

        return agent


# class DDPGMPPT:
#     def __init__(
#         self,
#         gamma,
#         n_steps,
#         batch_size,
#         buffer_size,
#         actor_lr,
#         actor_l2,
#         critic_lr,
#         critic_l2,
#         tau,
#         noise_steps,
#         decrease_noise,
#         eps_final,
#         norm_rewards,
#         epochs,
#     ):
#         # self.exp_conf = locals()
#         # self.epochs = epochs

#         # self.basepath = os.path.join(
#         #     "data", "dataframes", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         # )

#         # os.makedirs(self.basepath)

#         # time.sleep(3)

#         # reward_fn = RewardPowerDeltaPower(norm=True)
#         # pvarray = PVArray.from_json(
#         #     path=PV_PARAMS_PATH,
#         #     engine=ENGINE,
#         #     model_name=MODEL_NAME,
#         # )
#         weather_df = read_weather_csv(WEATHER_TRAIN_PATH, format=None)
#         weather_test_df = read_weather_csv(WEATHER_TEST_PATH, format=None)
#         weather_sim_df = read_weather_csv(WEATHER_SIM_PATH)
#         env = PVEnv(
#             pvarray=pvarray,
#             weather_df=weather_df,
#             # weather_df=weather_sim_df,
#             states=STATES,
#             reward_fn=reward_fn,
#             dc0=DUTY_CYCLE_INITIAL,
#         )
#         test_env = PVEnv(
#             pvarray=pvarray,
#             weather_df=weather_test_df,
#             # weather_df=weather_sim_df,
#             states=STATES,
#             reward_fn=reward_fn,
#             dc0=DUTY_CYCLE_INITIAL,
#         )
#         actor, critic = create_ddpg_actor_critic(env=env)
#         noise = GaussianNoise(mean=0.0, std=0.4)
#         # noise = OUNoise(mean=0.0, std=2, theta=0.0, dt=1)
#         schedule = LinearSchedule(
#             max_steps=noise_steps,
#             eps_final=eps_final,
#         )
#         # schedule = ConstantSchedule(1.0)
#         collect_policy = DDPGPolicy(
#             env=env,
#             net=actor,
#             noise=noise,
#             schedule=schedule,
#             decrease_noise=decrease_noise,
#         )
#         test_policy = DDPGPolicy(env=test_env, net=actor, noise=None, schedule=None)
#         po_policy = PerturbObservePolicyDCDC(
#             env=test_env, v_step=DC_STEP, dv_index="dv", dp_index="dp"
#         )
#         train_exp_source = ExperienceSourceDiscountedSteps(
#             policy=collect_policy,
#             gamma=gamma,
#             n_steps=n_steps,
#             steps=1,
#             pvenv_kwargs={
#                 "save_dataframe": True,
#                 "include_true_mpp": True,
#                 "policy_name": "ddpg-train",
#                 "basepath": self.basepath,
#             },
#         )
#         test_exp_source = ExperienceSourceDiscountedSteps(
#             policy=test_policy,
#             gamma=gamma,
#             n_steps=n_steps,
#             steps=1,
#             pvenv_kwargs={
#                 "save_dataframe": True,
#                 "include_true_mpp": True,
#                 "policy_name": "ddpg-test",
#                 "basepath": self.basepath,
#             },
#         )
#         self.po_exp_source = ExperienceSourceDiscountedSteps(
#             policy=po_policy,
#             gamma=gamma,
#             n_steps=n_steps,
#             steps=1,
#             pvenv_kwargs={
#                 "save_dataframe": True,
#                 "include_true_mpp": True,
#                 "policy_name": "po-test",
#                 "basepath": self.basepath,
#             },
#         )

#         self.agent = DDPGAgent(
#             train_exp_source=train_exp_source,
#             test_exp_sorce=test_exp_source,
#             actor=actor,
#             critic=critic,
#             buffer_size=buffer_size,
#             batch_size=batch_size,
#             actor_lr=actor_lr,
#             critic_lr=critic_lr,
#             actor_l2=actor_l2,
#             critic_l2=critic_l2,
#             tau=tau,
#             norm_rewards=norm_rewards,
#         )

# def learn(self) -> None:
#     self.agent.learn(epochs=self.epochs, train_steps=1, collect_steps=1)

# def make_results(self) -> None:
#     self.po_exp_source.play_episode()
#     self.agent.test_exp_sorce.play_episode()

#     path = plot_folder(y=["p"])
#     path = plot_folder(y=["dc"])
#     test_eff = find_test_eff(path)

#     self.exp_conf.update(
#         {
#             "test_eff": test_eff,
#             "states": STATES,
#             "duty_cycle_inital": DUTY_CYCLE_INITIAL,
#             "duty_cycle_step": DC_STEP,
#             "schedule": str(self.agent.train_exp_source.policy.schedule),
#             "noise": str(self.agent.train_exp_source.policy.noise),
#             "reward": str(self.agent.train_exp_source.policy.env.reward_fn),
#             "model_name": MODEL_NAME,
#         }
#     )
#     if "self" in self.exp_conf.keys():
#         self.exp_conf.pop("self")

#     save_dict(self.exp_conf, os.path.join(self.basepath, "exp_conf.json"))

#     result = f"{self.basepath}_test_eff_{test_eff}\n"

#     with open("results.txt", "a") as f:
#         f.write(result)


# hparams = {
#     "gamma": [0.1],
#     "n_steps": [1],
#     "batch_size": [64, 128],
#     "buffer_size": [50_000],
#     "actor_lr": [1e-3],
#     "actor_l2": [1e-3],
#     "critic_lr": [1e-3],
#     "critic_l2": [1e-5],
#     "tau": [1e-3],
#     "noise_steps": [5000],
#     "decrease_noise": [True],
#     "eps_final": [0.0],
#     "norm_rewards": [False],
#     "epochs": [20_000],
#     # "epochs": [20_000],
# }

# gg = grid_generator(hparams)

# for dic_hparams in gg:
#     for _ in range(5):
#         mppt = DDPGMPPT(**dic_hparams)
#         mppt.learn()
#         mppt.make_results()


if __name__ == "__main__":
    try:
        ENGINE.quit()  # type: ignore
    except NameError:
        pass
    ENGINE = matlab.engine.connect_matlab()

    grid = {
        "epochs": [100_000],
        "n_steps": [1],
        "use_real_weather": [True],
        # "use_real_weather": [False],
        "demo_buffer_size": [5000],
        "demo_batch_size": [512],
        "actor_lr": [1e-2],
        "actor_l2": [1e-4],
    }

    for agent in BCMPPT.from_grid(grid, repeat=50):
        agent.learn(val_every=500)
        agent.export_results()

    sort_eff()

    # bc_agent = BCMPPT(
    #     epochs=10_000,
    #     n_steps=1,
    #     use_real_weather=False,
    #     demo_buffer_size=5000,
    #     demo_batch_size=512,
    #     actor_lr=1e-3,
    #     actor_l2=1e-4,
    # )

    # bc_agent.learn(val_every=500)
    # bc_agent.export_results()