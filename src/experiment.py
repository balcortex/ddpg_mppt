from __future__ import annotations
import os
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union, Generator

import matlab.engine
import pandas as pd
import torch

from src import rl, utils
from src.experience import ExperienceSource, ExperienceSourceDiscountedSteps
from src.noise import GaussianNoise
from src.plot import find_test_eff, plot_folder, sort_eff
from src.policies import BasePolicy, DDPGPolicy, PerturbObservePolicyDCDC
from src.pv_array_dcdc import PVArray
from src.pv_env_dcdc import PVEnv
from src.reward import RewardPowerDeltaPower
from src.schedule import LinearSchedule
from src.utils import grid_generator, save_dict

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
    def _get_agent() -> Union[
        rl.Agent, rl.BCAgent, rl.DDPGAgent, rl.DDPGWarmStartAgent
    ]:
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
        actor_lr: float,
        actor_l2: float,
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
        return self.demo_test_source

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
        self.demo_test_source = self._create_exp_source(
            demo_tst_policy, "po-expert-test"
        )
        agent = rl.BCAgent(
            demo_train_source=demo_trn_source,
            demo_val_source=demo_val_source,
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
        gamma: float,
        use_real_weather: bool,
        buffer_size: int,
        batch_size: int,
        actor_lr: float,
        actor_l2: float,
        critic_lr: float,
        critic_l2: float,
        tau: float,
        noise_mean: float,
        noise_std: float,
        noise_steps: float,
        norm_rewards: bool,
        model_name: str = "pv_boost_avg_rload",
    ):
        self._locals = locals()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.actor_l2 = actor_l2
        self.critic_lr = critic_lr
        self.critic_l2 = critic_l2
        self.tau = tau
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_steps = noise_steps
        self.norm_rewards = norm_rewards
        super().__init__(
            epochs=epochs,
            gamma=gamma,
            n_steps=n_steps,
            use_real_weather=use_real_weather,
            model_name=model_name,
        )

    def _get_conf_dict(self) -> Dict[str, Any]:
        "Return the experiment configuration (to save to a file later)"
        return self._locals

    def _get_expert_test_source(self) -> ExperienceSource:
        "Return the expert source for the test environment"
        return self.expert_test_source

    def _get_agent_test_source(self) -> ExperienceSource:
        "Return the agent source for the test environment"
        return self.agent.agent_test_source

    def _get_agent(self, actor: Optional[rl.DDPGActor] = None) -> rl.DDPGAgent:
        "Return the agent to be used"
        trn_env, _, tst_env = self._get_envs()
        actor_, critic = rl.create_ddpg_actor_critic(trn_env)
        actor = actor or actor_
        agent_collect_policy = DDPGPolicy(
            env=trn_env,
            net=actor,
            noise=GaussianNoise(self.noise_mean, self.noise_std),
            schedule=LinearSchedule(max_steps=self.noise_steps),
            decrease_noise=True,
        )
        agent_test_policy = DDPGPolicy(
            env=tst_env,
            net=actor,
            noise=None,
            schedule=None,
        )
        demo_test_policy = PerturbObservePolicyDCDC(
            env=tst_env,
            v_step=DC_STEP,
            dv_index="dv",
            dp_index="dp",
        )
        collect_source = self._create_exp_source(
            agent_collect_policy, "ddpg-agent-collect"
        )
        agent_test_source = self._create_exp_source(
            agent_test_policy, "ddpg-agent-test"
        )
        self.expert_test_source = self._create_exp_source(
            demo_test_policy, "po-expert-test"
        )
        agent = rl.DDPGAgent(
            collect_exp_source=collect_source,
            agent_test_source=agent_test_source,
            actor=actor,
            critic=critic,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            actor_l2=self.actor_l2,
            critic_l2=self.critic_l2,
            tau=self.tau,
            norm_rewards=self.norm_rewards,
        )

        return agent


class DPPGWarmStartMPPT(RLMPPT):
    def __init__(
        self,
        demo_epochs: int,
        demo_buffer_size: int,
        demo_batch_size: int,
        demo_actor_lr: float,
        demo_actor_l2: float,
        demo_val_every_steps: int,
        epochs: int,
        n_steps: int,
        gamma: float,
        use_real_weather: bool,
        buffer_size: int,
        batch_size: int,
        actor_lr: float,
        actor_l2: float,
        critic_lr: float,
        critic_l2: float,
        tau: float,
        noise_mean: float,
        noise_std: float,
        noise_steps: float,
        norm_rewards: bool,
        decrease_noise: bool,
        critic_pretrain_steps: int,
        model_name: str = "pv_boost_avg_rload",
    ):
        self._locals = locals()
        self.demo_epochs = demo_epochs
        self.demo_buffer_size = demo_buffer_size
        self.demo_batch_size = demo_batch_size
        self.demo_actor_lr = demo_actor_lr
        self.demo_actor_l2 = demo_actor_l2
        self.demo_val_every_steps = demo_val_every_steps
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.actor_l2 = actor_l2
        self.critic_lr = critic_lr
        self.critic_l2 = critic_l2
        self.tau = tau
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_steps = noise_steps
        self.norm_rewards = norm_rewards
        self.decrease_noise = decrease_noise
        self.critic_pretrain_steps = critic_pretrain_steps

        super().__init__(
            epochs=epochs,
            gamma=gamma,
            n_steps=n_steps,
            use_real_weather=use_real_weather,
            model_name=model_name,
        )

    def _get_conf_dict(self) -> Dict[str, Any]:
        "Return the experiment configuration (to save to a file later)"
        return self._locals

    def _get_expert_test_source(self) -> ExperienceSource:
        "Return the expert source for the test environment"
        return self.expert_test_source

    def _get_agent_test_source(self) -> ExperienceSource:
        "Return the agent source for the test environment"
        return self.agent.agent_test_source

    def _get_agent(self) -> rl.DDPGWarmStartAgent:
        "Return the agent to be used"
        trn_env, val_env, tst_env = self._get_envs()
        actor, critic = rl.create_ddpg_actor_critic(trn_env)
        demo_collect_policy = PerturbObservePolicyDCDC(env=trn_env, v_step=DC_STEP)
        demo_val_policy = PerturbObservePolicyDCDC(env=val_env, v_step=DC_STEP)
        demo_test_policy = PerturbObservePolicyDCDC(env=tst_env, v_step=DC_STEP)
        agent_collect_policy = DDPGPolicy(
            env=trn_env,
            net=actor,
            noise=GaussianNoise(self.noise_mean, self.noise_std),
            schedule=LinearSchedule(max_steps=self.noise_steps),
            decrease_noise=self.decrease_noise,
        )
        agent_val_policy = DDPGPolicy(env=val_env, net=actor)
        agent_test_policy = DDPGPolicy(env=tst_env, net=actor)

        demo_collect_source = self._create_exp_source(
            demo_collect_policy, "po-expert-collect"
        )
        demo_val_source = self._create_exp_source(demo_val_policy, "po-expert-val")
        self.expert_test_source = self._create_exp_source(
            demo_test_policy, "po-expert-test"
        )
        collect_source = self._create_exp_source(
            agent_collect_policy, "ddpg-agent-collect"
        )
        agent_val_source = self._create_exp_source(agent_val_policy, "ddpg-agent-val")
        agent_test_source = self._create_exp_source(
            agent_test_policy, "ddpg-agent-test"
        )

        agent = rl.DDPGWarmStartAgent(
            demo_train_source=demo_collect_source,
            demo_val_source=demo_val_source,
            agent_val_source=agent_val_source,
            agent_test_source=agent_test_source,
            collect_exp_source=collect_source,
            actor=actor,
            critic=critic,
            demo_buffer_size=self.demo_buffer_size,
            demo_batch_size=self.demo_batch_size,
            demo_actor_lr=self.demo_actor_lr,
            demo_actor_l2=self.demo_actor_l2,
            demo_epochs=self.demo_epochs,
            demo_val_every_steps=self.demo_val_every_steps,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            actor_l2=self.actor_l2,
            critic_l2=self.critic_l2,
            tau=self.tau,
            norm_rewards=self.norm_rewards,
            critic_pretrain_steps=self.critic_pretrain_steps,
        )

        return agent


class DPPGCoLMPPT(RLMPPT):
    def __init__(
        self,
        demo_buffer_size: int,
        epochs: int,
        n_steps: int,
        gamma: float,
        use_real_weather: bool,
        buffer_size: int,
        batch_size: int,
        actor_lr: float,
        actor_l2: float,
        critic_lr: float,
        critic_l2: float,
        tau: float,
        lambda_q1: float,
        lambda_bc: float,
        lambda_a: float,
        noise_mean: float,
        noise_std: float,
        noise_steps: float,
        norm_rewards: bool,
        decrease_noise: bool,
        pretrain_steps: int,
        model_name: str = "pv_boost_avg_rload",
    ):
        self._locals = locals()
        self.demo_buffer_size = demo_buffer_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.actor_l2 = actor_l2
        self.critic_lr = critic_lr
        self.critic_l2 = critic_l2
        self.tau = tau
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_steps = noise_steps
        self.norm_rewards = norm_rewards
        self.decrease_noise = decrease_noise
        self.pretrain_steps = pretrain_steps
        self.lambda_q1 = lambda_q1
        self.lambda_bc = lambda_bc
        self.lambda_a = lambda_a

        super().__init__(
            epochs=epochs,
            gamma=gamma,
            n_steps=n_steps,
            use_real_weather=use_real_weather,
            model_name=model_name,
        )

    def _get_conf_dict(self) -> Dict[str, Any]:
        "Return the experiment configuration (to save to a file later)"
        return self._locals

    def _get_expert_test_source(self) -> ExperienceSource:
        "Return the expert source for the test environment"
        return self.expert_test_source

    def _get_agent_test_source(self) -> ExperienceSource:
        "Return the agent source for the test environment"
        return self.agent_test_source

    def _get_agent(self) -> rl.DDPGWarmStartAgent:
        "Return the agent to be used"
        trn_env, _, tst_env = self._get_envs()
        actor, critic = rl.create_ddpg_actor_critic(trn_env)
        demo_collect_policy = PerturbObservePolicyDCDC(env=trn_env, v_step=DC_STEP)
        demo_test_policy = PerturbObservePolicyDCDC(env=tst_env, v_step=DC_STEP)
        agent_collect_policy = DDPGPolicy(
            env=trn_env,
            net=actor,
            noise=GaussianNoise(self.noise_mean, self.noise_std),
            schedule=LinearSchedule(max_steps=self.noise_steps),
            decrease_noise=self.decrease_noise,
        )
        agent_test_policy = DDPGPolicy(env=tst_env, net=actor)

        demo_collect_source = self._create_exp_source(
            demo_collect_policy, "po-expert-collect"
        )
        self.expert_test_source = self._create_exp_source(
            demo_test_policy, "po-expert-test"
        )
        collect_source = self._create_exp_source(
            agent_collect_policy, "ddpg-agent-collect"
        )
        self.agent_test_source = self._create_exp_source(
            agent_test_policy, "ddpg-agent-test"
        )

        agent = rl.DDPGCoLAgent(
            actor=actor,
            critic=critic,
            demo_source=demo_collect_source,
            collect_source=collect_source,
            actor_lr=self.actor_lr,
            actor_l2=self.actor_l2,
            critic_lr=self.critic_lr,
            critic_l2=self.critic_l2,
            lambda_q1=self.lambda_q1,
            lambda_bc=self.lambda_bc,
            lambda_a=self.lambda_a,
            tau=self.tau,
            demo_buffer_size=self.demo_buffer_size,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            pretrain_steps=self.pretrain_steps,
            norm_rewards=self.norm_rewards,
        )

        return agent


def main_bc():
    grid = {
        "epochs": [100_000],
        "n_steps": [1],
        # "use_real_weather": [True],
        "use_real_weather": [False],
        "demo_buffer_size": [5000],
        "demo_batch_size": [512],
        "actor_lr": [1e-2],
        "actor_l2": [1e-4],
    }

    for agent in BCMPPT.from_grid(grid, repeat=1):
        agent.learn(val_every=500)
        agent.export_results()

    sort_eff()


def main_ddpg():
    grid = {
        "epochs": [20_000],
        "n_steps": [1],
        "gamma": [0.1],
        "use_real_weather": [True],
        # "use_real_weather": [False],
        "buffer_size": [50_000],
        "batch_size": [64, 128, 512],
        "actor_lr": [1e-3],
        "actor_l2": [1e-3],
        "critic_lr": [1e-3],
        "critic_l2": [1e-5],
        "tau": [1e-3],
        "noise_mean": [0.0],
        "noise_std": [0.4],
        "noise_steps": [5000],
        "norm_rewards": [False],
    }

    for agent in DDPGMPPT.from_grid(grid, repeat=5):
        agent.learn(val_every=500)
        agent.export_results()

    sort_eff()


def main_ddpg_warm_start():
    grid = {
        "demo_epochs": [100_000],
        "demo_buffer_size": [5_000],
        "demo_batch_size": [512],
        "demo_actor_lr": [1e-2],
        "demo_actor_l2": [1e-4],
        "demo_val_every_steps": [500],
        "epochs": [1_000],
        "n_steps": [1],
        "gamma": [0.1],
        # "use_real_weather": [True],
        "use_real_weather": [False],
        "buffer_size": [50_000],
        "batch_size": [64],
        "actor_lr": [1e-3],
        "actor_l2": [1e-3],
        "critic_lr": [1e-3],
        "critic_l2": [0],
        "tau": [1e-4],
        "noise_mean": [0.0],
        "noise_std": [0.001],
        "noise_steps": [5000],
        "norm_rewards": [False],
        "decrease_noise": [False],
        "critic_pretrain_steps": [1000, 5000, 10_000, 20_000],
    }
    # grid = {
    #     "demo_epochs": [100_000],
    #     "demo_buffer_size": [5_000],
    #     "demo_batch_size": [512],
    #     "demo_actor_lr": [1e-2],
    #     "demo_actor_l2": [1e-4],
    #     "demo_val_every_steps": [500],
    #     "epochs": [1_000],
    #     "n_steps": [1],
    #     "gamma": [0.1],
    #     # "use_real_weather": [True],
    #     "use_real_weather": [False],
    #     "buffer_size": [50_000],
    #     "batch_size": [64],
    #     "actor_lr": [1e-3],
    #     "actor_l2": [1e-3],
    #     "critic_lr": [1e-3],
    #     "critic_l2": [1e-5],
    #     "tau": [1e-3],
    #     "noise_mean": [0.0],
    #     "noise_std": [0.001],
    #     "noise_steps": [5000],
    #     "norm_rewards": [False],
    #     "decrease_noise": [False],
    #     "critic_pretrain_steps": [1000],
    # }

    for agent in DPPGWarmStartMPPT.from_grid(grid, repeat=1):
        agent.learn(val_every=500)
        agent.export_results()

    sort_eff()


def main_ddpg_col():
    grid = {
        "demo_buffer_size": [5000],
        "epochs": [1000],
        "n_steps": [1],
        "gamma": [0.1],
        "use_real_weather": [False],
        "buffer_size": [50_000],
        "batch_size": [128],
        "actor_lr": [1e-3],
        "actor_l2": [1e-4],
        "critic_lr": [1e-3],
        "critic_l2": [1e-2],
        "tau": [1e-3],
        "lambda_q1": [1.0],
        "lambda_bc": [0.1],
        "lambda_a": [1.0],
        "noise_mean": [0.0],
        "noise_std": [0.001],
        "noise_steps": [5_000],
        "norm_rewards": [False],
        "decrease_noise": [False],
        "pretrain_steps": [20_000],
    }

    for agent in DPPGCoLMPPT.from_grid(grid, repeat=1):
        agent.learn()
        agent.export_results()

    sort_eff()


if __name__ == "__main__":
    try:
        ENGINE.quit()  # type: ignore
    except NameError:
        pass
    ENGINE = matlab.engine.connect_matlab()

    try:
        os.remove("results.txt")
        os.remove("results_sorted.txt")
    except FileNotFoundError:
        pass

    # main_bc()
    # main_ddpg()
    # main_ddpg_warm_start()
    main_ddpg_col()