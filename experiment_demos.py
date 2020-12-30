import os
import time

import gym
import matlab.engine
import torch

from typing import Optional

from src.ddpg_demos import DDPGAgentDemo, create_ddpg_actor_critic
from src.experience import ExperienceSourceDiscountedSteps
from src.noise import GaussianNoise, OUNoise, Noise
from src.policies import DDPGPolicy, PerturbObservePolicy
from src.pv_env import PVEnv
from src.pv_array import PVArray
from src.schedule import LinearSchedule, ConstantSchedule, Schedule
from src.reward import RewardPowerDeltaPower
from src.logger import logger
from src.utils import read_weather_csv, save_dict


PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
PV_CKP_PATH = os.path.join("data", "01_pvarray_iv.json")
WEATHER_TRAIN_PATH = os.path.join("data", "weather_real_train.csv")
WEATHER_TEST_PATH = os.path.join("data", "weather_real_test.csv")
WEATHER_SIM_PATH = os.path.join("data", "weather_sim.csv")
STATES = ["v_norm", "p_norm", "dv_norm"]
V_INITIAL = 22

GAMMA = 0.5
N_STEPS = 1
BATCH_SIZE = 256
# BATCH_SIZE = 8
DEMO_BATCH_SIZE = 256
# DEMO_BATCH_SIZE = 8
BUFFER_SIZE = 5_000
DEMO_BUFFER_SIZE = 3_000
EPOCHS = 1_000
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
CRITIC_L2 = 1e-2
ACTOR_L2 = 1e-3
TAU = 1e-2
DECREASE_NOISE = True
NOISE_STEPS = 8_000
NORM_REWARDS = True
LAMBDA_RL = 1
LAMBDA_BC = 1
Q_FILTER = False


try:
    engine.quit()
except NameError:
    pass
engine = matlab.engine.connect_matlab()


def get_envs():
    reward_fn = RewardPowerDeltaPower(norm=True)
    pvarray = PVArray.from_json(
        path=PV_PARAMS_PATH,
        engine=engine,
        ckp_path=PV_CKP_PATH,
        f_precision=1,
    )
    weather_df = read_weather_csv(WEATHER_TRAIN_PATH)
    weather_test_df = read_weather_csv(WEATHER_TEST_PATH)
    env = PVEnv(
        pvarray=pvarray,
        weather_df=weather_df,
        states=STATES,
        reward_fn=reward_fn,
        max_steps=500,
    )
    demo_env = PVEnv(
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
        v0=12,
        deterministic=True,
    )

    return env, demo_env, test_env


class Experiment:
    def __init__(
        self,
        gamma: float,
        n_steps: int,
        batch_size: int,
        demo_batch_size: int,
        buffer_size: int,
        demo_buffer_size: int,
        actor_lr: float,
        critic_lr: float,
        actor_l2: float,
        critic_l2: float,
        tau: float,
        noise: Optional[Noise],
        schedule: Optional[Schedule],
        norm_rewards: bool,
        decrease_noise: bool,
        lambda_rl: float,
        lambda_bc: float,
        use_checkpoint: str = None,
        q_filter: bool = False,
    ):
        self.gamma = gamma
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.demo_batch_size = batch_size
        self.buffer_size = buffer_size
        self.demo_buffer_size = demo_buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_l2 = actor_l2
        self.critic_l2 = critic_l2
        self.tau = tau
        self.noise = noise
        self.schedule = schedule
        self.norm_rewards = norm_rewards
        self.decrease_noise = decrease_noise
        self.lambda_rl = lambda_rl
        self.lambda_bc = lambda_bc
        self.q_filter = q_filter

        name = "exp" + str(len(os.listdir("experiments")) // 4).zfill(4)
        self.path = os.path.join("experiments", name)

        self.env, self.demo_env, self.test_env = get_envs()
        actor, critic = create_ddpg_actor_critic(env=self.env)
        if use_checkpoint:
            actor, critic = self.load_actor_critic(actor, critic, use_checkpoint)
        collect_policy = PerturbObservePolicy(
            env=self.env, v_step=0.2, dv_index="dv", dp_index="dp", noise=None
        )
        # collect_policy = DDPGPolicy(
        #     env=self.env,
        #     net=actor,
        #     # noise=noise,
        #     # schedule=schedule,
        #     noise=None,
        #     schedule=None,
        #     decrease_noise=decrease_noise,
        # )
        test_policy = DDPGPolicy(
            env=self.test_env, net=actor, noise=None, schedule=None
        )
        po_policy = PerturbObservePolicy(
            env=self.demo_env, v_step=0.2, dv_index="dv", dp_index="dp", noise=None
        )
        self.train_exp_source = ExperienceSourceDiscountedSteps(
            policy=collect_policy, gamma=gamma, n_steps=n_steps, steps=1
        )
        self.test_exp_source = ExperienceSourceDiscountedSteps(
            policy=test_policy, gamma=gamma, n_steps=n_steps, steps=1
        )
        self.po_exp_source = ExperienceSourceDiscountedSteps(
            policy=po_policy, gamma=gamma, n_steps=n_steps, steps=1
        )
        self.agent = DDPGAgentDemo(
            train_exp_source=self.train_exp_source,
            demo_exp_source=self.po_exp_source,
            test_exp_sorce=self.test_exp_source,
            actor=actor,
            critic=critic,
            buffer_size=buffer_size,
            demo_buffer_size=demo_buffer_size,
            batch_size=batch_size,
            demo_batch_size=demo_batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_l2=actor_l2,
            critic_l2=critic_l2,
            tau=tau,
            norm_rewards=norm_rewards,
            lambda_rl=lambda_rl,
            lambda_bc=lambda_bc,
            q_filter=q_filter,
        )

    def run(self, epochs: int = 10_000) -> None:
        self.agent.learn(epochs=epochs, train_steps=5, log_every=1000)
        # self.test_exp_source.play_episode()
        # self.test_env.render_vs_true()
        # self.po_exp_source.play_episode()
        # self.test_env.render_vs_true(label="PO")

    def save(self) -> None:
        dic_path = self.path + ".json"

        dic = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "demo_batch_size": self.demo_batch_size,
            "buffer_size": self.buffer_size,
            "demo_buffer_size": self.demo_buffer_size,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "actor_l2": self.actor_l2,
            "critic_l2": self.critic_l2,
            "tau": self.tau,
            "noise": str(self.noise),
            "schedule": str(self.schedule),
            "norm_rewards": self.norm_rewards,
            "decrease_noise": self.decrease_noise,
            "lambda_rl": self.lambda_rl,
            "lambda_bc": self.lambda_bc,
            "q_filter": self.q_filter,
        }
        save_dict(dic, dic_path)
        self.save_agent(self.path + ".tar")

        self.test_exp_source.play_episode()
        self.test_env.render_vs_true(show=False, save_path=self.path)

    def test(self, episodes: int = 5) -> float:
        eff = 0
        for _ in range(episodes):
            self.test_exp_source.play_episode()
            p_real, v_real, _ = self.test_env.pvarray.get_true_mpp(
                self.test_env.history.g, self.test_env.history.t
            )
            eff += PVArray.mppt_eff(p_real, self.test_env.history.p)

        eff /= episodes
        return eff

    def save_agent(self, path: str) -> None:
        self.agent.save(path)

    @staticmethod
    def load_actor_critic(
        actor: torch.nn.Module, critic: torch.nn.Module, path: str
    ) -> (torch.nn.Module, torch.nn.Module):
        checkpoint = torch.load(path)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        return actor, critic


noise = GaussianNoise(mean=0.0, std=5.0)
# noise = OUNoise(mean=0.0, std=2, theta=0.0, dt=1)
schedule = LinearSchedule(max_steps=NOISE_STEPS, eps_final=0.1)
# schedule = ConstantSchedule(1.0)

max_test_eff = 0
for Q_FILTER in [True, False]:

    for _ in range(3):
        schedule.reset()

        experiment = Experiment(
            gamma=GAMMA,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            demo_batch_size=DEMO_BATCH_SIZE,
            buffer_size=BUFFER_SIZE,
            demo_buffer_size=DEMO_BUFFER_SIZE,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            actor_l2=ACTOR_L2,
            critic_l2=CRITIC_L2,
            tau=TAU,
            noise=noise,
            schedule=schedule,
            norm_rewards=NORM_REWARDS,
            decrease_noise=DECREASE_NOISE,
            lambda_rl=LAMBDA_RL,
            lambda_bc=LAMBDA_BC,
            q_filter=Q_FILTER,
        )

        experiment.train_exp_source.play_episode()
        experiment.env.render_vs_true(label="PO")
        experiment.po_exp_source.play_episode()
        experiment.demo_env.render_vs_true(label="PO")

        experiment.run(epochs=EPOCHS)
        experiment.save()
        eff = experiment.test(episodes=1)

        if eff > max_test_eff:
            max_test_eff = eff
            experiment.save_agent("experiments\\best.tar")

# Testing
experiment = Experiment(
    gamma=GAMMA,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    demo_batch_size=DEMO_BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    demo_buffer_size=DEMO_BUFFER_SIZE,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    actor_l2=ACTOR_L2,
    critic_l2=CRITIC_L2,
    tau=TAU,
    noise=noise,
    schedule=schedule,
    norm_rewards=NORM_REWARDS,
    decrease_noise=DECREASE_NOISE,
    use_checkpoint="experiments\\best.tar",
    lambda_bc=LAMBDA_BC,
    lambda_rl=LAMBDA_RL,
    q_filter=Q_FILTER,
)

# Re-run the best experiment
eff = experiment.test()
experiment.save()
print(eff)


# Compare RL to PO
experiment.test_exp_source.play_episode()
experiment.test_env.render_vs_true()

po_policy = PerturbObservePolicy(
    env=experiment.test_env, v_step=0.2, dv_index="dv", dp_index="dp", noise=None
)

po_exp_source = ExperienceSourceDiscountedSteps(
    policy=po_policy, gamma=GAMMA, n_steps=N_STEPS, steps=1
)
po_exp_source.play_episode()
experiment.test_env.render_vs_true(label="PO")
