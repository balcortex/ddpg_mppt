import os

import gym
import matlab.engine
import numpy as np
import torch
from src.noise import GaussianNoise
from src.policies import DDPGPolicy, MPPPolicy, PerturbObservePolicy, RandomPolicy
from src.pv_env import PVEnv
from src.reward import RewardPowerDeltaPower


def test_random_policy_box():
    env = gym.make("Pendulum-v0")
    rand_policy = RandomPolicy(env=env)
    obs = env.reset()
    action = rand_policy(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)
    assert action[0] >= env.action_space.low[0]
    assert action[0] <= env.action_space.high[0]


def test_po_info():
    env = gym.make("Pendulum-v0")
    policy = PerturbObservePolicy(
        env=env,
        v_step=0.1,
        dp_index="dp",
        dv_index="dv",
    )

    obs = np.array([1, 1, 1, 1])
    info = {
        "dp": 0.1,
        "dv": 0.1,
    }
    assert policy(obs, info) == 0.1

    obs = np.array([2, 1, 1, 1])
    info = {
        "dp": 0.1,
        "dv": -0.1,
    }
    assert policy(obs, info) == -0.1

    obs = np.array([1, 1, 3, 1])
    info = {
        "dp": -0.1,
        "dv": 0.1,
    }
    assert policy(obs, info) == -0.1

    obs = np.array([1, 1, 1, 1])
    info = {
        "dp": -0.1,
        "dv": -0.1,
    }
    assert policy(obs, info) == 0.1


def test_po_obs():
    env = gym.make("Pendulum-v0")
    policy = PerturbObservePolicy(
        env=env,
        v_step=0.1,
        dp_index=0,
        dv_index=1,
    )

    obs = np.array([1, 1, 1, 1])
    info = {
        "dp": -0.1,
        "dv": 0.1,
    }
    assert policy([obs], info) == 0.1

    obs = np.array([2, -1, 1, 1])
    info = {
        "dp": 0.1,
        "dv": 0.1,
    }
    assert policy([obs], info) == -0.1

    obs = np.array([-1, 1, 3, 1])
    info = {
        "dp": 0.1,
        "dv": 0.1,
    }
    assert policy([obs], info) == -0.1

    obs = np.array([-1, -1, 1, 1])
    info = {
        "dp": 0.1,
        "dv": -0.1,
    }
    assert policy([obs], info) == 0.1


def test_po_info_noise():
    env = gym.make("Pendulum-v0")
    noise = GaussianNoise(0.0, 0.1)
    policy = PerturbObservePolicy(
        env=env,
        v_step=0.1,
        dp_index="dp",
        dv_index="dv",
        noise=noise,
        epsilon=1.0,
    )

    obs = np.array([1, 1, 1, 1])
    info = {
        "dp": 0.1,
        "dv": 0.1,
    }
    assert policy(obs, info) != 0.1

    obs = np.array([2, 1, 1, 1])
    info = {
        "dp": 0.1,
        "dv": -0.1,
    }
    assert policy(obs, info) != -0.1

    obs = np.array([1, 1, 3, 1])
    info = {
        "dp": -0.1,
        "dv": 0.1,
    }
    assert policy(obs, info) != -0.1

    obs = np.array([1, 1, 1, 1])
    info = {
        "dp": -0.1,
        "dv": -0.1,
    }
    assert policy(obs, info) != 0.1


def test_ddpg_box_deterministic():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Identity()

        def forward(self, x):
            return self.net(x)

    env = gym.make("Pendulum-v0")
    net = Net()
    policy = DDPGPolicy(
        env=env,
        net=net,
        noise=None,
        epsilon=0.0,
    )
    obs = np.array([1])
    assert isinstance(policy([obs]), np.ndarray)
    assert policy([obs]).shape == (1,)
    assert (policy([obs])[0] - 2) < 1e-6

    obs = np.array([-1])
    assert (policy([obs])[0] + 2) < 1e-6

    obs = np.array([0])
    assert policy([obs])[0] < 1e-6


def test_ddpg_box_stochastic():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Identity()

        def forward(self, x):
            return self.net(x)

    env = gym.make("Pendulum-v0")
    net = Net()
    noise = GaussianNoise(0.0, 0.1)
    policy = DDPGPolicy(
        net=net,
        env=env,
        noise=noise,
        epsilon=1.0,
    )
    obs = np.array([0])
    assert isinstance(policy([obs]), np.ndarray)
    assert policy([obs]).shape == (1,)
    assert abs(policy([obs])[0]) > 1e-6  # with noise

    policy.epsilon = 0.0
    assert policy([obs])[0] < 1e-6  # with noise


def test_mpp_policy_deterministic():
    pv_params_path = os.path.join("parameters", "01_pvarray.json")
    weather_train_path = os.path.join("data", "weather_sim.csv")
    pvarray_ckp_path = os.path.join("data", "01_pvarray_iv.json")

    engine = matlab.engine.connect_matlab()
    env = PVEnv.from_file(
        pv_params_path=pv_params_path,
        weather_path=weather_train_path,
        pvarray_ckp_path=pvarray_ckp_path,
        engine=engine,
        states=["g", "t", "v"],
        reward_fn=RewardPowerDeltaPower(norm=True),
    )
    policy = MPPPolicy(
        env=env,
        g_index=0,
        t_index=1,
        v_index=2,
    )

    # True Vmpp is ~ 26.31
    obs = np.array([1000, 25, 0])
    action = policy([obs])
    assert env.action_space.high[0] == action

    # True Vmpp is ~ 26.31
    obs = np.array([1000, 25, 24])
    action = policy([obs])
    assert abs(action - 2.31) < 1
