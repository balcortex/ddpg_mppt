import collections
import os
from dataclasses import dataclass, field
from math import atan2
from typing import List, Optional
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.logger import logger
from src.pv_array_dcdc import PVArray
from src.utils import read_weather_csv

G_MAX = 1200
T_MAX = 60

StepResult = collections.namedtuple(
    "StepResult", field_names=["obs", "reward", "done", "info"]
)


@dataclass
class History:
    g: list = field(default_factory=list)
    amb_t: list = field(default_factory=list)
    cell_t: list = field(default_factory=list)
    p: list = field(default_factory=list)
    v: list = field(default_factory=list)
    i: list = field(default_factory=list)
    duty_cycle: list = field(default_factory=list)
    dp: list = field(default_factory=list)
    dv: list = field(default_factory=list)
    di: list = field(default_factory=list)
    g_norm: list = field(default_factory=list)
    amb_t_norm: list = field(default_factory=list)
    cell_t_norm: list = field(default_factory=list)
    p_norm: list = field(default_factory=list)
    v_norm: list = field(default_factory=list)
    i_norm: list = field(default_factory=list)
    dp_norm: list = field(default_factory=list)
    dv_norm: list = field(default_factory=list)
    deg: list = field(default_factory=list)


class PVEnvBase(gym.Env):
    "PV Environment abstract class for solving the MPPT by reinforcement learning"
    # metadata = {"render.modes": ["human"]}
    # spec = gym.envs.registration.EnvSpec("PVEnv-v0")

    def reset(self):
        raise NotImplementedError

    def step(self, action) -> np.ndarray:
        raise NotImplementedError

    def render(self, vars: List) -> None:
        raise NotImplementedError

    def _get_observation_space(self) -> gym.Space:
        raise NotImplementedError

    def _get_action_space(self) -> gym.Space:
        raise NotImplementedError

    def _get_delta_v(self, action: float) -> float:
        raise NotImplementedError

    # @classmethod
    # def from_file(
    #     cls,
    #     pv_params_path: str,
    #     weather_path: str,
    #     pvarray_ckp_path: str,
    #     engine,
    #     **kwargs,
    # ):
    #     pvarray = PVArray.from_json(
    #         pv_params_path, ckp_path=pvarray_ckp_path, engine=engine
    #     )
    #     weather = read_weather_csv(weather_path)
    #     return cls(pvarray, weather, **kwargs)


class PVEnv(PVEnvBase):
    """
    PV Continuos Environment for solving the MPPT by reinforcement learning

    Parameters:
        - pvarray: the pvarray object
        - weather_df: a pandas dataframe object containing weather readings
        - states: list of states to return as observations
        - reward_fn: function that calculates the reward
        - seed: for reproducibility
    """

    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        reward_fn: callable,
        states: List[str] = ["p_norm", "v_norm", "dp", "dv", "duty_cycle"],
        seed: Optional[int] = None,
        dc0: Optional[float] = None,
        max_steps: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:

        self.pvarray = pvarray
        self.weather = weather_df
        self.states = states
        self.reward_fn = reward_fn
        self.dc0 = dc0
        self.max_steps = min(max_steps or len(weather_df) - 1, len(weather_df) - 1)
        if seed:
            np.random.seed(seed)
        self.deterministic = deterministic

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    def reset(self) -> np.ndarray:
        self.history = History()
        self.step_counter = 0
        self.done = False

        if self.deterministic:
            self.step_idx = 0
        else:
            self.step_idx = random.randint(0, len(self.weather) - 1)

        dc = np.random.rand() if self.dc0 == None else self.dc0

        return self._store_step(dc)

    def step(self, action: float) -> StepResult:
        if self.done:
            raise ValueError("The episode is done")

        self.step_idx += 1
        self.step_counter += 1

        if self.step_idx == len(self.weather):
            self.step_idx = 0

        delta_dc = self._get_delta_dc(action)
        dc = np.clip(self.dc + delta_dc, 0.0, 1.0)
        obs = self._store_step(dc)
        reward = self.reward_fn(self.history)

        # if self.history.p[-1] < 0 or self.history.v[-1] < 1:
        #     self.done = True
        if self.step_counter >= self.max_steps:
            self.done = True

        info = {
            "step_idx": self.step_idx,
            "steps": self.step_counter,
            "dp": self.history.dp[-1],
            "dv": self.history.dv[-1],
            "g": self.history.g[-1],
            "amb_t": self.history.amb_t[-1],
            "cell_t": self.history.cell_t[-1],
            "v": self.history.v[-1],
            "duty_cycle": self.history.duty_cycle[-1],
        }

        return StepResult(
            obs,
            reward,
            self.done,
            info,
        )

    def render(self, vars: List[str]) -> None:
        for var in vars:
            plt.plot(getattr(self.history, var), label=var)
            plt.legend()
            plt.show()

    def render_vs_true(
        self,
        label: str = "RL",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> float:
        p_real, v_real, *_ = self.pvarray.get_true_mpp(
            self.history.g, self.history.amb_t
        )
        eff = PVArray.mppt_eff(p_real, self.history.p)

        plt.plot(p_real, label="P Max")
        plt.plot(self.history.p, label=f"P {label}")
        plt.legend()
        if show:
            plt.show()
        if save_path:
            path = fname = save_path + "_p_" + f"{eff:.2f}.png"
            plt.savefig(path)
            logger.info(f"Saved to {path}")

        plt.clf()

        plt.plot(v_real, label="Vmpp")
        plt.plot(self.history.v, label=f"V {label}")
        plt.legend()
        if show:
            plt.show()
        if save_path:
            plt.savefig(fname=save_path + "_v.png")
            logger.info(f'Saved to {save_path + "_v.png"}')

        plt.clf()

        logger.info(f"{label} Efficiency={eff}")

        return eff

    def _add_history(self, p, v, i, g, amb_t, cell_t, dc) -> None:
        self.history.p.append(p)
        self.history.v.append(v)
        self.history.i.append(i)
        self.history.g.append(g)
        self.history.amb_t.append(amb_t)
        self.history.cell_t.append(cell_t)
        self.history.duty_cycle.append(dc)
        self.history.p_norm.append(p / self.pvarray.pmax)
        self.history.v_norm.append(v / self.pvarray.voc)
        self.history.i_norm.append(i / self.pvarray.isc)
        self.history.g_norm.append(g / G_MAX)
        self.history.amb_t_norm.append(amb_t / T_MAX)
        self.history.cell_t_norm.append(cell_t / T_MAX)

        if len(self.history.p) < 2:
            self.history.dp.append(0.0)
            self.history.dv.append(0.0)
            self.history.di.append(0.0)
            self.history.deg.append(0.0)
            self.history.dp_norm.append(0.0)
            self.history.dv_norm.append(0.0)
        else:
            self.history.dp.append(self.history.p[-1] - self.history.p[-2])
            self.history.dv.append(self.history.v[-1] - self.history.v[-2])
            self.history.di.append(self.history.i[-1] - self.history.i[-2])
            self.history.dp_norm.append(
                self.history.p_norm[-1] - self.history.p_norm[-2]
            )
            self.history.dv_norm.append(
                self.history.v_norm[-1] - self.history.v_norm[-2]
            )
            self.history.deg.append(
                atan2(self.history.di[-1], self.history.dv[-1])
                + atan2(self.history.i[-1], self.history.v[-1])
            )

    def _get_delta_dc(self, action: float) -> float:
        if isinstance(action, (list, np.ndarray)):
            action = action[0]
        return action

    def _get_observation_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=np.array([-np.inf] * len(self.states)),
            high=np.array([-np.inf] * len(self.states)),
            shape=(len(self.states),),
            dtype=np.float32,
        )

    def _get_action_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def _store_step(self, dc: float) -> np.ndarray:
        g, t = self.weather[["Irradiance", "Temperature"]].iloc[self.step_idx]
        p, v, i, self.dc, g, t, cell_t = self.pvarray.simulate(dc, g, t)
        p = max(0, p)
        self._add_history(p, v, i, g, t, cell_t, dc)

        # getattr(handler.request, 'GET') is the same as handler.request.GET
        return np.array([getattr(self.history, state)[-1] for state in self.states])


class PVEnvDiscrete(PVEnv):
    """
    PV Discrete Environment for solving the MPPT by reinforcement learning

    Parameters:
        - pvarray: the pvarray object
        - weather_df: a pandas dataframe object containing weather readings
        - states: list of states to return as observations
        - reward_fn: function that calculates the reward
        - seed: for reproducibility
    """

    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        reward_fn: callable,
        actions: List[float],
        states: List[str] = ["p_norm", "v_norm", "dp", "dv", "duty_cycle"],
        seed: Optional[int] = None,
        dc0: Optional[float] = None,
        max_steps: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        self.actions = actions
        super().__init__(
            pvarray,
            weather_df,
            reward_fn,
            states,
            seed,
            dc0,
            max_steps,
            deterministic,
        )

    def _get_delta_dc(self, action: int) -> float:
        return self.actions[action]

    def _get_action_space(self) -> gym.Space:
        return gym.spaces.Discrete(len(self.actions))


if __name__ == "__main__":
    import matlab.engine
    from src.utils import read_weather_csv
    from src.reward import RewardPowerDeltaPower

    pv_params_path = os.path.join("parameters", "01_pvarray.json")
    pvarray_ckp_path = os.path.join("data", "02_pvarray_dcdc.json")

    engine = matlab.engine.connect_matlab()
    pvarray = PVArray.from_json(
        path=pv_params_path,
        ckp_path=pvarray_ckp_path,
        engine=engine,
        f_precision=3,
    )
    weather_df = read_weather_csv(os.path.join("data", "weather_sim.csv"))

    env = PVEnv(
        pvarray=pvarray,
        weather_df=weather_df,
        states=["p", "v", "dp", "dv", "duty_cycle", "g", "amb_t", "cell_t"],
        reward_fn=RewardPowerDeltaPower(norm=True),
        seed=None,
        dc0=0.0,
        max_steps=None,
        deterministic=True,
    )

    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # print(f"{obs=}")

        if done:
            break

    env.render(["p", "v", "duty_cycle"])
    env.render(["g", "amb_t", "cell_t"])

    env.render_vs_true(label="Random")
