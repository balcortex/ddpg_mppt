from typing import Optional, Union, Any, Dict
from src.noise import Noise

import gym
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from src.pv_env_dcdc import PVEnv
from src.schedule import Schedule


class BasePolicy(ABC):
    "Base class of Policy"

    def __init__(
        self,
        env: Union[gym.Env, PVEnv],
        noise: Optional[Noise],
        schedule: Schedule,
        decrease_noise: bool,
    ):
        self.env = env
        self.noise = noise
        self.schedule = schedule
        self.decrease_noise = decrease_noise

        self.low = self.env.action_space.low[0]
        self.high = self.env.action_space.high[0]

    @abstractmethod
    def __call__(
        self,
        obs: np.ndarray,
        info: Optional[Dict[Any, Any]],
    ) -> Union[float, np.ndarray]:
        pass

    def reset(self) -> None:
        "Reset the schedule to output its initial epsilon value"
        self.schedule.reset()

    def _unscale_actions(
        self,
        scled_actions: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        "Unscale the actions to match the environment limits"
        return scled_actions

    def _add_noise(
        self,
        actions: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        "Add noise to the actions"
        if not self.noise:
            return actions

        if self.schedule:
            epsilon = self.schedule()
            self.schedule.step()
        else:
            epsilon = 0.0

        if epsilon > np.random.rand():
            noise = self.noise.sample()
            if self.decrease_noise:
                noise *= epsilon
            actions += noise

        return actions

    def _clamp_actions(
        self,
        actions: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        "Clamp the actions between environment limits"
        if isinstance(actions, np.ndarray):
            return actions.clip(self.low, self.high)
        return actions.clamp(self.low, self.high)

    def _process_actions(
        self,
        scled_actions: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        "Unscale actions, add noise and clamp them"
        actions = self._unscale_actions(scled_actions)
        actions = self._add_noise(actions)
        actions = self._clamp_actions(actions)
        return actions


class RandomPolicy(BasePolicy):
    """
    Policy that returns a random action depending on the environment

    Parameters:
        env: gym environment
    """

    def __init__(self, env: gym.Env):
        self.env = env

    def __call__(
        self,
        obs: np.ndarray,
        info: Optional[Dict[Any, Any]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Get an action according to the observation

        Paramters:
            obs: observations from the environment
            info: additional info passed to the policy (not used)
        """
        return self.env.action_space.sample()

    def __str__(self):
        return "RandomPolicy"


class DDPGPolicy(BasePolicy):
    """
    Deep Deterministic Policy Gradient - Policy

    Parameters:
        env: gym environment used to scale the actions
        net: actor network, convert observations into actions
        noise: sample noise from this object to perform exploration
        schedule: class that keep track of epsilon
        decrease_noise: whether to multiply the noise by epsilon
        device: device where the calculations are performed ['cpu', 'cuda']
    """

    def __init__(
        self,
        env: gym.Env,
        net: nn.Module,
        noise: Optional[Noise] = None,
        schedule: Optional[Schedule] = None,
        decrease_noise: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(
            env=env,
            noise=noise,
            schedule=schedule,
            decrease_noise=decrease_noise,
        )
        self.net = net
        self.device = device

    @torch.no_grad()
    def __call__(
        self,
        obs: np.ndarray,
        info: Optional[Dict[Any, Any]] = None,
    ) -> np.ndarray:
        """
        Get an action according to the observation

        Parameters:
            obs: observations from the environment
            info: additional info passed to the policy (not used)
        """
        obs_v = torch.tensor(obs, dtype=torch.float32)
        actions = self.net(obs_v)
        actions = self._process_actions(actions)
        return actions.cpu().numpy()[0]

    def _unscale_actions(
        self,
        scled_actions: torch.Tensor,
    ) -> torch.Tensor:
        "Unscale the actions to match the environment limits"
        actions = self.low + (scled_actions + 1) * (self.high - self.low) / 2
        return actions

    def __str__(self):
        return "DDPGPolicy"


class PerturbObservePolicy(BasePolicy):
    """
    Perturb & Observe algorithm

    Parameters:
        env: gym environment used to scale the actions
        v_step: magnitude of the perturbation
        dv_index: index of the dv state (v - v_old). If the index is an int,
            the variable must be passed on the obs parameter, if is a str, it is
            localized in the info parameter
        dp_index: index of the dp state (p - p_old)
        noise: sample noise from this object to perform exploration
        schedule: class that keep track of epsilon
        decrease_noise: whether to multiply the noise by epsilon
    """

    def __init__(
        self,
        env: gym.Env,
        v_step: float,
        dv_index: Union[int, str] = "dv",
        dp_index: Union[int, str] = "dp",
        noise: Optional[Noise] = None,
        schedule: Optional[Schedule] = None,
        decrease_noise: bool = False,
    ):
        super().__init__(
            env=env,
            noise=noise,
            schedule=schedule,
            decrease_noise=decrease_noise,
        )
        self.v_step = v_step
        self.dv_index = dv_index
        self.dp_index = dp_index

        self.dp_in_obs = False
        self.dv_in_obs = False

        if isinstance(self.dp_index, int):
            self.dp_in_obs = True
        if isinstance(self.dv_index, int):
            self.dv_in_obs = True

    def __call__(
        self,
        obs: np.ndarray,
        info: Optional[Dict[Any, Any]] = None,
    ) -> np.ndarray:
        """
        Get an action according to the observation

        Parameters:
            obs: observations from the environment
            info: additional info passed to the policy
        """
        if self.dp_in_obs:
            delta_p = obs[0][self.dp_index]
        else:
            delta_p = info.get(self.dp_index, 0.0)

        if self.dv_in_obs:
            delta_v = obs[0][self.dv_index]
        else:
            delta_v = info.get(self.dv_index, 0.0)

        if delta_p >= 0:
            if delta_v >= 0:
                action = self.v_step
            else:
                action = -self.v_step
        else:
            if delta_v >= 0:
                action = -self.v_step
            else:
                action = self.v_step

        action = np.array([action])
        action = self._process_actions(action)

        return action

    def __str__(self):
        return "POPolicy"


class PerturbObservePolicyDCDC(PerturbObservePolicy):
    def __call__(
        self,
        obs: np.ndarray,
        info: Optional[Dict[Any, Any]] = None,
    ) -> np.ndarray:
        if self.dp_in_obs:
            delta_p = obs[0][self.dp_index]
        else:
            delta_p = info.get(self.dp_index, 0.0)

        if self.dv_in_obs:
            delta_v = obs[0][self.dv_index]
        else:
            delta_v = info.get(self.dv_index, 0.0)

        if delta_p >= 0:
            if delta_v > 0:
                action = -self.v_step
            else:
                action = self.v_step
        else:
            if delta_v >= 0:
                action = self.v_step
            else:
                action = -self.v_step

        action = np.array([action])
        action = self._process_actions(action)

        return action

    def __str__(self):
        return "POPolicy"


# class MPPPolicy(BasePolicy):
#     """
#     Policy that calculates the true VMPP of a pv array under G and T

#     Parameters:
#         env: pv array environment
#         g_index: index of the irradiance state. If the index is an int,
#             the variable must be passed on the obs parameter, if is a str, it is
#             localized in the info parameter
#         t_index: index of the temperature state
#         v_index: index of the voltage state
#         noise: sample noise from this object to perform exploration
#         schedule: class that keep track of epsilon
#         decrease_noise: whether to multiply the noise by epsilon
#     """

#     def __init__(
#         self,
#         env: PVEnv,
#         g_index: Union[int, str],
#         t_index: Union[int, str],
#         v_index: Union[int, str],
#         noise: Optional[Noise] = None,
#         schedule: Optional[Schedule] = None,
#         decrease_noise: bool = False,
#     ):
#         super().__init__(
#             env=env,
#             noise=noise,
#             schedule=schedule,
#             decrease_noise=decrease_noise,
#         )
#         self.g_index = g_index
#         self.t_index = t_index
#         self.v_index = v_index

#     def __call__(
#         self,
#         obs: np.ndarray,
#         info: Optional[Dict[Any, Any]] = None,
#     ) -> np.ndarray:
#         """
#         Get an action according to the observation

#         Parameters:
#             obs: observations from the environment
#             info: additional info passed to the policy
#         """
#         if isinstance(self.g_index, int):
#             g = obs[0][self.g_index]
#         elif isinstance(self.g_index, str):
#             if info == {}:
#                 g = self.env.history.g[-1]
#             else:
#                 g = info[self.g_index]
#         else:
#             raise ValueError(f"g_index must be str or int")

#         if isinstance(self.t_index, int):
#             t = obs[0][self.t_index]
#         elif isinstance(self.t_index, str):
#             if info == {}:
#                 t = self.env.history.t[-1]
#             else:
#                 t = info[self.t_index]
#         else:
#             raise ValueError(f"t_index must be str or int")

#         if isinstance(self.v_index, int):
#             v = obs[0][self.v_index]
#         elif isinstance(self.v_index, str):
#             if info == {}:
#                 v = self.env.history.v[-1]
#             else:
#                 v = info[self.v_index]
#         else:
#             raise ValueError(f"v_index must be str or int")

#         _, v_mpp, _ = self.env.pvarray.get_true_mpp(irradiance=[g], cell_temp=[t])

#         action = np.array([v_mpp - v])
#         action = self._process_actions(action)
#         return action


class MPPPolicyDCDC(BasePolicy):
    """
    Policy that calculates the true VMPP of a pv array under G and T

    Parameters:
        env: pv array environment
        g_index: index of the irradiance state. If the index is an int,
            the variable must be passed on the obs parameter, if is a str, it is
            localized in the info parameter
        t_index: index of the temperature state
        v_index: index of the voltage state
        noise: sample noise from this object to perform exploration
        schedule: class that keep track of epsilon
        decrease_noise: whether to multiply the noise by epsilon
    """

    def __init__(
        self,
        env: PVEnv,
        g_index: Union[int, str],
        t_index: Union[int, str],
        dc_index: Union[int, str],
        noise: Optional[Noise] = None,
        schedule: Optional[Schedule] = None,
        decrease_noise: bool = False,
    ):
        super().__init__(
            env=env,
            noise=noise,
            schedule=schedule,
            decrease_noise=decrease_noise,
        )
        self.g_index = g_index
        self.t_index = t_index
        self.dc_index = dc_index

    def __call__(
        self,
        obs: np.ndarray,
        info: Optional[Dict[Any, Any]] = None,
    ) -> np.ndarray:
        """
        Get an action according to the observation

        Parameters:
            obs: observations from the environment
            info: additional info passed to the policy
        """
        if isinstance(self.g_index, int):
            g = obs[0][self.g_index]
        elif isinstance(self.g_index, str):
            if info == {}:
                g = self.env.history.n_g[-1]
            else:
                g = info[self.g_index]
        else:
            raise ValueError(f"g_index must be str or int")

        if isinstance(self.t_index, int):
            t = obs[0][self.t_index]
        elif isinstance(self.t_index, str):
            if info == {}:
                t = self.env.history.n_amb_t[-1]
            else:
                t = info[self.t_index]
        else:
            raise ValueError(f"t_index must be str or int")

        if isinstance(self.dc_index, int):
            dc = obs[0][self.dc_index]
        elif isinstance(self.dc_index, str):
            if info == {}:
                dc = self.env.history.duty_cycle[-1]
            else:
                dc = info[self.dc_index]
        else:
            raise ValueError(f"dc_index must be str or int")

        result = self.env.pvarray.get_true_mpp(irradiance=[g], ambient_temp=[t])
        dc_optim = result.duty_cycle

        action = np.array([dc_optim - dc])
        action = self._process_actions(action)
        return action


# class MPPPolicyDCDC(BasePolicy):
#     """
#     Policy that calculates the true VMPP of a pv array under G and T

#     Parameters:
#         env: pv array environment
#         g_index: index of the irradiance state. If the index is an int,
#             the variable must be passed on the obs parameter, if is a str, it is
#             localized in the info parameter
#         t_index: index of the temperature state
#         v_index: index of the voltage state
#         noise: sample noise from this object to perform exploration
#         schedule: class that keep track of epsilon
#         decrease_noise: whether to multiply the noise by epsilon
#     """

#     def __init__(
#         self,
#         env: PVEnv,
#         # g_index: Union[int, str],
#         # t_index: Union[int, str],
#         # dc_index: Union[int, str],
#         noise: Optional[Noise] = None,
#         schedule: Optional[Schedule] = None,
#         decrease_noise: bool = False,
#     ):
#         super().__init__(
#             env=env,
#             noise=noise,
#             schedule=schedule,
#             decrease_noise=decrease_noise,
#         )
#         # self.g_index = g_index
#         # self.t_index = t_index
#         # self.dc_index = dc_index

#     def __call__(
#         self,
#         obs: np.ndarray,
#         info: Optional[Dict[Any, Any]] = None,
#     ) -> np.ndarray:
#         """
#         Get an action according to the observation

#         Parameters:
#             obs: observations from the environment
#             info: additional info passed to the policy
#         """
#         d = self.env.history.duty_cycle[-1]
#         d_opt_next = self.env.history.d_opt_nxt[-1]

#         action = np.array([d_opt_next - d])
#         action = self._process_actions(action)
#         return action