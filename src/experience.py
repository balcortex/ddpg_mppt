import collections
import gym
from src.policies import BasePolicy
import numpy as np
import datetime
import os
from typing import Dict, Any
import time

from src.logger import logger

DATAFRAME_BASEPATH = os.path.join("data", "dataframes")


Experience = collections.namedtuple(
    "Experience", ["state", "action", "reward", "last_state"]
)
ExperienceDiscounted = collections.namedtuple(
    "Experience",
    ["state", "action", "reward", "last_state", "discounted_reward", "steps"],
)


class ExperienceSource:
    def __init__(
        self,
        policy: BasePolicy,
        render: bool = False,
        pvenv_kwargs: Dict[str, Any] = {},
    ):
        self.policy = policy
        self.env = policy.env
        self.render = render
        self.save_dataframe = pvenv_kwargs.get("save_dataframe", False)
        self.include_true_mpp = pvenv_kwargs.get("include_true_mpp", False)
        self.policy_name = pvenv_kwargs.get("policy_name", None)

        if self.policy_name:
            self.policy_name = self.policy_name + "_" + self.env.pvarray.model_name

        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        return self.play_step()

    def reset(self) -> None:
        "Reset the step and episode counter"
        self.obs = self.env.reset()
        self.done = False
        self.info = {}
        self.step_counter = 0
        self.episode_counter = 0
        self.episode_reward = 0.0
        self.episode_rewards = []

    def play_step(self):
        self.step_counter += 1
        if self.done:
            self.obs = self.env.reset()
            self.info = {}
            self.done = False
        obs = self.obs
        action = self.policy(obs=[obs], info=self.info)
        new_obs, reward, done, self.info = self.env.step(action)
        self.episode_reward += reward
        if self.render:
            self.env.render()
        if done:
            self.episode_counter += 1
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0
            self.done = True
            if self.save_dataframe:
                self.save()
            return Experience(state=obs, action=action, reward=reward, last_state=None)
        self.obs = new_obs
        return Experience(state=obs, action=action, reward=reward, last_state=new_obs)

    def play_episode(self):
        ep_history = []
        self.obs = self.env.reset()

        while True:
            experience = self.play_step()
            ep_history.append(experience)

            if experience.last_state is None:
                return ep_history

    def play_episodes(self, episodes):
        return [self.play_episode() for _ in range(episodes)]

    @property
    def last_episode_reward(self) -> float:
        "Return the cumulative reward of the last episode"
        if self.episode_rewards == []:
            return np.nan
        return self.episode_rewards[-1]

    def mean_rewards(self, episodes: int = 10):
        "Return the mean reward of the last episodes"
        if self.episode_rewards == []:
            return np.nan
        episodes = min(episodes, len(self.episode_rewards))
        return sum(self.episode_rewards[-episodes:]) / episodes

    def save(self) -> None:
        name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        policy = self.policy_name or str(self.policy)
        name += f"_{policy}.csv"
        path = os.path.join(DATAFRAME_BASEPATH, name)
        time.sleep(2)

        logger.info(f"Saving dataframe to {path}")
        self.policy.env.save_dataframe(
            path=path, include_true_mpp=self.include_true_mpp
        )
        logger.info(f"Saved to {path}")


class ExperienceSourceEpisodes(ExperienceSource):
    def __init__(
        self,
        policy: BasePolicy,
        episodes: int,
        render: bool = False,
        pvenv_kwargs: Dict[str, str] = {},
    ):
        super().__init__(policy=policy, render=render, pvenv_kwargs=pvenv_kwargs)

        self.max_episodes = episodes

    def __next__(self):
        return self.play_episodes(self.max_episodes)


class ExperienceSourceDiscounted(ExperienceSource):
    def __init__(
        self,
        policy: BasePolicy,
        gamma: float,
        n_steps: int,
        render: bool = False,
        pvenv_kwargs: Dict[str, str] = {},
    ):
        super().__init__(policy=policy, render=render, pvenv_kwargs=pvenv_kwargs)

        self.gamma = gamma
        self.n_steps = n_steps

    def __next__(self):
        return self.play_n_steps()

    def play_n_steps(self):
        history = []
        discounted_reward = 0.0
        reward = 0.0

        for step_idx in range(self.n_steps):
            exp = self.play_step()
            reward += exp.reward
            discounted_reward += exp.reward * self.gamma ** (step_idx)
            history.append(exp)

            if exp.last_state is None:
                break

        return ExperienceDiscounted(
            state=history[0].state,
            action=history[0].action,
            last_state=history[-1].last_state,
            reward=reward,
            discounted_reward=discounted_reward,
            steps=step_idx + 1,
        )

    def play_episode(self):
        ep_history = []
        self.obs = self.env.reset()

        while True:
            experience = self.play_n_steps()
            ep_history.append(experience)

            if experience.last_state is None:
                return ep_history


class ExperienceSourceDiscountedSteps(ExperienceSourceDiscounted):
    def __init__(
        self,
        policy: BasePolicy,
        gamma: float,
        n_steps: int,
        steps: int,
        render: bool = False,
        pvenv_kwargs: Dict[str, str] = {},
    ):
        super().__init__(
            policy=policy,
            gamma=gamma,
            n_steps=n_steps,
            render=render,
            pvenv_kwargs=pvenv_kwargs,
        )

        self.steps = steps

    def __next__(self):
        return [self.play_n_steps() for _ in range(self.steps)]


class ExperienceSourceDiscountedEpisodes(ExperienceSourceDiscounted):
    def __init__(
        self,
        policy: BasePolicy,
        gamma: float,
        n_steps: int,
        episodes: int,
        render: bool = False,
        pvenv_kwargs: Dict[str, str] = {},
    ):
        super().__init__(
            policy=policy,
            gamma=gamma,
            n_steps=n_steps,
            render=render,
            pvenv_kwargs=pvenv_kwargs,
        )

        self.max_episodes = episodes

    def __next__(self):
        return self.play_episodes(self.max_episodes)


if __name__ == "__main__":
    import gym
    from src.policies import RandomPolicy

    env = gym.make("Pendulum-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSource(
        policy=policy,
    )