import collections
import pickle
from src.experience import ExperienceSource

# from collections import namedtuple
from typing import Deque, List, Tuple, NamedTuple, Union, Optional

import numpy as np

# Experience = namedtuple("Experience", ["obs", "action", "reward", "done", "new_obs"])
# ExperienceBatch = namedtuple(
#     "ExperienceBatch",
#     ["observations", "actions", "rewards", "dones", "new_observations"],
# )


class Experience(NamedTuple):
    state: Union[float, int]
    action: Union[float, int]
    reward: Union[float, int]
    done: bool
    last_state: Optional[Union[float, int]]


class ExperienceDiscounted(NamedTuple):
    state: Union[float, int]
    action: Union[float, int]
    reward: Union[float, int]
    last_state: Optional[Union[float, int]]
    discounted_reward: Union[float, int]
    steps: int


class ExperienceBatch(NamedTuple):
    state: np.array
    action: np.array
    reward: np.array
    done: np.array
    last_state: np.array


class ReplayBuffer:
    """
    Buffer to save the interactions of the agent with the environment

    Parameters:
        capacity: buffers' capacity to store a experience tuple

    Returns:
    Numpy arrays
    """

    def __init__(self, capacity: int, name: str = ""):
        assert isinstance(capacity, int)
        self.name = name
        self.buffer: Deque[Experience] = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer{self.name}"

    def append(self, experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> ExperienceBatch:
        assert (
            len(self.buffer) >= batch_size
        ), f"Cannot sample {batch_size} elements from buffer of length {len(self.buffer)}"
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )

        return ExperienceBatch(
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
        )

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.buffer, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_file(cls, filename: str) -> None:
        buffer = cls(1)
        with open(filename, "rb") as f:
            deq = pickle.load(f)
        buffer.buffer = deq

        return buffer


# def merge_batches(a: ExperienceBatch, b: ExperienceBatch):
#     return ExperienceBatch(*(np.append(a_, b_) for a_, b_ in zip(a, b)))


if __name__ == "__main__":
    import gym
    import os
    from src.experience import ExperienceSourceDiscountedSteps
    from src.policies import RandomPolicy

    env = gym.make("Pendulum-v0")
    rand_policy = RandomPolicy(env)
    exp_source = ExperienceSourceDiscountedSteps(
        policy=rand_policy,
        gamma=0.99,
        n_steps=1,
        steps=4,
    )
    buffer = ReplayBuffer(1_000)

    for disc_exp in next(exp_source):
        exp = Experience(
            disc_exp.state,
            disc_exp.action,
            disc_exp.discounted_reward,
            True if disc_exp.last_state is None else False,
            disc_exp.last_state,
        )
        buffer.append(exp)

    print(buffer)
