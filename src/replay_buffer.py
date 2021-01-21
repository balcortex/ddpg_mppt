import collections
import pickle
from collections import namedtuple
from typing import Deque, List, Tuple

import numpy as np

Experience = namedtuple("Experience", ["obs", "action", "reward", "done", "new_obs"])
ExperienceBatch = namedtuple(
    "ExperienceBatch",
    ["observations", "actions", "rewards", "dones", "new_observations"],
)


class ReplayBuffer:
    """
    Buffer to save the interactions of the agent with the environment

    Parameters:
        capacity: buffers' capacity to store a experience tuple

    Returns:
    Numpy arrays
    """

    def __init__(self, capacity: int):
        assert isinstance(capacity, int)
        self.buffer: Deque[Experience] = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

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
            obs=disc_exp.state,
            action=disc_exp.action,
            reward=disc_exp.discounted_reward,
            done=True if disc_exp.last_state is None else False,
            new_obs=disc_exp.last_state,
        )
        buffer.append(exp)

    print(buffer.buffer)

    buffer.save("buffer.pkl")

    del buffer

    buffer = ReplayBuffer.from_file("buffer.pkl")

    print()
    print(buffer.buffer)

    os.remove("buffer.pkl")
