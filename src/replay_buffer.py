import collections
import numpy as np
from typing import Tuple, Deque, List
from collections import namedtuple

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
