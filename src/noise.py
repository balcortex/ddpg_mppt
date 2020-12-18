import numpy as np
from abc import ABC, abstractmethod


class Noise(ABC):
    "Base class for noise"

    @abstractmethod
    def sample(self) -> float:
        pass

    @abstractmethod
    def reset(self) -> float:
        pass


class OUNoise(Noise):
    """
    Ornstein–Uhlenbeck process

    Parameters:
        mean: mean of the noise
        std: standard deviation of the noise
    """

    def __init__(self, mean: float, std: float, theta: float = 0.1, dt: float = 0.01):
        self.mean = mean
        self.theta = theta
        self.std = std
        self.dt = dt
        self.reset()

    def __repr__(self) -> str:
        return f"OUNoise, mean={self.mean}, std={self.std}, theta={self.theta}, dt={self.dt}"

    def reset(self) -> float:
        "Reset the state of the noise"
        self.state = self.mean
        return self.state

    def sample(self) -> float:
        "Sample from the noise"
        x = self.state
        dx = (
            self.theta * (self.mean - x) * self.dt
            + self.std * np.sqrt(self.dt) * np.random.randn()
        )
        self.state = x + dx
        return self.state


class GaussianNoise(Noise):
    """
    Noise sampled from a normal distribution

    Parameters:
        mean: mean of the noise
        std: standard deviation of the noise
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __repr__(self) -> str:
        return f"GaussianNoise, mean={self.mean}, std={self.std}"

    def reset(self) -> float:
        return np.random.normal(self.mean, self.std)

    def sample(self) -> float:
        return np.random.normal(self.mean, self.std)


if __name__ == "__main__":
    # ou = OUNoise(mean=0.0, std=0.1, theta=0.1, dt=1)
    # states = []
    # for i in range(1000000):
    #     states.append(ou.sample())
    # import matplotlib.pyplot as plt

    # plt.plot(states)
    # plt.show()

    noise = GaussianNoise(0, 0.5)
    states = []
    for i in range(800):
        states.append(noise.sample())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()