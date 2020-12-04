from abc import ABC, abstractclassmethod
from typing import Optional


class Schedule:
    @abstractclassmethod
    def __call__(self) -> float:
        pass

    @abstractclassmethod
    def step(self, steps: int) -> float:
        pass

    @abstractclassmethod
    def set_current_step(self, step: int) -> float:
        pass

    @abstractclassmethod
    def reset(self) -> float:
        pass


class ConstantSchedule(Schedule):
    """
    Constant schedule for epsilon

    Parameters:
        epsilon: the value returned by the schedule
    """

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __call__(self) -> float:
        return self.epsilon

    def step(self, steps: int = 1) -> float:
        return self.epsilon

    def set_current_step(self, step: int) -> float:
        return self.epsilon

    def reset(self) -> float:
        return self.epsilon


class LinearSchedule(Schedule):
    """
    Decay epsilon linearly

    Parameters:
        eps_start: epsilon initial value
        eps_final: epsilon final value
        max_steps: decrement epsilon during the number of steps
        current_step: state of the current step

    """

    def __init__(
        self,
        eps_start: float = 1.0,
        eps_final: float = 0.01,
        max_steps: int = 10_000,
        current_step: int = 0,
    ):
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.max_steps = max_steps
        self.current_step = current_step

    def __call__(self) -> float:
        return max(self.eps_final, self.eps_start - self.current_step / self.max_steps)

    def step(self, steps: int = 1) -> float:
        "Increment the step counter and return the current value of epsilon"
        self.current_step += steps
        return self()

    def set_current_step(self, step: int) -> float:
        "Set the current step and return the current value"
        self.current_step = step
        return self()

    def reset(self) -> float:
        "Reset the steps to zero and return the start value"
        self.current_step = 0
        return self()