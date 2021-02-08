from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import gym
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src import utils
from src.experience import ExperienceSource
from src.replay_buffer import Experience, ExperienceDiscounted, ReplayBuffer

Tensor = torch.Tensor


class ExperienceTensorBatch(NamedTuple):
    state: Tensor
    action: Tensor
    reward: Tensor
    done: Tensor
    last_state: Tensor


class DDPGActor(nn.Module):
    """
    Network that converts observations into actions.
    The network output is constrained between -1 and 1 (tanh activation),
    so it may require scaling depending on the environment.
    """

    def __init__(
        self,
        obs_size: int,
        act_size: int,
        hidden: Sequence[int] = (128,),
    ):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(obs_size, hidden[0]),
            nn.ReLU(),
        )
        self.hidden = nn.ModuleList()
        if len(hidden) > 1:
            self.hidden.extend(
                nn.Sequential(
                    nn.Linear(inp, outp),
                    nn.ReLU(),
                )
                for (inp, outp) in zip(hidden, hidden[1:])
            )

        self.output = nn.Sequential(nn.Linear(hidden[-1], act_size), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.input(obs)
        for hid in self.hidden:
            obs = hid(obs)
        return self.output(obs)


class DDPGCritic(nn.Module):
    """
    Network that estimates the value of a (state, action) pair.
    The output is unsconstrained (i.e. linear activation)
    """

    def __init__(
        self,
        obs_size: int,
        act_size: int,
        hidden1: int = 128,
        hidden2: int = 128,
    ):
        super().__init__()

        self.obs_input = nn.Sequential(
            nn.Linear(obs_size, hidden1),
            nn.ReLU(),
        )
        self.hidden = nn.Sequential(
            nn.Linear(hidden1 + act_size, hidden2),
            nn.ReLU(),
        )

        self.output = nn.Sequential(nn.Linear(hidden2, 1))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.obs_input(obs)
        x = self.hidden(torch.cat([x, action], dim=1))
        return self.output(x)


def init_weights_xavier_uniform(module: nn.Module) -> None:
    "Initialize the weights in the linear layers using `xavier uniform`"
    if type(module) == nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)


def create_ddpg_actor_critic(
    env: gym.Env,
    weight_init_fn: Optional[Callable] = init_weights_xavier_uniform,
    actor_kwargs: Optional[Dict[str, Union[Sequence[int], int]]] = {},
    critic_kwargs: Optional[Dict[str, int]] = {},
) -> Tuple[nn.Module, nn.Module]:
    actor = DDPGActor(
        obs_size=env.observation_space.shape[0],
        act_size=env.action_space.shape[0],
        **actor_kwargs,
    )
    critic = DDPGCritic(
        obs_size=env.observation_space.shape[0],
        act_size=env.action_space.shape[0],
        **critic_kwargs,
    )

    if weight_init_fn:
        actor.apply(weight_init_fn)
        critic.apply(weight_init_fn)

    return actor, critic


class Agent(ABC):
    def learn(
        self,
        epochs: int,
        val_every: int = -1,
        train_steps: int = 1,
        collect_steps: int = 1,
    ) -> None:

        for i in tqdm(range(1, epochs + 1)):

            if i % val_every == 0 and val_every > 0:
                if self._early_stoping():
                    break

            self._train_net(train_steps)
            self._collect_steps(collect_steps)

    def _early_stoping(self) -> bool:
        "Whether to stop the training based on the validation error"
        raise NotImplementedError

    def save_state_to_path(self, path: str) -> None:
        "Save the network state and other agent's states"
        dic = self.state_dict()
        torch.save(dic, path)

    def load_state_from_path(self, path: str) -> None:
        "Read a dict from the `path` and load it into the agent"
        dic = torch.load(path)
        self.load_state_from_dict(dic)

    @abstractmethod
    def fill_buffers(self) -> None:
        "Fill the Replay Buffers"

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        "Dictionary containing the agent's state"

    @abstractmethod
    def load_state_dict(self, dic: Dict[str, Any]) -> None:
        "Replace the agent's current states with the states in the dict"

    @abstractmethod
    def _train_net(self, steps: int) -> None:
        "Train the agents' networks"

    @abstractmethod
    def _collect_steps(self, steps: int) -> None:
        "Play a step in the environment using the actual policy"

    @staticmethod
    def _append_to_buffer(
        buffer: ReplayBuffer,
        discounted_exp: ExperienceDiscounted,
    ) -> None:
        "Take a discounted experience and append it to the replay buffer"
        obs = discounted_exp.state
        action = discounted_exp.action
        reward = discounted_exp.discounted_reward
        if discounted_exp.last_state is None:
            last_obs = obs
            done = True
        else:
            last_obs = discounted_exp.last_state
            done = False

        buffer.append(Experience(obs, action, reward, done, last_obs))

    @staticmethod
    def _prepare_batch(
        buffer: ReplayBuffer,
        batch_size: int,
        norm_rewards: bool = False,
    ) -> ExperienceTensorBatch:
        "Get a training batch for network's weights update"
        batch = buffer.sample(batch_size)
        state = torch.tensor(batch.state, dtype=torch.float32)
        action = torch.tensor(batch.action, dtype=torch.float32)
        reward = torch.tensor(batch.reward, dtype=torch.float32)
        done = torch.tensor(batch.done, dtype=torch.bool)
        last_state = torch.tensor(batch.last_state, dtype=torch.float32)

        if norm_rewards:
            reward = (reward - reward.mean()) / (reward.std() + 1e-6)

        return ExperienceTensorBatch(state, action, reward, done, last_state)

    @staticmethod
    def _fill_buffer_steps(
        exp_source: ExperienceSource,
        buffer: ReplayBuffer,
        num_experiences: int,
    ) -> None:
        "Fill the replay buffer with the specified num of experiences"
        for _ in tqdm(range(num_experiences), desc=str(buffer)):
            exp = exp_source.play_n_steps()
            Agent._append_to_buffer(buffer, exp)

    @staticmethod
    def _fill_buffer_episodes(
        exp_source: ExperienceSource,
        buffer: ReplayBuffer,
        num_episodes: int,
    ) -> None:
        "Fill the replay buffer with the specified num of episodes"
        for _ in tqdm(range(num_episodes), desc=str(buffer)):
            episodes = exp_source.play_episodes(num_episodes)
            for episode in episodes:
                for exp in episode:
                    Agent._append_to_buffer(buffer, exp)


class BCAgent(Agent):
    def __init__(
        self,
        demo_train_source: ExperienceSource,
        demo_val_source: ExperienceSource,
        demo_test_source: ExperienceSource,
        agent_val_source: ExperienceSource,
        agent_test_source: ExperienceSource,
        actor: DDPGActor,
        demo_buffer_size: int = 50_000,
        demo_batch_size: int = 64,
        actor_lr: float = 1e-4,
        actor_l2: float = 1e-2,
    ):
        self.demo_train_source = demo_train_source
        self.demo_val_source = demo_val_source
        self.demo_test_source = demo_test_source
        self.agent_val_source = agent_val_source
        self.agent_test_source = agent_test_source
        self.actor = actor
        self.demo_buffer_size = demo_buffer_size
        self.demo_batch_size = demo_batch_size

        self.demo_buffer = ReplayBuffer(capacity=demo_buffer_size)
        self.actor_optim = Adam(
            self.actor.parameters(), lr=actor_lr, weight_decay=actor_l2
        )

        self.fill_buffers()

        self.val_set_target = []
        for _ in range(self.demo_val_source.available_episodes):
            self.demo_val_source.play_episode()
            self.val_set_target.extend(
                self.demo_val_source.policy.env.history.duty_cycle
            )

        self._best_actor_weights = self.actor.state_dict()
        self._val_error = 1e6

    def _collect_steps(self, steps: int) -> None:
        pass

    def _early_stoping(self) -> bool:
        val_set_agent = []
        for _ in range(self.agent_val_source.available_episodes):
            self.agent_val_source.play_episode()
            val_set_agent.extend(self.agent_val_source.policy.env.history.duty_cycle)
        val_eror = utils.mse(self.val_set_target, val_set_agent)

        if val_eror > self._val_error:
            self.actor.load_state_dict(self._best_actor_weights)
            return True

        self._val_error = val_eror
        self._best_actor_weights = self.actor.state_dict()

        return False

    def _train_net(self, train_steps: int = 1) -> None:
        for _ in range(train_steps):
            demo_batch = Agent._prepare_batch(
                self.demo_buffer, self.demo_batch_size, norm_rewards=False
            )
            agent_action = self.actor(demo_batch.state)
            # Supervised traning
            loss = torch.nn.functional.mse_loss(
                agent_action.squeeze(-1), demo_batch.action.squeeze(-1)
            )
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

    def fill_buffers(self) -> None:
        Agent._fill_buffer_steps(
            exp_source=self.demo_train_source,
            buffer=self.demo_buffer,
            num_experiences=self.demo_buffer_size,
        )

    def state_dict(self) -> Dict[str, Any]:
        dic = {"actor_state_dict": self.actor.state_dict()}
        return dic

    def load_state_dict(self, dic: Dict[str, Any]) -> None:
        self.actor.load_state_dict(dic["actor_state_dict"])
        self.agent_val_source.policy.net = self.actor
        self.agent_test_source.policy.net = self.actor


class DDPGAgent(Agent):
    def __init__(self):
        pass


class BCDDPGAgent(Agent):
    def __init__(self):
        pass


if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    actor, critic = create_ddpg_actor_critic(env)
