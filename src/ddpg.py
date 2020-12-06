import torch
import torch.nn as nn
import gym
import numpy as np

# from src.policies import BasePolicy
from src.experience import ExperienceSourceDiscountedSteps, ExperienceDiscounted
from typing import Optional, Tuple, Dict, Any, Sequence
from src.common import TargetNet
from src.replay_buffer import ReplayBuffer, Experience
from collections import namedtuple
from tqdm import tqdm

Tensor = torch.Tensor
Training_Batch = namedtuple(
    "Training_Batch", field_names=["obs", "actions", "rewards", "dones", "last_obs"]
)


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


def create_ddpg_actor_critic(
    env: gym.Env, actor_kwargs: Dict[Any, Any] = {}, critic_kwargs: Dict[Any, Any] = {}
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
    return actor, critic


class DDPGAgent:
    def __init__(
        self,
        train_exp_source: ExperienceSourceDiscountedSteps,
        test_exp_sorce: ExperienceSourceDiscountedSteps,
        actor: DDPGActor,
        critic: DDPGCritic,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        tau: float = 1e-3,
        norm_rewards: bool = False,
    ):
        self.train_exp_source = train_exp_source
        self.test_exp_sorce = test_exp_sorce
        self.actor = actor
        self.critic = critic
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size
        self.tau = tau
        self.norm_rewards = norm_rewards

        self.target_actor = TargetNet(self.actor)
        self.target_critic = TargetNet(self.critic)
        self.target_actor.sync()
        self.target_critic.sync()

        self.gamma = self.train_exp_source.gamma
        self.n_steps = self.train_exp_source.n_steps

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self._fill_buffer(num_experiences=batch_size)

    def learn(self, steps: int, log_every: int = -1) -> None:
        losses = {}
        for i in tqdm(range(1, steps + 1)):
            if i % log_every == 0 and log_every > 0:
                print()
                print(f"{losses}")
                print(
                    f"env_steps={self.train_exp_source.step_counter}, ",
                    f"episodes={self.train_exp_source.episode_counter}",
                )
                print(
                    f"last_rew={self.train_exp_source.last_episode_reward:.2f}, ",
                    f"mean_rew={self.train_exp_source.mean_rewards(10):.2f}",
                )
                print()
            losses = self._train_net()

    def _train_net(self) -> Dict[Any, Any]:
        batch = self._prepare_training_batch()

        # - - - Critic Training - - -
        pred_last_act = self.target_actor(batch.last_obs)
        q_last = self.target_critic(batch.last_obs, pred_last_act).squeeze(-1)
        q_last[batch.dones] = 0.0  # The value of the last state is 0
        q_ref = batch.rewards + q_last * self.gamma ** self.n_steps
        q_pred = self.critic(batch.obs, batch.actions).squeeze(-1)
        # .detach() to stop gradient propogation for q_ref
        critic_loss = torch.nn.functional.mse_loss(q_ref.detach(), q_pred)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # - - - Actor Training - - -
        act_pred = self.actor(batch.obs)
        actor_loss = -self.critic(batch.obs, act_pred).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.target_actor.alpha_sync(alpha=self.tau)
        self.target_critic.alpha_sync(alpha=self.tau)

        return {
            "critic_loss": round(critic_loss.item(), 4),
            "actor_loss": round(actor_loss.item(), 4),
        }

    def _prepare_training_batch(self) -> Training_Batch:
        "Get a training batch for networks weight update"
        for disc_exp in next(self.train_exp_source):
            self._append_to_buffer(discounted_exp=disc_exp)

        batch = self.buffer.sample(batch_size=self.batch_size)
        obs = torch.tensor(batch.observations, dtype=torch.float32)
        actions = torch.tensor(batch.actions, dtype=torch.float32)
        rewards = torch.tensor(batch.rewards, dtype=torch.float32)
        dones = torch.tensor(batch.dones, dtype=torch.bool)
        last_obs = torch.tensor(batch.new_observations, dtype=torch.float32)

        if self.norm_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        return Training_Batch(
            obs=obs, actions=actions, rewards=rewards, dones=dones, last_obs=last_obs
        )

    def _fill_buffer(self, num_experiences: int) -> None:
        "Fill the replay buffer with the specified num of experiences"
        for _ in range(num_experiences):
            disc_exp = self.train_exp_source.play_n_steps()
            self._append_to_buffer(discounted_exp=disc_exp)

    def _append_to_buffer(self, discounted_exp: ExperienceDiscounted) -> None:
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

        exp = Experience(
            obs=obs, action=action, reward=reward, done=done, new_obs=last_obs
        )
        self.buffer.append(experience=exp)
