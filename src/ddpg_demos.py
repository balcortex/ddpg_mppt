import torch
import torch.nn as nn
import gym
import numpy as np

from src.experience import ExperienceSourceDiscountedSteps, ExperienceDiscounted
from typing import Tuple, Dict, Any, Sequence, Optional
from src.common import TargetNet
from src.replay_buffer import ReplayBuffer, Experience
from collections import namedtuple
from tqdm import tqdm

Tensor = torch.Tensor
TrainingBatch = namedtuple(
    "TrainingBatch", field_names=["obs", "actions", "rewards", "dones", "last_obs"]
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


class DDPGAgentDemo:
    def __init__(
        self,
        train_exp_source: ExperienceSourceDiscountedSteps,
        demo_exp_source: ExperienceSourceDiscountedSteps,
        test_exp_sorce: ExperienceSourceDiscountedSteps,
        actor: DDPGActor,
        critic: DDPGCritic,
        buffer_size: int = 10_000,
        demo_buffer_size: int = 10_000,
        batch_size: int = 64,
        demo_batch_size: int = 64,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        actor_l2: float = 0.0,
        critic_l2: float = 0.0,
        tau: float = 1e-3,
        norm_rewards: bool = False,
        lambda_rl: float = 1e-3,
        lambda_bc: float = 7e-3,
        q_filter: bool = False,
    ):
        self.train_exp_source = train_exp_source
        self.demo_exp_source = demo_exp_source
        self.test_exp_sorce = test_exp_sorce
        self.actor = actor
        self.critic = critic
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.demo_buffer = ReplayBuffer(capacity=demo_buffer_size)
        self.demo_buffer_size = demo_buffer_size
        self.batch_size = batch_size
        self.demo_batch_size = demo_batch_size
        self.tau = tau
        self.norm_rewards = norm_rewards
        self.lambda_rl = lambda_rl
        self.lambda_bc = lambda_bc
        self.q_filter = q_filter

        self.target_actor = TargetNet(self.actor)
        self.target_critic = TargetNet(self.critic)
        self.target_actor.sync()
        self.target_critic.sync()

        self.gamma = self.train_exp_source.gamma
        self.n_steps = self.train_exp_source.n_steps

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, weight_decay=actor_l2
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=critic_l2
        )

        self._fill_buffers()

    def learn(self, epochs: int, train_steps: int = 1, log_every: int = -1) -> None:
        losses = {}
        for i in tqdm(range(1, epochs + 1)):
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

            for j in range(train_steps):
                batch = self._prepare_training_batch(
                    self.buffer, self.batch_size, self.norm_rewards
                )
                batch_demo = self._prepare_training_batch(
                    self.demo_buffer, self.demo_batch_size, self.norm_rewards
                )
                losses = self._train_net(batch, batch_demo)

            self._play_step()

    def save(self, path: str) -> None:
        torch.save(
            {
                "critic_state_dict": self.critic.state_dict(),
                "actor_state_dict": self.actor.state_dict(),
            },
            path,
        )

    # def load(self, path: str) -> None:
    #     checkpoint = torch.load(path)
    #     self.critic.load_state_dict(checkpoint["critic_state_dict"])
    #     self.actor.load_state_dict(checkpoint["actor_state_dict"])

    def _train_net(
        self, batch: TrainingBatch, demo_batch: TrainingBatch
    ) -> Dict[Any, Any]:
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
        # Behaviour cloning loss
        act_pred_bc = self.actor(demo_batch.obs)
        if self.q_filter:
            q_bc = self.critic(demo_batch.obs, demo_batch.actions)
            # print(f"{q_bc=}")
            q_rl = self.critic(demo_batch.obs, act_pred_bc)
            # print(f"{q_rl=}")
            idx = q_bc < q_rl
            # print(f"{idx=}")
            # print(f"{act_pred_bc=}")
            # print(f"{demo_batch.actions=}")
            # act_pred_bc[idx] = 0.0
            # demo_batch.actions[idx] = 0.0
            demo_batch.actions[idx] = act_pred_bc[idx]

            # print(f"{act_pred_bc=}")
            # print(f"{demo_batch.actions=}")

        bc_loss = torch.nn.functional.mse_loss(
            act_pred_bc.squeeze(-1).detach(), demo_batch.actions.squeeze(-1)
        )
        # print(f"{bc_loss=}")
        # RL Loss
        act_pred = self.actor(batch.obs)
        actor_loss = -self.critic(batch.obs, act_pred).mean() * self.lambda_rl
        # Total loss
        actor_loss += bc_loss * self.lambda_bc
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.target_actor.alpha_sync(alpha=self.tau)
        self.target_critic.alpha_sync(alpha=self.tau)

        return {
            "critic_loss": round(critic_loss.item(), 4),
            "actor_loss": round(actor_loss.item(), 4),
        }

    def _play_step(self) -> None:
        for disc_exp in next(self.train_exp_source):
            self._append_to_buffer(buffer=self.buffer, discounted_exp=disc_exp)
        # for disc_exp in next(self.demo_exp_source):
        #     self._append_to_buffer(buffer=self.demo_buffer, discounted_exp=disc_exp)

    def _fill_buffers(self) -> None:
        self._fill_buffer(
            exp_source=self.train_exp_source,
            buffer=self.buffer,
            num_experiences=self.batch_size,
        )
        self._fill_buffer(
            exp_source=self.demo_exp_source,
            buffer=self.demo_buffer,
            num_experiences=self.demo_buffer_size,
        )

    @staticmethod
    def _fill_buffer(
        exp_source: ExperienceSourceDiscountedSteps,
        buffer: ReplayBuffer,
        num_experiences: int,
    ) -> None:
        "Fill the replay buffer with the specified num of experiences"
        for _ in tqdm(range(num_experiences), desc="Buffer"):
            disc_exp = exp_source.play_n_steps()
            DDPGAgentDemo._append_to_buffer(buffer=buffer, discounted_exp=disc_exp)

    @staticmethod
    def _prepare_training_batch(
        buffer: ReplayBuffer, batch_size: int, norm_rewards: bool
    ) -> TrainingBatch:
        "Get a training batch for networks weight update"
        batch = buffer.sample(batch_size=batch_size)
        obs = torch.tensor(batch.observations, dtype=torch.float32)
        actions = torch.tensor(batch.actions, dtype=torch.float32)
        rewards = torch.tensor(batch.rewards, dtype=torch.float32)
        dones = torch.tensor(batch.dones, dtype=torch.bool)
        last_obs = torch.tensor(batch.new_observations, dtype=torch.float32)

        if norm_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        return TrainingBatch(
            obs=obs, actions=actions, rewards=rewards, dones=dones, last_obs=last_obs
        )

    @staticmethod
    def _append_to_buffer(
        buffer: ReplayBuffer, discounted_exp: ExperienceDiscounted
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

        exp = Experience(
            obs=obs, action=action, reward=reward, done=done, new_obs=last_obs
        )
        buffer.append(experience=exp)
