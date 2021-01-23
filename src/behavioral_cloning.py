import os
from collections import namedtuple
from typing import Any, Dict, Optional, Sequence

import gym
import torch
import torch.nn as nn
from tqdm import tqdm

from src.experience import ExperienceDiscounted, ExperienceSourceDiscountedSteps
from src.logger import logger
from src.replay_buffer import Experience, ReplayBuffer
from src.utils import efficiency, mse

Tensor = torch.Tensor
TrainingBatch = namedtuple(
    "TrainingBatch", field_names=["obs", "actions", "rewards", "dones", "last_obs"]
)


class Actor(nn.Module):
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


class BehavioralCloning:
    def __init__(
        self,
        demo_exp_source: ExperienceSourceDiscountedSteps,
        val_po_exp_source: ExperienceSourceDiscountedSteps,
        val_ddpg_exp_source: ExperienceSourceDiscountedSteps,
        test_po_exp_source: ExperienceSourceDiscountedSteps,
        test_ddpg_exp_source: ExperienceSourceDiscountedSteps,
        actor: Optional[Actor] = None,
        demo_buffer_size: int = 10_000,
        demo_batch_size: int = 64,
        actor_lr: float = 1e-4,
        actor_l2: float = 0.0,
        actor_ckp: Optional[str] = None,
    ):
        self.demo_exp_source = demo_exp_source
        self.val_po_exp_source = val_po_exp_source
        self.val_ddpg_exp_source = val_ddpg_exp_source
        self.test_po_exp_source = test_po_exp_source
        self.test_ddpg_exp_source = test_ddpg_exp_source
        self.demo_buffer = ReplayBuffer(capacity=demo_buffer_size)
        self.demo_buffer_size = demo_buffer_size
        self.demo_batch_size = demo_batch_size

        self.actor = actor or self.create_actor(self.demo_exp_source)
        if actor_ckp:
            self.load(actor_ckp)
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, weight_decay=actor_l2
        )

        self._fill_buffers()
        # The validation episode from P&O just needs to run once
        self.val_po_exp_source.play_episode()
        self.po_dc = self.val_po_exp_source.policy.env.history.duty_cycle

    def learn(
        self,
        epochs: int,
        val_every: int = 500,
        train_steps: int = 1,
        log_every: int = -1,
    ) -> None:
        losses = {}

        error = 1e6
        best_weights = self.actor.state_dict()

        for i in tqdm(range(1, epochs + 1)):
            if i % log_every == 0 and log_every > 0:
                print()
                print(f"{losses}")

            if i % val_every == 0:
                error_ = self.run_validation()
                print(f"val error={error_}")
                if error_ > error:
                    self.actor.load_state_dict(best_weights)
                    break
                error = error_
                best_weights = self.actor.state_dict()

            for j in range(train_steps):
                batch_demo = self._prepare_training_batch(
                    self.demo_buffer, self.demo_batch_size, norm_rewards=False
                )
                losses = self._train_net(batch_demo)

        self.run_test(self.test_po_exp_source)
        self.run_test(self.test_ddpg_exp_source)

    def run_validation(self) -> float:
        self.val_ddpg_exp_source.play_episode()
        rl_dc = self.val_ddpg_exp_source.policy.env.history.duty_cycle
        return mse(self.po_dc, rl_dc)

    def save(self, path: str) -> None:
        torch.save({"actor_state_dict": self.actor.state_dict()}, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        logger.info(f"Loaded from {path}")
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.test_exp_sorce.policy.net = self.actor

    def _train_net(self, demo_batch: TrainingBatch) -> Dict[str, float]:
        # Behaviour cloning loss
        act_pred_bc = self.actor(demo_batch.obs)
        bc_loss = torch.nn.functional.mse_loss(
            act_pred_bc.squeeze(-1), demo_batch.actions.squeeze(-1)
        )
        self.actor_optim.zero_grad()
        bc_loss.backward()
        self.actor_optim.step()

        return {"bc_loss": round(bc_loss.item(), 4)}

    def _fill_buffers(self) -> None:
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
            BehavioralCloning._append_to_buffer(buffer=buffer, discounted_exp=disc_exp)

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

    @staticmethod
    def create_actor(env: gym.Env, actor_kwargs: Dict[Any, Any] = {}) -> nn.Module:
        actor = Actor(
            obs_size=env.observation_space.shape[0],
            act_size=env.action_space.shape[0],
            **actor_kwargs,
        )
        return actor

    @staticmethod
    def run_test(exp_source: ExperienceSourceDiscountedSteps) -> float:
        test_env = exp_source.policy.env
        exp_source.play_episode()
        p_real, *_ = test_env.pvarray.get_true_mpp(
            test_env.history.g, test_env.history.amb_t
        )
        return efficiency(p_real, test_env.history.p)
