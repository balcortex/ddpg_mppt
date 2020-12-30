from src.ddpg import DDPGAgent
import gym
from src.experience import ExperienceSourceDiscountedSteps
from src.policies import DDPGPolicy
from src.ddpg import create_ddpg_actor_critic
from src.noise import OUNoise
from src.schedule import LinearSchedule

GAMMA = 0.99
N_STEPS = 4
BATCH_SIZE = 32
BUFFER_SIZE = 10_000
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
TAU = 1e-3

env = gym.make("Pendulum-v0")
test_env = gym.make("Pendulum-v0")
actor, critic = create_ddpg_actor_critic(env=env)
noise = OUNoise(mean=0.0, std=2, theta=0.0, dt=1)
schedule = LinearSchedule(max_steps=1000)
collect_policy = DDPGPolicy(
    env=env,
    net=actor,
    noise=noise,
    schedule=schedule,
)
test_policy = DDPGPolicy(
    env=test_env,
    net=actor,
    noise=None,
    schedule=None,
)
train_exp_source = ExperienceSourceDiscountedSteps(
    policy=collect_policy,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
)
test_exp_source = ExperienceSourceDiscountedSteps(
    policy=test_policy,
    gamma=GAMMA,
    n_steps=N_STEPS,
    steps=1,
    render=True,
)
agent = DDPGAgent(
    train_exp_source=train_exp_source,
    test_exp_sorce=test_exp_source,
    actor=actor,
    critic=critic,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    tau=TAU,
    norm_rewards=True,
)
agent.learn(epochs=1000, train_steps=4, log_every=50)
test_exp_source.play_episode()
test_env.close()
