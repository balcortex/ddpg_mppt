import gym
from src.policies import RandomPolicy
from src.experience import (
    ExperienceSource,
    ExperienceSourceEpisodes,
    ExperienceSourceDiscounted,
    ExperienceSourceDiscountedSteps,
    ExperienceSourceDiscountedEpisodes,
)
import numpy as np


def test_exp_source():
    env = gym.make("Pendulum-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSource(policy)
    experience_1 = next(exp_source)
    assert exp_source.step_counter == 1
    experience_2 = next(exp_source)
    assert exp_source.step_counter == 2
    assert np.array_equal(experience_1.last_state, experience_2.state)
    experience_3 = exp_source.play_step()
    assert exp_source.step_counter == 3
    assert np.array_equal(experience_2.last_state, experience_3.state)

    env.reset()
    episode = exp_source.play_episode()
    assert exp_source.step_counter == 203
    assert exp_source.episode_counter == 1
    assert len(episode) == 200
    assert episode[-1].last_state == None

    exp_source.reset()
    episodes = exp_source.play_episodes(episodes=3)
    for episode in episodes:
        assert len(episode) == 200
        assert episode[-1].last_state == None
    assert exp_source.step_counter == 600
    assert exp_source.episode_counter == 3


def test_exp_source_episodes():
    env = gym.make("Pendulum-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceEpisodes(policy, episodes=3)

    episodes = next(exp_source)
    assert exp_source.step_counter == 600
    assert exp_source.episode_counter == 3
    assert len(episodes) == 3
    for episode in episodes:
        assert np.array_equal(episode[0].last_state, episode[1].state)
        assert len(episode) == 200
        assert episode[-1].last_state == None

    env.reset()
    experience_1 = exp_source.play_step()
    experience_2 = exp_source.play_step()
    assert exp_source.step_counter == 602
    assert exp_source.episode_counter == 3
    assert np.array_equal(experience_1.last_state, experience_2.state)
    experience_3 = exp_source.play_step()
    assert exp_source.step_counter == 603
    assert exp_source.episode_counter == 3
    assert np.array_equal(experience_2.last_state, experience_3.state)

    exp_source.reset()
    episode = exp_source.play_episode()
    assert len(episode) == 200
    assert episode[-1].last_state == None
    assert exp_source.step_counter == 200
    assert exp_source.episode_counter == 1


def test_exp_source_discounted_1step():
    env = gym.make("CartPole-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceDiscounted(policy, gamma=0.99, n_steps=1)
    experience_1 = next(exp_source)
    experience_2 = next(exp_source)
    assert exp_source.step_counter == 2
    assert exp_source.last_episode_reward == None
    assert np.array_equal(experience_1.last_state, experience_2.state)
    assert experience_1.discounted_reward == 1.0
    assert experience_1.reward == 1.0
    assert experience_1.steps == 1
    experience_3 = exp_source.play_step()
    assert exp_source.step_counter == 3
    assert np.array_equal(experience_2.last_state, experience_3.state)
    assert experience_3.reward == 1.0

    env.reset()
    episode = exp_source.play_episode()
    assert exp_source.episode_counter == 1
    assert episode[-1].last_state == None
    assert episode[-1].discounted_reward == 1.0
    assert episode[-1].steps == 1
    assert episode[-1].reward == 1
    assert exp_source.step_counter == exp_source.last_episode_reward
    assert exp_source.step_counter == exp_source.mean_rewards(episodes=3)

    env.reset()
    episodes = exp_source.play_episodes(episodes=3)
    assert exp_source.episode_counter == 4
    for episode in episodes:
        assert episode[-1].last_state == None
        assert episode[-1].discounted_reward == 1.0
        assert episode[-1].steps == 1
        assert episode[-1].reward == 1
    assert len(episodes[-1]) == exp_source.last_episode_reward
    assert exp_source.mean_rewards(episodes=4) == exp_source.step_counter / 4
    assert exp_source.mean_rewards(episodes=5) == exp_source.step_counter / 4
    assert (
        len(episodes[0]) + len(episodes[1]) + len(episodes[2])
        == exp_source.mean_rewards(episodes=3) * 3
    )

    exp_source.reset()
    episodes = exp_source.play_episodes(episodes=3)
    assert exp_source.episode_counter == 3
    for episode in episodes:
        assert episode[-1].last_state == None
        assert episode[-1].discounted_reward == 1.0
        assert episode[-1].steps == 1
        assert episode[-1].reward == 1
    assert len(episodes[-1]) == exp_source.last_episode_reward
    assert exp_source.mean_rewards(episodes=4) == exp_source.step_counter / 3
    assert exp_source.mean_rewards(episodes=5) == exp_source.step_counter / 3
    assert (
        len(episodes[0]) + len(episodes[1]) + len(episodes[2])
        == exp_source.mean_rewards(episodes=3) * 3
    )

    env = gym.make("Pendulum-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceDiscounted(policy, gamma=0.99, n_steps=1)

    episode = exp_source.play_episode()
    assert len(episode) == 200
    assert exp_source.step_counter == 200
    assert exp_source.episode_counter == 1

    episodes = exp_source.play_episodes(episodes=3)
    for episode in episodes:
        assert len(episode) == 200
    assert exp_source.step_counter == 800
    assert exp_source.episode_counter == 4


def test_exp_source_discounted_2step():
    env = gym.make("CartPole-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceDiscounted(policy, gamma=0.99, n_steps=2)
    experience_1 = next(exp_source)
    experience_2 = next(exp_source)
    assert exp_source.step_counter == 4
    assert np.array_equal(experience_1.last_state, experience_2.state)
    assert experience_1.discounted_reward == 1 + 0.99
    assert experience_1.reward == 2.0
    assert experience_1.steps == 2
    experience_3 = exp_source.play_step()
    assert exp_source.step_counter == 5
    assert np.array_equal(experience_2.last_state, experience_3.state)
    assert experience_3.reward == 1.0

    env.reset()
    episode = exp_source.play_episode()
    assert exp_source.episode_counter == 1
    assert episode[-1].last_state == None
    if episode[-1].steps == 1:
        assert episode[-1].discounted_reward == 1.0
        assert episode[-1].reward == 1
    else:
        assert episode[-1].steps == 2
        assert episode[-1].discounted_reward == 1.0 + 0.99
        assert episode[-1].reward == 2

    env.reset()
    episodes = exp_source.play_episodes(episodes=3)
    assert exp_source.episode_counter == 4
    for episode in episodes:
        assert episode[-1].last_state == None
        if episode[-1].steps == 1:
            assert episode[-1].discounted_reward == 1.0
            assert episode[-1].reward == 1
        else:
            assert episode[-1].steps == 2
            assert episode[-1].discounted_reward == 1.0 + 0.99
            assert episode[-1].reward == 2

    env = gym.make("Pendulum-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceDiscounted(policy, gamma=0.99, n_steps=2)

    episode = exp_source.play_episode()
    assert len(episode) == 100
    assert exp_source.step_counter == 200
    assert exp_source.episode_counter == 1

    episodes = exp_source.play_episodes(episodes=3)
    for episode in episodes:
        assert len(episode) == 100
    assert exp_source.step_counter == 800
    assert exp_source.episode_counter == 4

    env.reset()
    exp_source = ExperienceSourceDiscounted(policy, gamma=0.99, n_steps=3)
    episode = exp_source.play_episode()
    assert len(episode) == 67
    assert exp_source.step_counter == 200
    assert exp_source.episode_counter == 1

    env.reset()
    exp_source = ExperienceSourceDiscounted(policy, gamma=0.99, n_steps=11)
    episode = exp_source.play_episode()
    assert len(episode) == 19
    assert exp_source.step_counter == 200
    assert exp_source.episode_counter == 1


def test_exp_source_steps_discounted_1step():
    env = gym.make("CartPole-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceDiscountedSteps(
        policy,
        gamma=0.99,
        n_steps=1,
        steps=8,
    )
    experience_batch_1 = next(exp_source)
    assert len(experience_batch_1) == 8
    assert exp_source.step_counter == 8
    assert exp_source.episode_counter == 0
    assert np.array_equal(experience_batch_1[0].last_state, experience_batch_1[1].state)
    assert experience_batch_1[0].discounted_reward == 1.0
    assert experience_batch_1[0].reward == 1.0
    assert experience_batch_1[0].steps == 1
    assert experience_batch_1[-1].reward == 1.0
    assert experience_batch_1[-1].discounted_reward == 1.0
    assert experience_batch_1[-1].steps == 1
    experience = exp_source.play_step()
    assert experience.reward == 1.0
    assert exp_source.step_counter == 9
    assert exp_source.episode_counter == 0

    env.reset()
    episode = exp_source.play_episode()
    assert episode[-1].last_state == None
    assert episode[-1].discounted_reward == 1.0
    assert episode[-1].reward == 1
    assert exp_source.episode_counter == 1

    env.reset()
    episodes = exp_source.play_episodes(episodes=3)
    for episode in episodes:
        assert episode[-1].last_state == None
        assert episode[-1].discounted_reward == 1.0
        assert episode[-1].reward == 1
    assert exp_source.episode_counter == 4

    env = gym.make("Pendulum-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceDiscountedSteps(
        policy,
        gamma=0.99,
        n_steps=1,
        steps=8,
    )

    episode = exp_source.play_episode()
    assert len(episode) == 200
    assert exp_source.step_counter == 200
    assert exp_source.episode_counter == 1

    episodes = exp_source.play_episodes(episodes=3)
    for episode in episodes:
        assert len(episode) == 200
    assert exp_source.step_counter == 800
    assert exp_source.episode_counter == 4


def test_exp_source_steps_discounted_2step():
    env = gym.make("CartPole-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceDiscountedSteps(
        policy,
        gamma=0.99,
        n_steps=2,
        steps=4,
    )
    experience_batch_1 = next(exp_source)
    assert len(experience_batch_1) == 4
    assert exp_source.step_counter == 8
    assert exp_source.episode_counter == 0
    assert np.array_equal(experience_batch_1[0].last_state, experience_batch_1[1].state)
    assert experience_batch_1[0].discounted_reward == 1 + 0.99
    assert experience_batch_1[0].reward == 2.0
    assert experience_batch_1[0].steps == 2
    assert experience_batch_1[-1].reward == 2.0
    assert experience_batch_1[-1].discounted_reward == 1 + 0.99
    assert experience_batch_1[-1].steps == 2
    experience = exp_source.play_step()
    assert experience.reward == 1.0
    assert exp_source.step_counter == 9
    assert exp_source.episode_counter == 0

    env.reset()
    episode = exp_source.play_episode()
    assert episode[-1].last_state == None
    if episode[-1].steps == 1:
        assert episode[-1].discounted_reward == 1.0
        assert episode[-1].reward == 1.0
    else:
        assert episode[-1].discounted_reward == 1 + 0.99
        assert episode[-1].reward == 2.0
        assert episode[-1].steps == 2
    assert exp_source.episode_counter == 1

    env.reset()
    episodes = exp_source.play_episodes(episodes=3)
    for episode in episodes:
        if episode[-1].steps == 1:
            assert episode[-1].discounted_reward == 1.0
            assert episode[-1].reward == 1.0
        else:
            assert episode[-1].discounted_reward == 1 + 0.99
            assert episode[-1].reward == 2.0
            assert episode[-1].steps == 2
    assert exp_source.episode_counter == 4

    env = gym.make("Pendulum-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceDiscountedSteps(
        policy,
        gamma=0.99,
        n_steps=8,
        steps=1,
    )

    episode = exp_source.play_episode()
    assert len(episode) == 25
    assert exp_source.episode_counter == 1
    assert exp_source.step_counter == 200

    episodes = exp_source.play_episodes(episodes=3)
    for episode in episodes:
        assert len(episode) == 25
    assert exp_source.episode_counter == 4
    assert exp_source.step_counter == 800


def test_exp_source_episode_discounted_1step():
    env = gym.make("CartPole-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceDiscountedEpisodes(
        policy, gamma=0.99, n_steps=1, episodes=3
    )
    episodes_batch_1 = next(exp_source)
    assert exp_source.episode_counter == 3
    assert len(episodes_batch_1) == 3
    for episode in episodes_batch_1:
        assert np.array_equal(episode[0].last_state, episode[1].state)
        assert episode[-1].last_state == None
        assert episode[0].reward == 1.0
        assert episode[0].discounted_reward == 1.0
        assert episode[0].steps == 1

    env = gym.make("Pendulum-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceDiscountedEpisodes(
        policy, gamma=0.99, n_steps=1, episodes=3
    )
    episodes_batch_1 = next(exp_source)
    assert len(episodes_batch_1) == 3
    for episode in episodes_batch_1:
        assert len(episode) == 200
    assert exp_source.episode_counter == 3
    assert exp_source.step_counter == 600


def test_exp_source_episode_discounted_3step():
    env = gym.make("CartPole-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceDiscountedEpisodes(
        policy, gamma=1.0, n_steps=3, episodes=3
    )
    episodes_batch_1 = next(exp_source)
    assert exp_source.episode_counter == 3
    assert len(episodes_batch_1) == 3
    for episode in episodes_batch_1:
        assert np.array_equal(episode[0].last_state, episode[1].state)
        assert episode[-1].last_state == None
        assert episode[0].reward == 3.0
        assert episode[0].discounted_reward == 3.0
        assert episode[0].steps == 3

    env.reset()
    exp_source = ExperienceSourceDiscountedEpisodes(
        policy, gamma=0.99, n_steps=3, episodes=3
    )
    episodes_batch_1 = next(exp_source)
    assert exp_source.episode_counter == 3
    assert len(episodes_batch_1) == 3
    for episode in episodes_batch_1:
        assert np.array_equal(episode[0].last_state, episode[1].state)
        assert episode[-1].last_state == None
        assert episode[0].reward == 3.0
        assert episode[0].discounted_reward == 1.0 + 0.99 + 0.99 ** 2
        assert episode[0].steps == 3

    env = gym.make("Pendulum-v0")
    policy = RandomPolicy(env=env)
    exp_source = ExperienceSourceDiscountedEpisodes(
        policy, gamma=1.0, n_steps=3, episodes=3
    )
    episodes_batch_1 = next(exp_source)
    assert len(episodes_batch_1) == 3
    for episode in episodes_batch_1:
        assert len(episode) == 67
    assert exp_source.episode_counter == 3
    assert exp_source.step_counter == 600
