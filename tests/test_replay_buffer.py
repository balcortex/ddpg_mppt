from src.replay_buffer import ReplayBuffer, Experience
import numpy as np


def test_replay_buffer():
    buffer = ReplayBuffer(capacity=3)
    assert len(buffer) == 0
    exp1 = Experience([1], 2, 3, True, [4])
    buffer.append(exp1)
    assert len(buffer) == 1
    sample = buffer.sample(1)
    assert np.array_equal(sample[0], np.array([[1]]))
    assert np.array_equal(sample[1], np.array([2]))
    assert np.array_equal(sample[2], np.array([3]))
    assert np.array_equal(sample[3], np.array([1]))
    assert np.array_equal(sample[4], np.array([[4]]))

    exp2 = Experience([5], 6, 7, False, [8])
    buffer.append(exp2)
    assert len(buffer) == 2
    samples = buffer.sample(2)
    if samples[3][0] == 1:
        assert np.array_equal(samples[0], np.array([[1], [5]]))
        assert np.array_equal(samples[1], np.array([2, 6]))
        assert np.array_equal(samples[2], np.array([3, 7]))
        assert np.array_equal(samples[4], np.array([[4], [8]]))
    else:
        assert np.array_equal(samples[0], np.array([[5], [1]]))
        assert np.array_equal(samples[1], np.array([6, 2]))
        assert np.array_equal(samples[2], np.array([7, 3]))
        assert np.array_equal(samples[4], np.array([[8], [4]]))

    buffer.append(exp2)
    buffer.append(exp2)
    buffer.append(exp2)
    assert len(buffer) == 3
    samples = buffer.sample(3)
    assert np.array_equal(samples[0], np.array([[5], [5], [5]]))
    assert np.array_equal(samples[1], np.array([6, 6, 6]))
    assert np.array_equal(samples[2], np.array([7, 7, 7]))
    assert np.array_equal(samples[4], np.array([[8], [8], [8]]))

    buffer.append(exp1)
    buffer.append(exp1)
    buffer.append(exp1)
    assert len(buffer) == 3
    samples = buffer.sample(2)
    assert np.array_equal(samples[0], np.array([[1], [1]]))
    assert np.array_equal(samples[1], np.array([2, 2]))
    assert np.array_equal(samples[2], np.array([3, 3]))
    assert np.array_equal(samples[4], np.array([[4], [4]]))
