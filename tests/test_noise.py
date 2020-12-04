from src.noise import OUNoise, GaussianNoise


def test_ou_noise():
    noise = OUNoise(mean=0.0, std=0.01)
    assert noise.reset() == 0.0
    assert isinstance(noise.sample(), float)
    for _ in range(100):
        assert -1 <= noise.sample() <= 1


def test_gaussian_noise():
    noise = GaussianNoise(mean=0.0, std=0.01)
    assert -1 < noise.reset() < 1
    assert isinstance(noise.sample(), float)
    for _ in range(100):
        assert -1 <= noise.sample() <= 1