import pickle

import numpy as np

from whisper_labia.priors import LogUniform, Prior, Uniform


def test_uniform_sample_in_bounds():
    rng = np.random.default_rng(0)
    u = Uniform(2, 5)
    xs = [u.sample(rng) for _ in range(200)]
    assert all(2 <= x <= 5 for x in xs)


def test_uniform_log_prob():
    u = Uniform(0, 10)
    assert np.isclose(u.log_prob(5), -np.log(10))
    assert u.log_prob(-1) == -np.inf


def test_uniform_rescale():
    u = Uniform(0, 10)
    assert np.isclose(u.rescale(0.0), 0.0)
    assert np.isclose(u.rescale(0.5), 5.0)
    assert np.isclose(u.rescale(1.0), 10.0)


def test_loguniform_in_bounds():
    rng = np.random.default_rng(0)
    lu = LogUniform(1e-2, 1e2)
    xs = [lu.sample(rng) for _ in range(200)]
    assert all(1e-2 <= x <= 1e2 for x in xs)


def test_prior_sample_reproducible():
    p = Prior({"a": Uniform(0, 1), "b": Uniform(1, 2)})
    s1 = p.sample(np.random.default_rng(42))
    s2 = p.sample(np.random.default_rng(42))
    assert s1 == s2 and set(s1) == {"a", "b"}


def test_prior_picklable():
    # Required so priors can cross process boundaries in parallel ABC.
    p = Prior({"a": Uniform(0, 1), "b": LogUniform(1e-3, 1e3)})
    assert pickle.loads(pickle.dumps(p)).names == ["a", "b"]
