import numpy as np

from whisper_labia import chi2_distance


def test_chi2_zero_when_equal():
    obs = np.array([1.0, 2.0, 3.0])
    err = np.array([0.1, 0.1, 0.1])
    assert chi2_distance(obs, err, obs) == 0.0


def test_chi2_value():
    obs = np.array([1.0, 2.0])
    sim = np.array([1.1, 1.8])
    err = np.array([0.1, 0.2])
    expected = ((1.0 - 1.1) / 0.1) ** 2 + ((2.0 - 1.8) / 0.2) ** 2
    assert np.isclose(chi2_distance(obs, err, sim), expected)
