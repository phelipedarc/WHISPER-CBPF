"""Edge cases in space resolution, the mixture likelihood, flux-to-magnitude and WAIC.

Each of these produced a wrong number with no error at some point, so each has a test that pins
the correct behaviour. None needs JAX, so they run on the minimal install.
"""
from __future__ import annotations

import numpy as np
import pytest


def _toy_lc(n=24, seed=0):
    """A small flux-space light curve in two bands."""
    import whisper_cbpf as wp

    t = np.linspace(1.0, 30.0, n)
    band = np.array(["ztfg" if i % 2 else "ztfr" for i in range(n)])
    flux = 2.0 * np.exp(-t / 12.0) + 0.5
    rng = np.random.default_rng(seed)
    return wp.LightCurve(time=t, band=band, flux=flux + rng.normal(0, 0.02, n),
                         flux_err=np.full(n, 0.02), name="toy", redshift=0.01)


def test_resolve_space_is_public_and_auto_follows_data_mode():
    """``resolve_space`` was private; the samplers need the rule without building a likelihood."""
    from whisper_cbpf.likelihood import resolve_space

    lc = _toy_lc()
    assert resolve_space(lc, "auto") == "flux"          # flux-mode data
    assert resolve_space(lc, "mag") == "magnitude"
    assert resolve_space(lc, "magnitude") == "magnitude"
    assert resolve_space(lc, "flux") == "flux"
    with pytest.raises(ValueError, match="space must be"):
        resolve_space(lc, "nonsense")


def test_mixture_likelihood_pointwise_sums_to_its_total():
    """It overrode ``log_likelihood`` but not the pointwise form, so WAIC scored a plain Gaussian.

    Measured before the fix on 25 points at sigma = 0.1 with a uniform residual of 0.05: total
    29.1446 against a pointwise sum of 31.4662. On THIS test's own fixture (``_toy_lc()``, n = 24,
    sigma = 0.02, model = flux x 1.05) the pre-fix pair was -5.6509 against -55.1652 -- a different
    configuration, so do not read the two as the same measurement.
    """
    import whisper_cbpf as wp

    lc = _toy_lc()
    mf = np.asarray(lc.flux) * 1.05
    lik = wp.make_likelihood(lc, kind="mixture", space="flux")
    assert lik.log_likelihood(mf) == pytest.approx(float(lik.log_likelihood_pointwise(mf).sum()))
    # ...and it must still differ from the plain Gaussian, or the override is a no-op.
    plain = wp.make_likelihood(lc, kind="gaussian", space="flux")
    assert lik.log_likelihood(mf) != pytest.approx(plain.log_likelihood(mf))


def test_every_model_side_flux_to_mag_converter_agrees_on_zero_flux():
    """Eleven converters inside ``whisper_cbpf/``, one floor. The SNPE torch path used a bare
    ``finfo.tiny``, which agreed in float32 and disagreed by **19.13 mag** in float64 -- 778.03
    against everyone else's 758.90."""
    torch = pytest.importorskip("torch")
    from whisper_cbpf.plotting import _flux_to_quantity
    from whisper_cbpf.samplers.snpe import _torch_model_in_space

    expected_f64 = 758.9001
    got = float(_torch_model_in_space("magnitude", 3631.0, torch)(torch.zeros(1, dtype=torch.float64)))
    assert got == pytest.approx(expected_f64, abs=1e-3)
    assert float(_flux_to_quantity(np.zeros(1), "apparent_mag")[0]) == pytest.approx(expected_f64, abs=1e-3)
    # float32 keeps the dtype's smallest normal, ~103.72 mag
    got32 = float(_torch_model_in_space("magnitude", 3631.0, torch)(torch.zeros(1, dtype=torch.float32)))
    assert got32 == pytest.approx(103.7246, abs=1e-3)


def test_waic_flags_itself_unreliable_when_p_waic_exceeds_half_the_data():
    """p_waic = 939505 against 88 data points used to be returned with no warning at all."""
    from whisper_cbpf.metrics._numpy import _loo_waic

    rng = np.random.default_rng(0)
    good = rng.normal(-1.0, 0.05, size=(200, 40))                 # tight: reliable
    _, w, _ = _loo_waic(good)
    assert w["p_waic_reliable"] is True

    bad = good.copy()
    bad[:, 0] = rng.normal(0.0, 5e3, size=200)                    # one wildly varying point
    with pytest.warns(UserWarning, match="WAIC is unreliable"):
        _, w, _ = _loo_waic(bad)
    assert w["p_waic_reliable"] is False
    assert w["n_data"] == 40
