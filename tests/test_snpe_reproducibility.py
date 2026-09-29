"""SNPE's MCMC escape hatch: seeded, its own chain count, timed, and flagged when it was needed.

The fallback is not a corner case. It ended 39 of the 46 SNPE fits in a demonstration run, and those
runs sit a median 1.61 emcee sigma from emcee against 1.00 for the runs that did not need it. So it
has to be as reproducible as the rejection path, and a report has to be able to tell the two apart.

The unit tests stub sbi at its boundary and run instantly. The ``slow`` fits run the TORCH simulate
path (``predict_torch=``), the one ``snpe_gpu`` takes: the numpy path goes through sbi's
``simulate_for_sbi``, which re-seeds numpy's global generator every round and so hid the missing
seed. Their fallback is forced through the public threshold (``proposal_min_acceptance`` above 1
is failed by every rate), so no test depends on whether a tiny network happens to leak.
"""
import warnings

import numpy as np
import pytest

import whisper_cbpf as wp

sbi = pytest.importorskip("sbi")
torch = pytest.importorskip("torch")


# ------------------------------------------------------------------ stubbed at the sbi boundary
class _Estimator:
    def sample(self, shape, condition=None):
        return torch.zeros((int(shape[0]), 2))


class _Posterior:
    posterior_estimator = _Estimator()


class _Prior:
    def log_prob(self, theta):
        return torch.zeros((theta.shape[0],))


class _Inference:
    """Records what the MCMC posterior is built with; its sampler draws from the two global
    generators sbi's ``slice_np_vectorized`` uses (numpy's for the steps, torch's for the starts)."""

    def __init__(self):
        self.num_chains = []

    def build_posterior(self, de_net, **kw):
        self.num_chains.append(kw["mcmc_parameters"]["num_chains"])
        return self

    def set_default_x(self, x):
        pass

    def sample(self, shape, x=None, **kw):
        n = int(shape[0])
        return torch.as_tensor(np.random.rand(n, 2), dtype=torch.float32) + torch.rand(n, 2)


def _final_draw(seed, num_chains=4, inference=None):
    from whisper_cbpf.samplers.snpe import _robust_final_sample

    with pytest.warns(UserWarning, match="falling back to MCMC"):
        samples, method, _ = _robust_final_sample(
            _Posterior(), inference or _Inference(), None, _Prior(), torch, torch.zeros(3), 5,
            False, num_chains, min_acceptance=1.01, seed=seed)
    assert method == "mcmc_fallback"
    return samples


def _proposal_draw(seed):
    from whisper_cbpf.samplers.snpe import _robust_proposal_draw

    with pytest.warns(UserWarning, match="round-to-round proposal"):
        theta, method, _ = _robust_proposal_draw(
            _Posterior(), 5, _Inference(), None, _Prior(), torch, torch.zeros(3), False, 4,
            min_acceptance=1.01, seed=seed)
    assert method == "mcmc_fallback"
    return theta.numpy()


@pytest.mark.parametrize("draw", [_final_draw, _proposal_draw])
def test_the_fallback_draw_depends_on_seed_alone(draw):
    """Regression: nothing seeded the generators the slice sampler draws from, so two seed-0 GPU
    runs of SN2025pgp/arnett ended on max ln L -1798.5 and -1805.9 (chi2/N 377 and 1604)."""
    np.random.seed(123)
    torch.manual_seed(123)
    first = draw(seed=0)
    np.random.seed(456)                      # whatever the fit left the global generators in...
    torch.manual_seed(456)
    assert np.array_equal(first, draw(seed=0))              # ...the same seed gives the same draw
    assert not np.array_equal(first, draw(seed=1))


def test_num_chains_reaches_sbi_whatever_num_workers_is():
    """The chain count used to be ``min(20, max(4, num_workers))``: 4 at one worker, 20 at 30."""
    inference = _Inference()
    for chains in (2, 7, 33):
        _final_draw(seed=0, num_chains=chains, inference=inference)
    assert inference.num_chains == [2, 7, 33]


# ----------------------------------------------------------------------------- end to end (slow)
def _predict(parameters, times, bands=None):
    return float(parameters["amp"]) * np.exp(-np.asarray(times, float) / float(parameters["tau"])) + 0.5


def _predict_torch(theta, times):
    return theta[:, 0:1] * torch.exp(-times[None, :] / theta[:, 1:2]) + 0.5


PRIOR = wp.Prior({"amp": wp.Uniform(0.5, 5.0), "tau": wp.Uniform(4.0, 30.0)})
TINY = dict(num_simulations=150, num_samples=200, max_num_epochs=5, max_logl_scan=200,
            predict_torch=_predict_torch, device="cpu")


def _lc():
    t = np.linspace(1.0, 30.0, 30)
    err = np.full(30, 0.02)
    flux = _predict({"amp": 2.0, "tau": 12.0}, t) + np.random.default_rng(0).normal(0, err)
    return wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=err, name="toy")


@pytest.fixture(scope="module")
def runs():
    """Two seed-0 fits forced onto both fallbacks, a healthy fit (this toy's flow accepts ~9% at
    x_o, far above the 1e-3 default threshold), and a fit to data no prior draw can reach."""
    wp.register_model("snpe_repro_decay", _predict, ["amp", "tau"], prior=PRIOR, overwrite=True)
    lc = _lc()
    far = wp.LightCurve(time=lc.time, band=lc.band, flux=20.0 * lc.flux, flux_err=lc.flux_err)
    forced = dict(TINY, num_rounds=2, proposal_min_acceptance=1.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")               # the fallback notices, WAIC, max_num_epochs
        return {"a": wp.fit_SNPE(lc, "snpe_repro_decay", seed=0, **forced),
                "b": wp.fit_SNPE(lc, "snpe_repro_decay", seed=0, **forced),
                "healthy": wp.fit_SNPE(lc, "snpe_repro_decay", seed=0, num_rounds=1, **TINY),
                "ood": wp.fit_SNPE(far, "snpe_repro_decay", seed=0, num_rounds=1, **TINY)}


@pytest.mark.slow
def test_a_fit_that_falls_back_twice_is_bit_identical_at_a_fixed_seed(runs):
    a, b = runs["a"], runs["b"]
    assert a.info["proposal_draw_methods"] == ["mcmc_fallback"]
    assert a.max_log_likelihood == b.max_log_likelihood
    assert np.array_equal(a.samples.to_numpy(), b.samples.to_numpy())


@pytest.mark.slow
def test_the_threshold_that_guards_the_proposal_also_guards_the_final_draw(runs):
    """One ``proposal_min_acceptance``, so ``leakage`` and ``final_sample_method`` cannot disagree."""
    assert runs["a"].info["final_sample_method"] == "mcmc_fallback"


@pytest.mark.slow
def test_leakage_and_converged_flags_separate_a_leaky_fit_from_a_healthy_one(runs):
    leaky, healthy = runs["a"].info, runs["healthy"].info
    assert leaky["leakage"] is True and leaky["converged"] is False
    assert healthy["final_sample_method"] == "rejection"
    assert healthy["leakage"] is False and healthy["converged"] is True


@pytest.mark.slow
def test_x_o_min_rms_z_is_the_closest_prior_predictive_draw_in_sigma(runs):
    """An out-of-distribution statistic available before any training is spent.

    Data drawn from inside the prior have a round-0 simulation within a few sigma of them; the same
    curve scaled 20x is out of reach of every simulation, and the statistic says so.
    """
    inside, outside = runs["healthy"].info["x_o_min_rms_z"], runs["ood"].info["x_o_min_rms_z"]
    assert np.isfinite(inside) and 0.0 < inside < 20.0
    assert outside > 10.0 * inside


@pytest.mark.slow
def test_postprocess_s_reports_the_time_runtime_s_leaves_out(runs):
    for key in ("a", "healthy"):
        info = runs[key].info
        assert np.isfinite(info["postprocess_s"]) and info["postprocess_s"] > 0.0
    # the fallback draw is the expensive half of a forced run
    assert runs["a"].info["postprocess_s"] > runs["healthy"].info["postprocess_s"]


@pytest.mark.slow
def test_num_chains_is_its_own_setting_and_is_recorded(runs):
    assert runs["a"].info["num_chains"] == 4 and runs["a"].info["num_workers"] == 1
