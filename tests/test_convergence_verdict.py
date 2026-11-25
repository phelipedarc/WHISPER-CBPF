"""``wp.fit`` says, in plain English, when an MCMC fit has not converged.

The claims, each a test:

1. **One sentence per failed check, with the exact words.** Each check ``diagnostics()`` can fail
   for an MCMC-type fit -- R-hat between independent ensembles, R-hat across walkers or chains,
   bulk and tail ESS, chain length over the autocorrelation time, stuck and frozen walkers, the
   prior-edge pile-up, too few data points, no draws, divergences, the log-likelihood R-hat,
   stranded and frozen chains, the gaps to the prior scan's optimum and to the likelihood maximum
   -- gives one sentence naming the metric in words, the parameter, the value and the threshold,
   and what to do. A check this module does not know still gives a sentence.
2. **The message** is a first line ``"Not converged:"`` and one sentence per line; nothing when
   every check passes. A sampler that flags itself when no check fails is not lost.
3. **``wp.fit`` warns once.** After an MCMC-type fit that fails a check, ``wp.fit`` issues exactly
   one ``UserWarning`` -- the message of claim 2, at the caller's line -- and keeps it in
   ``info["convergence_verdict"]``; the sampler's own not-converged warning is not repeated.
4. **A converged fit is silent**, and ``check_convergence=False`` turns the check and every
   convergence warning off.
5. **A fit that is not MCMC** gets no verdict, but a sampler that flags itself still warns.
6. **A posterior problem is not called non-convergence.** When the only failed checks judge the
   posterior itself (draws piled at a prior bound, too few data points) and the sampler did not
   flag itself, the first line is ``"Check the posterior:"``; with any mixing check failing too,
   or the sampler's own flag down, it stays ``"Not converged:"``. ``wp.fit`` warns with it.
7. **The advice names the cap that stopped the run.** A run until converged that stopped at its
   time cap is told to raise ``max_time_s`` (to the time its pace needs for the target), and
   ``max_steps`` as well only when the target is past that cap; one stopped at ``max_steps`` is
   told to raise ``max_steps``.
"""
from __future__ import annotations

import copy
import math
import warnings

import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp
from whisper_cbpf import results as R
from whisper_cbpf.priors import Prior, Uniform
from whisper_cbpf.results import DiagnosticRow, DiagnosticsReport
from whisper_cbpf.samplers import _SAMPLERS
from whisper_cbpf.samplers._verdict import (verdict_heading, verdict_message,
                                             verdict_sentences)
from whisper_cbpf.samplers.base import SamplerResult
from whisper_cbpf.samplers.jax import _diagnostics as dg

NAMES = ["mstar", "b"]
LONGER = "Run longer (max_steps=10000)"


def _result(sampler, x, info, *, n_data=40, n_params=2, max_ll=0.0, prior=None):
    """A result whose draws by walker (or chain) are ``x`` (chains, draws, 2)."""
    res = SamplerResult(sampler=sampler, model="toy", parameters=list(NAMES),
                        samples=pd.DataFrame(x.reshape(-1, 2), columns=NAMES), summary={},
                        best_params={}, n_data=n_data, n_params=n_params, runtime_s=1.0,
                        info=info, max_log_likelihood=max_ll)
    res.samples_by_chain = x
    if prior is not None:
        res.provenance = {"model": {"prior": R.prior_record(prior)}}
    return res


def _ensemble(x=None, **info):
    """A healthy 4 x 8-walker emcee_jax result run until converged for 5000 steps; ``info``
    overrides the recorded checks to fail one."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 400, 2)) if x is None else x
    base = {"nwalkers": 32, "n_ensembles": 4, "nsteps": 5000, "stuck_walkers": [],
            "frozen_walkers": [], "max_autocorr_time": 40.0, "autocorr_time": [40.0, 20.0],
            "ensemble_rhat": {"mstar": 1.002, "b": 1.001}, "init": "prior_scan",
            "until_converged": {"enabled": True, "steps_run": 5000}, "converged": True,
            "convergence_problems": []}
    return _result("emcee_jax", x, {**base, **info})


def _nuts(shift_chain=None, **info):
    """A healthy 4-chain NUTS result (500 draws each), as ``nuts_gpu`` records it."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 500, 2))
    if shift_chain is not None:
        x[shift_chain, :, 0] += 8.0
    ll = -0.5 * (x ** 2).sum(-1)
    rec = dg.chain_health(x, NAMES, ll, n_divergences=0, reference_ll=float(ll.max()) + 0.5)
    rec.update(num_chains=4, num_samples=500, n_divergences=0, frozen_chains=[],
               prior_scan={"best_log_likelihood": float(ll.max()) + 0.5})
    rec.update(info)
    return _result("nuts_gpu", x, rec, max_ll=float(ll.max()))


def _says(res, **kw):
    return verdict_sentences(res, res.diagnostics(**kw))


def _row(res, check, **kw):
    return next(r for r in res.diagnostics(**kw).rows if r.check == check)


# --- 1. one sentence per failed check ---------------------------------------------------------

def test_a_healthy_ensemble_and_a_healthy_nuts_run_say_nothing():
    pytest.importorskip("arviz")
    assert _says(_ensemble()) == []
    assert _says(_nuts()) == []


def test_the_independent_runs_disagreeing():
    res = _ensemble(ensemble_rhat={"mstar": 1.18, "b": 1.002})
    assert _says(res) == [
        "The 4 ensembles disagree on mstar (R-hat 1.18, needs < 1.01). Run longer "
        "(max_steps=10000) or check for a second mode."]
    res = _ensemble(ensemble_rhat={"mstar": 1.012, "b": 1.002})
    assert "(R-hat 1.012, needs < 1.01)" in _says(res)[0]
    res = _ensemble(ensemble_rhat={"mstar": float("nan"), "b": 1.0})
    assert _says(res) == [
        "The 4 ensembles could not be compared on mstar (R-hat undefined, needs < 1.01). "
        "Run longer (max_steps=10000) and check that every ensemble moves."]


def test_the_chain_too_short_for_its_autocorrelation_time():
    res = _ensemble(max_autocorr_time=200.0, autocorr_time=[200.0, 30.0])
    assert _says(res) == [
        "The chain is too short for its autocorrelation time: 25.0 autocorrelation times long "
        "(largest tau 200 steps on mstar, needs at least 50). Run longer (max_steps=10000)."]
    res = _ensemble(max_autocorr_time=float("nan"))
    assert _says(res) == [
        "The chain is too short to estimate its autocorrelation time (needs at least 50 "
        "autocorrelation times). Run longer (max_steps=10000)."]


def test_stuck_and_frozen_walkers():
    assert _says(_ensemble(stuck_walkers=[5, 7])) == [
        "2 of 32 walkers are stuck more than 10 nats below the best walker (needs none), and "
        "their draws widen the posterior. Start the walkers around the likelihood peak "
        "(init=result.likelihood_max_opt(lc).params) or run longer."]
    assert _says(_ensemble(frozen_walkers=[3])) == [
        "1 of 32 walkers never moved across the kept draws (needs none). Run longer "
        "(max_steps=10000) or start the walkers closer to the posterior (init=)."]


def test_the_walkers_disagreeing():
    pytest.importorskip("arviz")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 400, 2))
    x[:8, :, 0] += 3.0                                      # 8 walkers in another place on mstar
    res = _ensemble(x)
    row = _row(res, "split R-hat across walkers (largest)")
    assert row.passed is False
    shown = f"{row.value:.2f}" if row.value >= 1.1 else f"{row.value:.3f}"
    assert (f"The walkers disagree on mstar (R-hat {shown}, needs < 1.05). Run longer "
            f"(max_steps=10000) or check for walkers left in a second mode.") in _says(res)


def test_too_few_independent_draws_bulk_and_tail():
    pytest.importorskip("arviz")
    rng = np.random.default_rng(2)
    x = rng.standard_normal((32, 80, 2))
    for t in range(1, 80):                                  # AR(1), phi 0.9: ESS about 130
        x[:, t] = 0.9 * x[:, t - 1] + np.sqrt(1 - 0.81) * x[:, t]
    res = _ensemble(x)
    said = _says(res)
    for kind, words in (("bulk", "of {p} for its median and width (bulk ESS"),
                        ("tail", "in the tails of {p} for its 16th and 84th percentiles (tail "
                                 "ESS")):
        row = _row(res, f"{kind} ESS (smallest)")
        assert row.passed is False
        p = row.reason.split("'")[1]
        target = math.ceil(5000 * 400 / row.value)
        expected = (f"There are too few independent draws {words.format(p=p)} {row.value:.0f}, "
                    f"needs at least 400). Run longer (max_steps={target}).")
        assert expected in said, (expected, said)


def test_draws_piled_against_a_prior_bound():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((32, 400, 2))
    flat = rng.uniform(0.1, 0.9, 32 * 400)
    flat[rng.permutation(flat.size)[: flat.size // 5]] = 0.9995       # exactly 20 %
    x[:, :, 0] = flat.reshape(32, 400)
    res = _ensemble(x)
    res.provenance = {"model": {"prior": R.prior_record(
        Prior({"mstar": Uniform(0.0, 1.0), "b": Uniform(-50.0, 50.0)}))}}
    assert _says(res) == [
        "20% of the mstar draws sit at its upper prior bound (1), needs at most 5% in the outer "
        "1% of the range, so the bound and not the data limits it. Widen the prior if the bound "
        "is not physical, or report a limit."]


def test_too_few_data_points_and_no_draws():
    res = _ensemble()
    res.n_data = 2
    assert _says(res) == [
        "There are only 2 data points for 2 free parameters (needs more points than "
        "parameters), so the posterior mostly restates the prior. Wait for more detections or "
        "fix some parameters."]
    empty = _ensemble(np.zeros((32, 0, 2)))
    assert ("The sampler returned no posterior draws (needs at least one). Check the start "
            "(init=) and the prior.") in _says(empty)


def test_the_best_draw_below_the_likelihood_maximum():
    res = _ensemble()
    assert _says(res, likelihood_max_opt=12.0) == [
        "The best draw is 12.0 nats below the optimised likelihood maximum (needs at most 5 "
        "nats), so the sampler never reached the peak. Start from the likelihood maximum (init=) "
        "and run again."]


def test_the_nuts_checks():
    pytest.importorskip("arviz")
    assert _says(_nuts(n_divergences=3)) == [
        "3 transitions diverged (needs 0), so the draws may be biased. Raise target_accept_prob "
        "or reparameterise."]
    assert _says(_nuts(n_divergences=-1)) == [
        "Divergent transitions were not recorded, so they cannot be ruled out (needs 0). Rerun "
        "with a sampler version that records them."]
    assert _says(_nuts(frozen_chains=[2])) == [
        "1 chain(s) (2) froze and barely move (needs none). Enable x64 or shift the clock."]
    res = _nuts()
    res.info["prior_scan"] = {"best_log_likelihood": res.max_log_likelihood + 25.0}
    assert _says(res) == [
        "Every draw is 25.0 nats below the prior scan's optimum (needs at most 10 nats), so all "
        "chains missed the best region. Start from init_strategy='prior_scan' or run longer."]
    shifted = _nuts(shift_chain=3)
    said = _says(shifted)
    rhat = _row(shifted, "split R-hat (largest)")
    rll = _row(shifted, "log-likelihood R-hat")
    fmt = (lambda v: f"{v:.2f}" if v >= 1.1 else f"{v:.3f}")
    assert (f"The 4 chains disagree on mstar (R-hat {fmt(rhat.value)}, needs < 1.01). Run longer "
            f"(num_samples=1000) or check for chains left in a second mode.") in said
    assert (f"The chains sit at different likelihood levels (log-likelihood R-hat "
            f"{fmt(rll.value)}, needs < 1.01). Run longer (num_samples=1000) or check for a "
            f"second mode.") in said
    assert ("1 chain(s) (3) are stranded more than 10 nats below the best chain (needs none). "
            "Start from init_strategy='prior_scan' or from a previous result.") in said
    one = _nuts(num_chains=1)
    one.samples_by_chain = one.samples_by_chain[:1]
    assert ("R-hat could not be computed because there is only one chain (needs at least 2). "
            "Run more chains.") in _says(one)


def test_an_unknown_check_still_gives_a_sentence():
    res = _ensemble()
    report = DiagnosticsReport("emcee_jax", "toy", "ensemble",
                               [DiagnosticRow("mystery check", 3, "0", False, "it is odd")])
    assert verdict_sentences(res, report) == ["Mystery check failed (it is odd)."]


# --- 2. the message ---------------------------------------------------------------------------

def test_the_message_is_not_converged_then_one_sentence_per_line():
    res = _ensemble(stuck_walkers=[5], ensemble_rhat={"mstar": 1.18, "b": 1.0})
    said = _says(res)
    msg = verdict_message(said)
    assert msg.splitlines() == ["Not converged:", *said] and len(said) == 2
    assert verdict_message([]) is None


def test_a_sampler_that_flags_itself_is_not_lost():
    res = _ensemble(converged=False, convergence_problems=["it wandered"])
    assert _says(res) == ["The sampler flagged a problem the checks did not catch: it wandered."]


# --- 3. and 4. wp.fit warns once, or not at all ------------------------------------------------

class _Prebuilt:
    """A registered sampler that returns a copy of a prepared result, and, like the real ones,
    warns itself when the result is flagged."""

    result = None

    def fit(self, lc, model, **kwargs):
        res = copy.deepcopy(type(self).result)
        if res.info.get("convergence_problems"):
            dg.warn_if_unconverged(res.sampler, res.info)
        return res


@pytest.fixture
def prebuilt():
    _SAMPLERS["prebuilt_for_test"] = _Prebuilt
    yield _Prebuilt
    _SAMPLERS.pop("prebuilt_for_test", None)


def _fit_caught(**kw):
    lc = wp.LightCurve(time=np.arange(5.0), band=["r"] * 5, flux=np.ones(5),
                       flux_err=np.ones(5))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = wp.fit(lc, "flare", sampler="prebuilt_for_test", **kw)
    return res, [w for w in caught if "onverged" in str(w.message)]


def test_fit_warns_once_with_the_plain_message_at_the_callers_line(prebuilt):
    prebuilt.result = _ensemble(ensemble_rhat={"mstar": 1.18, "b": 1.0}, converged=False,
                                convergence_problems=["the 4 ensembles disagree"])
    res, caught = _fit_caught()
    assert len(caught) == 1 and caught[0].category is UserWarning
    assert str(caught[0].message) == (
        "Not converged:\nThe 4 ensembles disagree on mstar (R-hat 1.18, needs < 1.01). "
        "Run longer (max_steps=10000) or check for a second mode.")
    assert caught[0].filename == __file__
    v = res.info["convergence_verdict"]
    assert v["passed"] is False and v["message"] == str(caught[0].message)
    assert v["failed_checks"] == ["R-hat between independent ensembles (largest)"]


def test_a_converged_fit_is_silent_and_the_opt_out_turns_everything_off(prebuilt):
    pytest.importorskip("arviz")
    prebuilt.result = _ensemble()
    res, caught = _fit_caught()
    assert caught == [] and res.info["convergence_verdict"]["passed"] is True
    assert res.info["convergence_verdict"]["message"] is None
    prebuilt.result = _ensemble(stuck_walkers=[1], converged=False,
                                convergence_problems=["1 of 32 walker(s) stuck"])
    res, caught = _fit_caught(check_convergence=False)
    assert caught == [] and "convergence_verdict" not in res.info


# --- 5. not MCMC ------------------------------------------------------------------------------

def test_a_fit_that_is_not_mcmc_gets_no_verdict_but_its_own_flag_still_warns(prebuilt):
    abc = SamplerResult("abc", "toy", ["a"], pd.DataFrame({"a": np.arange(500.0)}), {}, {}, 40,
                        1, 0.1, {"n_accepted": 500})
    prebuilt.result = abc
    res, caught = _fit_caught()
    assert caught == [] and "convergence_verdict" not in res.info
    flagged = copy.deepcopy(abc)
    flagged.info.update(converged=False, convergence_problems=["it wandered"])
    prebuilt.result = flagged
    res, caught = _fit_caught()
    assert len(caught) == 1 and "it wandered" in str(caught[0].message)


# --- 6. a posterior problem is not called non-convergence -------------------------------------

def _piled():
    """A healthy ensemble whose mstar draws pile 20 % at the upper bound of its prior."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((32, 400, 2))
    flat = rng.uniform(0.1, 0.9, 32 * 400)
    flat[rng.permutation(flat.size)[: flat.size // 5]] = 0.9995
    x[:, :, 0] = flat.reshape(32, 400)
    res = _ensemble(x)
    res.provenance = {"model": {"prior": R.prior_record(
        Prior({"mstar": Uniform(0.0, 1.0), "b": Uniform(-50.0, 50.0)}))}}
    return res


def test_only_a_posterior_problem_is_headed_check_the_posterior():
    pytest.importorskip("arviz")
    res = _piled()
    report = res.diagnostics()
    assert [r.check for r in report.rows if r.passed is False] == ["prior-edge pile-up"]
    assert verdict_heading(res, report) == "Check the posterior:"
    few = _ensemble()
    few.n_data = 2
    assert verdict_heading(few, few.diagnostics()) == "Check the posterior:"
    mixed = _piled()
    mixed.info["ensemble_rhat"] = {"mstar": 1.18, "b": 1.0}
    assert verdict_heading(mixed, mixed.diagnostics()) == "Not converged:"
    flagged = _piled()
    flagged.info.update(converged=False, convergence_problems=["it wandered"])
    assert verdict_heading(flagged, flagged.diagnostics()) == "Not converged:"
    assert verdict_heading(_ensemble(), _ensemble().diagnostics()) == "Not converged:"


def test_fit_warns_check_the_posterior_for_a_prior_edge_pile_up(prebuilt):
    pytest.importorskip("arviz")
    prebuilt.result = _piled()
    lc = wp.LightCurve(time=np.arange(5.0), band=["r"] * 5, flux=np.ones(5),
                       flux_err=np.ones(5))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = wp.fit(lc, "flare", sampler="prebuilt_for_test")
    ours = [str(w.message) for w in caught if "prior bound" in str(w.message)]
    assert len(ours) == 1 and ours[0].startswith("Check the posterior:\n20% of the mstar draws")
    assert not [w for w in caught if "Not converged" in str(w.message)]
    v = res.info["convergence_verdict"]
    assert v["passed"] is False and v["failed_checks"] == ["prior-edge pile-up"]


# --- 7. the advice names the cap that stopped the run ---------------------------------------------

def test_a_run_stopped_by_its_time_cap_is_told_to_raise_max_time_s():
    pytest.importorskip("arviz")
    timed = {"enabled": True, "steps_run": 5000, "stop_reason": "max_time_s", "max_steps": None,
             "chunks": [{"elapsed_s": 300.0}]}
    res = _ensemble(max_autocorr_time=200.0, autocorr_time=[200.0, 30.0], until_converged=timed)
    assert _says(res) == [
        "The chain is too short for its autocorrelation time: 25.0 autocorrelation times long "
        "(largest tau 200 steps on mstar, needs at least 50). Run longer (max_time_s=600)."]
    capped = dict(timed, max_steps=8000)                    # the 10 000-step target is past it
    res = _ensemble(max_autocorr_time=200.0, autocorr_time=[200.0, 30.0], until_converged=capped)
    assert _says(res)[0].endswith("Run longer (max_time_s=600, max_steps=10000).")
    res = _ensemble(ensemble_rhat={"mstar": 1.18, "b": 1.002},
                    until_converged=dict(timed, chunks=[]))           # no pace recorded
    assert _says(res)[0].endswith("Run longer (a larger max_time_s) or check for a second mode.")
    stepped = dict(timed, stop_reason="max_steps", max_steps=5000)
    res = _ensemble(max_autocorr_time=200.0, autocorr_time=[200.0, 30.0], until_converged=stepped)
    assert _says(res)[0].endswith("Run longer (max_steps=10000).")
