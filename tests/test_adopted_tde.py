"""The two redback TDE models bound in ``tde_fallback.py`` and ``tde_analytical.py``.

Six claims, each with a number (plus the priors: redback's file, transcribed for use without
redback, and ``tde_analytical``'s one deliberate difference from it, ``kappa`` LogUniform):

1. **Agreement with redback.** ``tde_analytical`` is redback's to float64 round-off, because it is
   assembled from the same kernel on the same grid. ``tde_fallback``'s engine is MOSFiT's
   ``Fallback`` node for node, which redback transcribes: its tables are redback's arrays bit for
   bit and its luminosity before the viscous delay is redback's ``_tde_mosfit_engine`` to 1.64e-13
   (relative, the largest over 634 sets). What is left is the viscous kernel's interpolation of
   log10 L between the dense nodes (redback interpolates L): about 2e-4 dex at most in the
   luminosity (maxima 2.2e-4 over 60 prior draws x 60 epochs, 2.6e-4 over 200 x 701) and 8e-4 mag
   in band magnitudes (7.9e-4 over 300 draws).
   Photometry is redback's ``flux_density`` output through whisper's own band integral on both
   sides, so only the models differ.
2. **Agreement with redback-jax**: the stellar radius is the same; the fall-back engine and the
   viscous bolometric light curve now differ by redback-jax's own resampling of the tables (512
   equal phase steps), by exactly as much as redback-jax differs from redback. Skipped without
   redback-jax, and run in a subprocess because importing it turns on ``jax_enable_x64`` for the
   whole process.
3. **It is differentiable and traceable** under ``jit`` and ``vmap`` on prior draws, with the
   epochs, ``t_exp`` and the redshift as ordinary arguments.
4. **float32** works for the log10 bolometric engine of ``tde_fallback`` (error in dex below) and
   fails loudly, naming float64, for every photometric function.
5. **Known answers**: the viscous kernel on a constant source, the late-time ``t^-5/3`` decline and
   the half-star fall-back mass of a full disruption, the Eddington cap, and the diffusion kernel's
   late-time limit.
6. **The engine's parts** (tested with claim 1): the offline node builder against MOSFiT's
   ``__init__`` (as redback runs it), the pair selection and MOSFiT's equality branch at simulated
   betas (also under ``jit``, where XLA would otherwise fuse the beta mapping into one rounding),
   the gamma mapping on padded curves, the floor outside the nodes, and finite gradients across
   the jump at simulated betas.

FLOAT64. The photometric functions require it; the fixture below enables it for this module and
restores the previous setting afterwards.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from whisper_cbpf.models.jax import tde_analytical as TA  # noqa: E402
from whisper_cbpf.models.jax import tde_fallback as TF  # noqa: E402
from whisper_cbpf.priors import Prior, Uniform  # noqa: E402

Z = 0.05
#: LSST g, r, i, z and ZTF g, r.
BANDS = ["lsstg", "lsstr", "lssti", "lsstz", "ztfg", "ztfr"]
N_DRAWS = 60
SEED = 20261001
#: Planck18 luminosity distance at z = 0.05 [cm], the one redback's TDE models use.
DL = 7.093375016731583e26
T_FALLBACK = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0])
T_ANALYTICAL = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
MSUN = 1.988409870698051e33
CLIGHT = 2.99792458e10


@pytest.fixture(scope="module", autouse=True)
def _x64():
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


def _draws(model, n, seed=SEED):
    rng = np.random.default_rng(seed)
    return [model.default_prior.sample(rng) for _ in range(n)]


def _layout(epochs):
    """Every epoch in every band, time-ordered (the last observation is the latest)."""
    return np.repeat(epochs, len(BANDS)), np.tile(np.array(BANDS), epochs.size)


def _redback_mags(model_call, t, b):
    """redback's ``flux_density`` through whisper's band integral: AB magnitudes."""
    from whisper_cbpf.synphot import band_flux_jy, filter_set_for

    fjy = band_flux_jy(lambda tt, nu: np.asarray(model_call(tt, nu)) * 1e-3, t, b,
                       filter_set_for(BANDS))
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(fjy / 3631.0)


def _mags(model, p, t, b):
    return -2.5 * np.log10(np.asarray(model.predict(p, t, b)) / 3631.0)


def _stats(x):
    x = np.abs(np.asarray(x))
    return float(np.median(x)), float(np.percentile(x, 99)), float(np.max(x))


# =======================================================================================
# 1. agreement with redback
# =======================================================================================

def test_tde_fallback_photometry_matches_redback():
    """60 prior draws, LSST griz and ZTF gr at z = 0.05. Measured over these 60: median 1.7e-5
    mag, 99th percentile 3.2e-4, maximum 4.5e-4; over 300 draws median 2.2e-5, p99 3.7e-4,
    maximum 7.9e-4 (the viscous kernel's log10 interpolation of L; with the 512-step resampled
    tables of 0.3.0 the 60 draws gave 4.7e-5 / 6.8e-3 / 1.5e-2 mag)."""
    pytest.importorskip("redback")
    from redback.transient_models import tde_models as ot

    from whisper_cbpf.models.redback_adapter import redback_luminosity_distance_cm

    dl = redback_luminosity_distance_cm(Z, "tde_fallback")
    model = TF.tde_fallback_model(BANDS, Z, dl)
    t, b = _layout(T_FALLBACK)
    err = []
    for p in _draws(model, N_DRAWS):
        ref = _redback_mags(
            lambda tt, nu, p=p: ot.tde_fallback(
                tt, Z, p["mbh6"], p["mstar"], p["tvisc"], p["bb"], p["eta"], 1.0, p["rph0"],
                p["lphoto"], output_format="flux_density", frequency=nu), t, b)
        err.append(_mags(model, p, t, b) - ref)
    med, p99, mx = _stats(err)
    print(f"\ntde_fallback vs redback, {N_DRAWS} draws x {t.size} points: "
          f"median {med:.2e}, p99 {p99:.2e}, max {mx:.2e} mag")
    assert med < 1e-4 and p99 < 2e-3 and mx < 3e-3, (med, p99, mx)


def test_tde_analytical_photometry_matches_redback():
    """Same epochs' kernel and grid as redback: the two agree to float64 round-off (measured
    2e-14 mag over 100 draws); the budget is ten orders above that."""
    pytest.importorskip("redback")
    from redback.transient_models import tde_models as ot

    from whisper_cbpf.models.redback_adapter import redback_luminosity_distance_cm

    dl = redback_luminosity_distance_cm(Z, "tde_analytical")
    model = TA.tde_analytical_model(BANDS, Z, dl, constraint=None)
    t, b = _layout(T_ANALYTICAL)
    err, n_bright = [], 0
    for p in _draws(model, N_DRAWS):
        ref = _redback_mags(
            lambda tt, nu, p=p: ot.tde_analytical(
                tt, Z, p["l0"], p["t_0_turn"], kappa=p["kappa"], kappa_gamma=p["kappa_gamma"],
                mej=p["mej"], vej=p["vej"], temperature_floor=p["temperature_floor"],
                output_format="flux_density", frequency=nu), t, b)
        mine = _mags(model, p, t, b)
        ok = np.isfinite(ref) & (ref < 35.0)
        n_bright += int(ok.sum())
        err.append(np.where(ok, mine - ref, 0.0))
    assert n_bright > 0.5 * N_DRAWS * t.size, "too few finite points: the test cannot fail"
    assert np.max(np.abs(err)) < 1e-9, _stats(err)


def test_tde_analytical_bolometric_is_redbacks():
    pytest.importorskip("redback")
    from redback.transient_models import tde_models as ot

    from whisper_cbpf.models.jax import supernova as S

    t = np.geomspace(0.3, 300.0, 50)
    grid = S.build_sn_grid(t)
    model = TA.tde_analytical_model(BANDS, Z, DL, constraint=None)
    worst = 0.0
    for p in _draws(model, 40, SEED + 1):
        ref = ot.tde_analytical_bolometric(t, p["l0"], p["t_0_turn"], kappa=p["kappa"],
                                           kappa_gamma=p["kappa_gamma"], mej=p["mej"],
                                           vej=p["vej"])
        mine = np.asarray(TA.tde_analytical_bolometric(grid, p["l0"], p["t_0_turn"], p["kappa"],
                                                       p["kappa_gamma"], p["mej"], p["vej"]))
        ok = ref > 1e30
        assert ok.sum() > 10
        worst = max(worst, float(np.max(np.abs(np.log10(mine[ok]) - np.log10(ref[ok])))))
    assert worst < 1e-10, worst


def test_tde_fallback_bolometric_matches_redback():
    """Over 60 draws x 60 epochs (0.5 to 1000 d): median 1.1e-5 dex, 99th percentile 1.4e-4,
    maximum 2.2e-4, all of it the viscous kernel's log10 interpolation of L between the dense
    nodes (redback's own kernel on this engine gives 2e-14). With the 512-step resampled tables of
    0.3.0: 5.4e-5 / 0.014 / 0.021."""
    pytest.importorskip("redback")
    from redback.transient_models import tde_models as ot

    t = np.geomspace(0.5, 1000.0, 60)
    model = TF.tde_fallback_model(BANDS, Z, DL)
    err = []
    for p in _draws(model, N_DRAWS, SEED + 2):
        ref = ot.tde_fallback_bolometric(t, p["mbh6"], p["mstar"], p["tvisc"], p["bb"], p["eta"],
                                         1.0)
        mine = np.asarray(TF.tde_fallback_bolometric_log10(
            t, p["mbh6"], p["mstar"], p["tvisc"], p["bb"], p["eta"], 1.0))
        ok = ref > 1e30
        err.append((mine - np.log10(np.maximum(ref, 1e-300)))[ok])
    med, p99, mx = _stats(np.concatenate(err))
    print(f"\ntde_fallback_bolometric vs redback: median {med:.2e}, p99 {p99:.2e}, max {mx:.2e} dex")
    assert med < 5e-5 and p99 < 1e-3 and mx < 1e-3, (med, p99, mx)


def test_tde_fallback_photosphere_inputs_are_redbacks():
    """The stellar radius, the time of the peak rate and the penetration factor that redback hands
    to its photosphere (measured: relative 1.4e-15)."""
    pytest.importorskip("redback")
    from redback.transient_models import tde_models as ot

    dense = np.geomspace(1e-5, 1100.0, 1000)
    model = TF.tde_fallback_model(BANDS, Z, DL)
    for p in _draws(model, 30, SEED + 3):
        o = ot._tde_mosfit_engine(dense, p["mbh6"], p["mstar"], p["bb"], p["eta"], 1.0)
        _, rstar, t_peak, beta = TF.fallback_engine(jnp.asarray(dense), p["mbh6"], p["mstar"],
                                                    p["bb"], p["eta"], 1.0)
        assert float(rstar) == pytest.approx(o.Rstar, rel=1e-12)
        assert float(t_peak) == pytest.approx(o.tpeak, rel=1e-12)
        assert float(beta) == pytest.approx(o.beta, rel=1e-12)


@pytest.mark.parametrize("module", [TF, TA])
def test_transcribed_prior_is_redbacks_prior_file(module):
    """The prior used without redback is redback 1.20's file, parameter for parameter."""
    pytest.importorskip("redback")
    rb, rb_pinned = getattr(module, "redback_default_prior", module.default_prior)()
    tr, tr_pinned = module.fallback_prior()
    assert {k: repr(v) for k, v in rb.distributions.items()} == \
        {k: repr(v) for k, v in tr.distributions.items()}
    assert rb_pinned == tr_pinned
    assert set(rb.distributions) | set(rb_pinned) == set(module.PARAMETERS)


def test_default_prior_falls_back_to_the_transcription_without_redback(monkeypatch):
    from whisper_cbpf.models.jax import tde

    def no_redback(model_name, **kw):
        raise ImportError("redback is not installed")

    monkeypatch.setattr(tde, "redback_prior", no_redback)
    for module in (TF, TA):
        prior, pinned = module.default_prior()
        assert set(prior.distributions) | set(pinned) == set(module.PARAMETERS)
    # without redback the analytical default still differs from redback's in kappa alone
    assert repr(TA.default_prior()[0].distributions["kappa"]) == "LogUniform(0.05, 100.0)"
    assert repr(TA.redback_default_prior()[0].distributions["kappa"]) == "Uniform(0.05, 100.0)"


def test_analytical_default_prior_is_redbacks_with_kappa_log_uniform():
    """whisper's ``tde_analytical`` default differs from redback's prior in one place, on purpose:
    ``kappa`` LogUniform over redback's bounds (0.05, 100) instead of Uniform; every other
    distribution is redback's, and every other scale parameter spanning two decades or more
    (``mej``, ``vej``, ``kappa_gamma``, ``l0``, ``t_0_turn``) is LogUniform in redback's file."""
    ours, ours_pinned = TA.default_prior()
    rb, rb_pinned = TA.redback_default_prior()
    assert TA.LOG_UNIFORM_OVER_REDBACK == ("kappa",)
    assert ours_pinned == rb_pinned == {}
    k_ours, k_rb = ours.distributions["kappa"], rb.distributions["kappa"]
    assert type(k_ours).__name__ == "LogUniform" and type(k_rb).__name__ == "Uniform"
    assert k_ours.bounds == k_rb.bounds == (0.05, 100.0)
    others = set(rb.distributions) - {"kappa"}
    assert {k: repr(ours.distributions[k]) for k in others} == \
        {k: repr(rb.distributions[k]) for k in others}
    for k in ("mej", "vej", "kappa_gamma", "l0", "t_0_turn"):
        d = rb.distributions[k]
        assert type(d).__name__ == "LogUniform" and np.log10(d.high / d.low) >= 2.0, k
    # the point of the change: prior mass below 0.5 cm^2/g, 0.45 % under redback's Uniform
    rng = np.random.default_rng(SEED)
    below = np.mean([k_ours.sample(rng) < 0.5 for _ in range(4000)])
    assert below == pytest.approx(np.log(10.0) / np.log(2000.0), abs=0.03)       # 0.303
    assert (0.5 - 0.05) / (100.0 - 0.05) == pytest.approx(0.0045, abs=1e-4)


def test_analytical_model_prior_default_and_redback():
    """The factory's default prior has ``kappa`` LogUniform and says so; ``prior="redback"`` gives
    redback's prior unchanged; an unknown string is refused, naming the choices."""
    ours = TA.tde_analytical_model(BANDS, Z, DL)
    rb = TA.tde_analytical_model(BANDS, Z, DL, prior="redback")
    assert repr(ours.default_prior.distributions["kappa"]) == "LogUniform(0.05, 100.0)"
    assert repr(rb.default_prior.distributions["kappa"]) == "Uniform(0.05, 100.0)"
    assert ours.parameters == rb.parameters == list(TA.PARAMETERS)
    want = TA.redback_default_prior()[0].distributions
    assert {k: repr(v) for k, v in rb.default_prior.distributions.items()} == \
        {k: repr(v) for k, v in want.items()}
    assert "kappa prior LogUniform" in ours.description
    assert "kappa prior LogUniform" not in rb.description
    # a caller's own kappa wins over both
    own = TA.tde_analytical_model(BANDS, Z, DL, prior=Prior({"kappa": Uniform(0.1, 1.0)}))
    assert repr(own.default_prior.distributions["kappa"]) == "Uniform(0.1, 1.0)"
    # a pinned kappa has no prior, so the description does not claim one
    pinned = TA.tde_analytical_model(BANDS, Z, DL, pin={"kappa": 0.2})
    assert "kappa" not in pinned.parameters and "kappa prior" not in pinned.description
    # redback's prior with a free t_exp: merge its distributions with the t_exp one
    free = TA.tde_analytical_model(
        BANDS, Z, DL, free=["t_exp"],
        prior=Prior({**TA.redback_default_prior()[0].distributions,
                     "t_exp": Uniform(-20.0, 0.0)}))
    assert repr(free.default_prior.distributions["kappa"]) == "Uniform(0.05, 100.0)"
    assert free.parameters[-1] == "t_exp"
    with pytest.raises(ValueError, match="prior='bogus' is not understood.*'redback'"):
        TA.tde_analytical_model(BANDS, Z, DL, prior="bogus")
    assert TA._resolve_prior(" Redback ", "x").distributions.keys() == want.keys()
    assert TA._resolve_prior(None, "x") is None


def test_compare_binds_analytical_with_kappa_log_uniform():
    """``compare``'s by-name binding of ``"tde_analytical"`` gets the factory default (``kappa``
    LogUniform); redback's prior, passed as a ``Prior``, gives ``kappa`` Uniform."""
    import whisper_cbpf as wp
    import importlib

    C = importlib.import_module("whisper_cbpf.compare")

    t = 60000.0 + np.array([-3.0, 2.0, 5.0, 9.0, 14.0, 20.0])
    lc = wp.LightCurve(time=t, band=np.array(["lsstg", "lsstr"] * 3),
                       magnitude=np.array([23.8, 20.0, 19.5, 19.4, 19.8, 20.3]),
                       magnitude_err=np.r_[np.nan, np.full(5, 0.05)],
                       upper_limit=np.r_[True, np.zeros(5, bool)], redshift=Z, name="tde")
    m = C._bind_family("tde_analytical", lc, None)
    assert repr(m.default_prior.distributions["kappa"]) == "LogUniform(0.05, 100.0)"
    rb = C._bind_family("tde_analytical", lc, TA.redback_default_prior()[0])
    assert repr(rb.default_prior.distributions["kappa"]) == "Uniform(0.05, 100.0)"
    assert "t_exp" in rb.parameters


def _guillochon_dir():
    redback = pytest.importorskip("redback")
    src = Path(redback.__file__).parent / "tables" / "guillochon_tde_data"
    if not src.is_dir():
        pytest.skip("redback's Guillochon tables are not installed")
    return src


def test_shipped_nodes_are_build_fallback_tables_of_the_guillochon_files():
    """``build_fallback_tables`` on the files redback distributes gives the shipped arrays, exactly
    (1.3 MB), padded to 166 rising and 925 falling nodes."""
    built = TF.build_fallback_tables(_guillochon_dir())
    with np.load(TF.TABLE_FILE) as shipped:
        assert set(built) == set(shipped.files)
        for k, v in built.items():
            assert np.array_equal(v, shipped[k]), k
    assert TF.TABLE_FILE.name == "tde_fallback_nodes.npz"
    assert TF.TABLE_FILE.stat().st_size < 2_000_000
    tables = TF.fallback_tables()
    for gamma, n in (("4-3", 23), ("5-3", 20)):
        tab = tables[gamma]
        assert tab["sim_beta"].shape == (n,)
        assert tab["premaptime"][0].shape == (n, 166) and tab["premaptime"][1].shape == (n, 925)
        assert tab["beta_slope"][1].shape == (n - 1, 925)
        assert tab["premap_count"].shape == (n, 2) and tab["mapped_count"].shape == (n - 1, 2)
    prov = json.loads(str(built["provenance"]))
    assert len(prov["files_sha256"]["4-3"]) == 23 and len(prov["files_sha256"]["5-3"]) == 20


def test_build_fallback_tables_is_mosfits_init():
    """MOSFiT's ``Fallback.__init__`` as redback runs it (``_initialize_mosfit_tde_model``, the same
    code): every simulation's branches, and per pair of neighbouring betas the mapped times, slopes
    and intercepts, bit for bit (measured: identical), with the padding a repeat of the last node."""
    pytest.importorskip("redback")
    from redback.transient_models import tde_models as ot

    rb = ot._initialize_mosfit_tde_model()
    tables = TF.fallback_tables()
    for gamma in ("4-3", "5-3"):
        tab = tables[gamma]
        assert np.array_equal(tab["sim_beta"], np.asarray(rb.sim_beta[gamma]))
        for name, count in (("premaptime", "premap_count"), ("premapdmdt", "premap_count"),
                            ("mapped_time", "mapped_count"), ("beta_slope", "mapped_count"),
                            ("beta_yinter", "mapped_count")):
            for j in (0, 1):
                rows = tab[name][j]
                for i, theirs in enumerate(getattr(rb, name)[gamma]):
                    c = int(tab[count][i, j])
                    theirs = np.asarray(theirs[j], dtype=float)
                    assert c == theirs.size, (gamma, name, i, j)
                    assert np.array_equal(rows[i, :c], theirs), (gamma, name, i, j)
                    assert np.all(rows[i, c:] == rows[i, c - 1])


def _gate_sets():
    """(mbh6, mstar, bb, eta, leddlimit) for the engine checks: prior-like draws, every gamma regime
    and its limits, stars below 0.1 solar masses, bb = 0, 1, 2."""
    rng = np.random.default_rng(SEED + 7)
    sets = [(rng.uniform(.01, 20), rng.uniform(.1, 30), rng.uniform(0, 2),
             np.exp(rng.uniform(np.log(1e-4), np.log(.4))), 1.0) for _ in range(24)]
    for m in (0.05, 0.2, 0.3, 0.5, 1.0, 5.0, 15.0, 18.0, 22.0, 25.0):
        for bb in (0.0, 0.37, 1.0, 1.61, 2.0):
            sets.append((1.0, m, bb, 0.1, 1.0))
    return sets


def test_engine_is_redbacks_tde_mosfit_engine():
    """The un-viscous luminosity, the stellar radius, ``t_peak`` and ``beta`` against redback's
    ``_tde_mosfit_engine`` on the engine's dense grid, with epochs before 0, at 0 (the first node,
    which redback and MOSFiT include) and after the last node: relative 1.64e-13 at worst over 634
    parameter sets when this was written; where redback's luminosity is exactly 0 the port is at
    its 1e-30 erg/s floor."""
    pytest.importorskip("redback")
    from redback.transient_models import tde_models as ot

    times = np.concatenate(([-1.0, 0.0], np.geomspace(1e-5, 1100.0, 1000), [1e7, 1e9]))
    eng = jax.jit(lambda t, p: TF.fallback_engine(t, p[0], p[1], p[2], p[3], p[4]))
    worst = 0.0
    n_zero = 0
    for p in _gate_sets():
        o = ot._tde_mosfit_engine(times, *p)
        ref = np.asarray(o.luminosities)
        assert ref[0] == 0.0 and ref[1] > 0.0
        log_l, rstar, t_peak, beta = (np.asarray(a) for a in eng(jnp.asarray(times), jnp.asarray(p)))
        pos = ref > 0
        worst = max(worst, float(np.max(np.abs(10.0 ** (log_l[pos] - np.log10(ref[pos])) - 1.0))))
        assert np.all(log_l[~pos] == -30.0)
        n_zero += int((~pos).sum())
        assert float(rstar) == pytest.approx(o.Rstar, rel=1e-13)
        assert float(t_peak) == pytest.approx(o.tpeak, rel=1e-13)
        assert float(beta) == pytest.approx(o.beta, rel=1e-13)
    assert n_zero >= 2 * len(_gate_sets())          # the epoch at 0 and the late ones were tested
    assert worst < 1e-11, worst


def _bb_giving(beta_sim, gamma):
    """A float ``bb`` in [0, 2] at which MOSFiT's own expression gives exactly ``beta_sim``."""
    if gamma == "4-3":
        f = lambda b: 0.6 + 1.25 * b if b < 1 else 1.85 + 2.15 * (b - 1)
        b0 = (beta_sim - 0.6) / 1.25 if beta_sim < 1.85 else 1.0 + (beta_sim - 1.85) / 2.15
    else:
        f = lambda b: 0.5 + 0.4 * b if b < 1 else 0.9 + 1.6 * (b - 1)
        b0 = (beta_sim - 0.5) / 0.4 if beta_sim < 0.9 else 1.0 + (beta_sim - 0.9) / 1.6
    near = [b0]
    for direction in (3.0, -1.0):
        b = b0
        for _ in range(8):
            b = float(np.nextafter(b, direction))
            near.append(b)
    for b in near:
        if 0.0 <= b <= 2.0 and f(b) == beta_sim:
            return b
    return None


def _mosfit_beta_step(beta, gamma):
    """MOSFiT's search loop and beta step (fallback.py:312-372), literally, on the shipped tables:
    per branch ``(log10 t, log10 dM/dt)`` without padding."""
    tab = TF.fallback_tables()[gamma]
    sim = list(tab["sim_beta"])
    for i in range(len(sim)):
        if beta == sim[i]:
            c = tab["premap_count"][i]
            return [(tab["premaptime"][j][i, :c[j]], tab["premapdmdt"][j][i, :c[j]]) for j in (0, 1)]
        if beta < sim[i]:
            hi, lo = i, i - 1
            break
    out = []
    for j in (0, 1):
        c = tab["mapped_count"][lo][j]
        mt = tab["mapped_time"][j][lo, :c]
        tl = tab["premaptime"][j][lo, :tab["premap_count"][lo][j]]
        th = tab["premaptime"][j][hi, :tab["premap_count"][hi][j]]
        t_lo = mt * (tl[-1] - tl[0]) + tl[0]
        t_hi = mt * (th[-1] - th[0]) + th[0]
        out.append((t_lo + (t_hi - t_lo) * (beta - sim[lo]) / (sim[hi] - sim[lo]),
                    tab["beta_yinter"][j][lo, :c] + tab["beta_slope"][j][lo, :c] * beta))
    return out


@pytest.mark.parametrize("gamma", ["4-3", "5-3"])
def test_pair_selection_and_the_equality_branch(gamma):
    """``_beta_nodes`` is MOSFiT's loop: between two simulated betas their pair (searchsorted), at a
    simulated beta that simulation's own nodes, under ``jit`` too (19 of 23 and 18 of 20 simulated
    betas are reached exactly by a float ``bb``). Every node to 1e-12, and the count of valid
    nodes is MOSFiT's array length."""
    sims = TF.fallback_tables()[gamma]["sim_beta"]
    rng = np.random.default_rng(3)
    betas = list(sims) + list(rng.uniform(sims[0], sims[-1], 40)) + \
        [float(np.nextafter(s, 9.0)) for s in sims[:-1]] + [float(np.nextafter(s, 0.0)) for s in sims[1:]]
    step = jax.jit(lambda b: TF._beta_nodes(b, gamma))
    for beta in betas:
        want = _mosfit_beta_step(float(beta), gamma)
        got = step(jnp.asarray(beta))
        for (wt, wd), (gt, gd, gc) in zip(want, got):
            assert int(gc) == wt.size
            gt, gd = np.asarray(gt), np.asarray(gd)
            assert np.allclose(gt[:wt.size], wt, rtol=0, atol=1e-12)
            assert np.all(gt[wt.size:] == gt[wt.size - 1])
            assert np.allclose(gd[:wd.size], wd, rtol=0, atol=1e-12)
    # the equality branch is reached from bb, as MOSFiT reaches it, also under jit (no fused
    # multiply-add in the mapping)
    beta_of = jax.jit(lambda b: TF._fallback_nodes(1.0, 2.0 if gamma == "4-3" else 0.2, b)[5])
    reached = 0
    for s in sims:
        bb = _bb_giving(float(s), gamma)
        if bb is None:
            continue
        reached += 1
        assert float(beta_of(jnp.asarray(bb))) == float(s), (s, bb)
    assert reached == (19 if gamma == "4-3" else 18)


def test_the_rate_jumps_at_a_simulated_beta_as_mosfits_does():
    """At ``bb`` giving exactly a simulated beta the engine takes that simulation's own nodes, one
    unit in the last place away the pair's mapped nodes: the luminosity before the peak jumps (0.014
    dex at the 5/3 simulation 0.85, as in MOSFiT and redback) and the gradient in ``bb`` is finite
    on both sides and at the point itself."""
    pytest.importorskip("redback")
    from redback.transient_models import tde_models as ot

    bb = _bb_giving(0.85, "5-3")
    t = np.geomspace(1e-3, 300.0, 400)
    eng = jax.jit(lambda b: TF.fallback_engine(jnp.asarray(t), 1.0, 0.2, b, 0.1, 1.0)[0])
    at, above = np.asarray(eng(bb)), np.asarray(eng(float(np.nextafter(bb, 3.0))))
    jump = np.max(np.abs(at - above))
    assert 0.01 < jump < 0.02, jump
    for b in (bb, float(np.nextafter(bb, 3.0))):
        ref = np.asarray(ot._tde_mosfit_engine(t, 1.0, 0.2, b, 0.1, 1.0).luminosities)
        mine = 10.0 ** np.asarray(eng(b))
        assert np.allclose(mine[ref > 0], ref[ref > 0], rtol=1e-11, atol=0)
    grad = jax.jit(jax.grad(lambda b: jnp.sum(TF.fallback_engine(jnp.asarray(t), 1.0, 0.2, b, 0.1,
                                                                 1.0)[0])))
    for b in (bb, float(np.nextafter(bb, 3.0)), float(np.nextafter(bb, -1.0)), bb + 1e-3, bb - 1e-3):
        assert np.isfinite(float(grad(b)))


def test_engine_gradients_are_finite_on_prior_draws_and_at_the_edges():
    """``jit(vmap(grad))`` of the summed log10 luminosity in all five engine arguments, on 48 prior
    draws and at the edges: ``bb`` = 0, 1, 2 and exactly at simulated betas (where the rate jumps,
    so the gradient in ``bb`` is piecewise: finiteness is the claim, not smoothness), stars on the
    gamma limits 0.3, 1, 15, 22 and below 0.1 solar masses. Every gradient finite, none all zero."""
    rng = np.random.default_rng(SEED + 9)
    rows = [(rng.uniform(.01, 20), rng.uniform(.1, 30), rng.uniform(0, 2),
             np.exp(rng.uniform(np.log(1e-4), np.log(.4))), 1.0) for _ in range(48)]
    edge_bb = [0.0, 1.0, 2.0, _bb_giving(0.85, "5-3"), _bb_giving(1.5, "4-3")]
    rows += [(1.0, m, bb, 0.1, 1.0) for m in (0.05, 0.3, 0.6, 1.0, 15.0, 18.0, 22.0) for bb in edge_bb]
    dense = jnp.geomspace(1e-5, 1100.0, 1000)

    def total(p):
        return jnp.sum(TF.fallback_engine(dense, p[0], p[1], p[2], p[3], p[4])[0])

    grad = np.asarray(jax.jit(jax.vmap(jax.grad(total)))(jnp.asarray(rows)))
    assert np.isfinite(grad).all(), np.argwhere(~np.isfinite(grad))
    assert (np.abs(grad).max(axis=1) > 0).all()


def test_the_compiled_engine_does_not_grow_with_the_batch():
    """The final interpolation's search leaves XLA no constant of batch x epochs entries under
    ``jit(vmap)``. With ``jnp.searchsorted(method="scan_unrolled")`` every query starts from the
    same bounds, so XLA folded the first step's gather indices into one: on 1000 epochs the largest
    constant of the optimised program had 128000 entries at 64 parameter sets against 21275 at 4,
    and on a GPU the generated code grew from 2.0 MB at one set to 18.4 MB at 2048. Here the largest
    constant is the same at 4 and at 64 sets."""
    dense = jnp.geomspace(1e-5, 1100.0, 1000)
    rng = np.random.default_rng(SEED + 11)

    def largest_constant(n):
        p = jnp.asarray(np.stack([rng.uniform(.01, 20, n), rng.uniform(.1, 30, n),
                                  rng.uniform(0, 2, n), np.full(n, 0.1), np.ones(n)], 1))
        hlo = jax.jit(jax.vmap(lambda q: TF.fallback_engine(dense, *q)[0])).lower(p).compile().as_text()
        shapes = re.findall(r"= [a-z]+\d*\[([\d,]*)\]\S* constant\(", hlo)
        return max(int(np.prod([int(d) for d in s.split(",") if d])) for s in shapes)

    small, large = largest_constant(4), largest_constant(64)
    assert large == small, (small, large)
    assert large < 64 * 1000, large


def _mosfit_gamma_blend(c43, c53, gfrac):
    """MOSFiT's gamma blend (fallback.py:376-440), literally, on unpadded branches; log10 out."""
    out = []
    for (t43, d43), (t53, d53) in zip(c43, c53):
        td = {"4-3": t43, "5-3": t53}
        dd = {"4-3": d43.copy(), "5-3": d53.copy()}
        interp, nointerp = ("4-3", "5-3") if len(t43) < len(t53) else ("5-3", "4-3")
        m = {}
        for k in (nointerp, interp):
            m[k] = 1. / (td[k][-1] - td[k][0]) * (td[k] - td[k][0])
            m[k][0], m[k][-1] = 0, 1
        dd[interp] = np.interp(m[nointerp], m[interp], dd[interp])
        if interp == "5-3":
            t53m = m["4-3"] * (t53[-1] - t53[0]) + t53[0]
            time = t43 + (t53m - t43) * gfrac
        else:
            t43m = m["5-3"] * (t43[-1] - t43[0]) + t43[0]
            time = t43m + (t53 - t43m) * gfrac
        out.append((time, dd["4-3"] + (dd["5-3"] - dd["4-3"]) * gfrac))
    return out


@pytest.mark.parametrize("b43,b53,gfrac", [(1.85, 0.9, 0.4), (0.7, 0.53, 0.9), (3.3, 2.2, 0.15),
                                           (1.37, 0.71, 0.5), (0.6, 0.5, 0.0), (4.0, 2.5, 1.0)])
def test_gamma_mapping_on_padded_curves(b43, b53, gfrac):
    """``_gamma_blend`` at the padded size is MOSFiT's blend of the two curves: the curve with fewer
    nodes is mapped onto the other's (chosen by a ``where`` on the two counts, both ways covered
    here), and the result has the denser curve's count, to 1e-12 in log t and log dM/dt."""
    c43 = TF._beta_nodes(jnp.asarray(b43), "4-3")
    c53 = TF._beta_nodes(jnp.asarray(b53), "5-3")
    want = _mosfit_gamma_blend(_mosfit_beta_step(b43, "4-3"), _mosfit_beta_step(b53, "5-3"), gfrac)
    got = jax.jit(TF._gamma_blend)(c43, c53, jnp.asarray(gfrac))
    for (wt, wd), (gt, gd, gc) in zip(want, got):
        assert int(gc) == wt.size
        assert np.allclose(np.asarray(gt)[:wt.size], wt, rtol=0, atol=1e-12)
        assert np.allclose(np.asarray(gd)[:wd.size], wd, rtol=0, atol=1e-12)
    counts = [(int(a[2]), int(b[2])) for a, b in zip(c43, c53)]
    assert all(int(g[2]) == max(a, b) for g, (a, b) in zip(got, counts))


# =======================================================================================
# 2. agreement with redback-jax
# =======================================================================================

_REDBACK_JAX_SCRIPT = textwrap.dedent("""
    import json
    import numpy as np
    import redback_jax.models.tde_models as J          # turns on jax_enable_x64
    import jax.numpy as jnp
    from whisper_cbpf.models.jax import tde_fallback as F
    try:
        from redback.transient_models import tde_models as R
    except ImportError:
        R = None

    rng = np.random.default_rng(5)
    n = 60
    P = np.stack([rng.uniform(.01, 20, n), rng.uniform(.1, 30, n),
                  np.exp(rng.uniform(np.log(1e-3), np.log(1e2), n)), rng.uniform(0, 2, n),
                  np.exp(rng.uniform(np.log(1e-4), np.log(.4), n)), np.ones(n)], 1)
    t = np.geomspace(0.5, 1000, 60)
    dense = np.geomspace(1e-5, 1100., 1000)
    out = {"radius": max(abs(float(F.stellar_radius(m)) / float(J._stellar_radius_tout(m)) - 1)
                         for m in P[:, 1])}
    eng, bol, own = [], [], []
    for p in P:
        a = np.asarray(F.fallback_engine(jnp.asarray(dense), p[0], p[1], p[3], p[4], p[5])[0])
        b = np.asarray(J._tde_fallback_engine_log10(jnp.asarray(dense), p[0], p[1], p[3], p[4], p[5]))
        ok = np.isfinite(b) & (b > -30) & (a > -30)
        eng.append(np.abs(a - b)[ok])
        if R is not None:     # redback-jax's own difference from redback, epoch by epoch
            r = np.asarray(R._tde_mosfit_engine(dense, p[0], p[1], p[3], p[4], p[5]).luminosities)
            own.append(np.abs(np.abs(a - b) - np.abs(b - np.log10(np.where(r > 0, r, 1.0))))[ok])
        a = np.asarray(F.tde_fallback_bolometric_log10(jnp.asarray(t), *p))
        b = np.asarray(J.tde_fallback_bolometric(jnp.asarray(t), *p))
        ok = np.isfinite(b) & (b > -20)
        bol.append(np.abs(a - b)[ok])
    eng, bol = np.concatenate(eng), np.concatenate(bol)
    out.update(engine_median=float(np.median(eng)), engine_max=float(eng.max()),
               bol_median=float(np.median(bol)), bol_max=float(bol.max()),
               own=float(np.concatenate(own).max()) if own else None)
    print("RESULT" + json.dumps(out))
""")


def test_tde_fallback_engine_matches_redback_jax():
    """Same radius and viscous convolution; the engine is MOSFiT's here and redback-jax resamples
    the tables onto 512 equal phase steps, so the two differ by exactly redback-jax's own
    difference from redback (to 2e-14 dex, epoch by epoch). Measured over 60 draws (redback-jax
    0.4.1): radius identical; engine median 6.0e-5, maximum 2.4e-2 dex; bolometric median 4.7e-5,
    maximum 2.3e-2 dex (with the resampled tables of 0.3.0 the two agreed to 6e-14 and 9e-6).
    The claim is the comparison with redback, so the test needs redback as well; the bounds on the
    difference itself are about 3 times the measured maxima."""
    if importlib.util.find_spec("redback_jax") is None:
        pytest.skip("redback_jax is not installed")
    if importlib.util.find_spec("redback") is None:
        pytest.skip("redback is not installed: the difference is checked against redback's own")
    run = subprocess.run([sys.executable, "-c", _REDBACK_JAX_SCRIPT], capture_output=True,
                         text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    out = json.loads(run.stdout.split("RESULT", 1)[1])
    assert out["radius"] < 1e-12
    assert out["engine_median"] < 3e-4 and out["engine_max"] < 0.08, out
    assert out["bol_median"] < 2e-4 and out["bol_max"] < 0.07, out
    assert out["own"] is not None and out["own"] < 1e-10, out


# =======================================================================================
# 3. traceable and differentiable
# =======================================================================================

def _models():
    prior = Prior({"t_exp": Uniform(-20.0, -1.0)})
    zprior = {"type": "Uniform", "low": 0.02, "high": 0.2}
    return {
        "tde_fallback": TF.tde_fallback_model(BANDS, Z, DL),
        "tde_analytical": TA.tde_analytical_model(BANDS, Z, DL, constraint=None),
        "tde_fallback_free": TF.tde_fallback_model(BANDS, free=["t_exp", "redshift"], prior=prior,
                                                   redshift_prior=zprior),
        "tde_analytical_free": TA.tde_analytical_model(BANDS, free=["t_exp", "redshift"],
                                                       constraint=None, prior=prior,
                                                       redshift_prior=zprior),
    }


@pytest.mark.parametrize("key", ["tde_fallback", "tde_analytical", "tde_fallback_free",
                                 "tde_analytical_free"])
def test_jit_vmap_grad_are_finite_on_prior_draws(key):
    """32 prior draws through ``jit(vmap(...))`` and its gradient, with epochs before the free
    ``t_exp`` of some draws (those are at ``mag_floor``): every flux and every gradient finite."""
    model = _models()[key]
    t = np.repeat(np.array([-3.0, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0]), len(BANDS))
    b = np.tile(np.array(BANDS), 8)
    pj = model.predict_jax
    bidx = pj.band_index(b)
    theta = np.array([[p[k] for k in model.parameters] for p in _draws(model, 32, SEED + 4)])

    flux = np.asarray(jax.jit(jax.vmap(lambda x: pj(x, t, bidx)))(jnp.asarray(theta)))

    def total_mag(x):
        return jnp.sum(-2.5 * jnp.log10(jnp.maximum(pj(x, t, bidx), 1e-300) / 3631.0))

    grad = np.asarray(jax.jit(jax.vmap(jax.grad(total_mag)))(jnp.asarray(theta)))
    floor = 3631.0 * 10.0 ** (-0.4 * pj.mag_floor)
    assert np.isfinite(flux).all()
    assert np.isfinite(grad).all(), f"{int((~np.isfinite(grad).all(1)).sum())} draws with NaN grad"
    assert (flux > 2.0 * floor).mean() > 0.5            # mostly real light, not the floor
    assert np.abs(grad).max() > 0                       # not a dead model
    t_exp = theta[:, model.parameters.index("t_exp")] if "t_exp" in model.parameters else 0.0
    dead = np.broadcast_to(t_exp >= -3.0, (theta.shape[0],))   # the epoch at -3 d is not after t_exp
    assert dead.any() and np.allclose(flux[dead][:, :len(BANDS)], floor)


def test_epochs_before_the_time_zero_are_at_mag_floor():
    for key in ("tde_fallback", "tde_analytical"):
        model = _models()[key]
        p = _draws(model, 1)[0]
        t = np.array([-5.0, 0.0, 30.0])
        flux = model.predict(p, t, np.array(["lsstg"] * 3))
        floor = 3631.0 * 10.0 ** (-0.4 * 40.0)
        assert flux[0] == pytest.approx(floor) and flux[1] == pytest.approx(floor)
        assert flux[2] > floor


def test_explosion_time_and_redshift_are_traced_and_shift_the_curve():
    """``t_exp`` shifts the epochs and the redshift scales them and the distance, in one compiled
    function: a draw with ``t_exp = 10`` at epoch ``t`` is the draw with ``t_exp = 0`` at ``t - 10``."""
    prior = Prior({"t_exp": Uniform(-20.0, 20.0)})
    for factory in (TF.tde_fallback_model, TA.tde_analytical_model):
        kw = dict(constraint=None) if factory is TA.tde_analytical_model else {}
        free = factory(BANDS, free=["t_exp"], prior=prior, redshift=Z, dl_cm=DL, **kw)
        fixed = factory(BANDS, redshift=Z, dl_cm=DL, **kw)
        p = _draws(fixed, 1)[0]
        t = np.array([15.0, 40.0, 90.0, 200.0])
        b = np.array(["ztfr"] * 4)
        a = free.predict({**p, "t_exp": 10.0}, t, b)
        c = fixed.predict(p, t - 10.0, b)
        assert np.allclose(a, c, rtol=1e-8)


def test_epochs_may_be_traced():
    """The epochs are an ordinary argument: ``jit`` over them gives the concrete answer."""
    for model in (_models()["tde_fallback"], _models()["tde_analytical"]):
        p = _draws(model, 1, SEED + 6)[0]
        theta = jnp.asarray([p[k] for k in model.parameters])
        t = np.array([4.0, 9.0, 33.0, 120.0])
        bidx = model.predict_jax.band_index(np.array(["lsstg", "lsstr", "lssti", "ztfr"]))
        traced = jax.jit(lambda tt: model.predict_jax(theta, tt, bidx))(jnp.asarray(t))
        assert np.allclose(np.asarray(traced), model.predict_jax(theta, t, bidx), rtol=1e-10)


def test_wall_and_pickling():
    """``tde_analytical`` is walled as redback is (zero flux, ``constraint_ok``), ``constraint=None``
    is the bare physics, and the model pickles (the multiprocess samplers need it)."""
    walled = TA.tde_analytical_model(BANDS, Z, DL)
    bare = TA.tde_analytical_model(BANDS, Z, DL, constraint=None)
    assert walled.predict_jax.constraint_ok is not None and bare.predict_jax.constraint_ok is None
    t, b = np.array([5.0, 20.0]), np.array(["lsstg", "lsstr"])
    # kinetic energy 1e48 erg < 1e51, nebular time 440 d: inside both bounds
    inside = dict(l0=1e55, t_0_turn=1.0, mej=0.1, vej=1e3, kappa=0.3, kappa_gamma=1.0,
                  temperature_floor=5000.0)
    outside = dict(inside, kappa=100.0)                  # nebular time 8000 d
    for p, ok in ((inside, True), (outside, False)):
        theta = jnp.asarray([p[k] for k in walled.parameters])
        assert bool(walled.predict_jax.constraint_ok(theta)) is ok
        assert walled.predict.__self__.physical([p[k] for k in walled.parameters]) is ok
        flux = walled.predict(p, t, b)
        assert (flux > 3631.0 * 1e-16).all() == ok and (ok or (flux == 0).all())
        assert np.all(bare.predict(p, t, b) > 0)         # constraint=None: the physics, always
    assert TF.tde_fallback_model(BANDS, Z, DL).predict_jax.constraint_ok is None
    for m in (walled, TF.tde_fallback_model(BANDS, Z, DL)):
        assert pickle.loads(pickle.dumps(m.predict)).__self__.params == m.parameters


def test_registration_and_parameters():
    from whisper_cbpf import models as M

    prior = Prior({"t_exp": Uniform(-20.0, 0.0)})
    try:
        f = TF.register_tde_fallback(BANDS, Z, DL, name="_adopted_tde_fallback")
        a = TA.register_tde_analytical(BANDS, Z, DL, name="_adopted_tde_analytical")
        assert {"_adopted_tde_fallback", "_adopted_tde_analytical"} <= set(M.list_models())
        assert f.parameters == ["mbh6", "mstar", "tvisc", "bb", "eta", "rph0", "lphoto"]
        assert a.parameters == ["l0", "t_0_turn", "mej", "vej", "kappa", "kappa_gamma",
                                "temperature_floor"]
        assert f.default_prior.distributions.keys() == set(f.parameters)
        assert a.default_prior.distributions.keys() == set(a.parameters)
        assert "float64" in f.description and "float64" in a.description
    finally:
        for k in ("_adopted_tde_fallback", "_adopted_tde_analytical"):
            M._REGISTRY.pop(k, None)
    free = TF.tde_fallback_model(BANDS, free=["t_exp", "redshift"], prior=prior)
    assert free.parameters[-2:] == ["t_exp", "redshift"]
    pinned = TF.tde_fallback_model(BANDS, Z, DL, pin={"bb": 1.0})
    assert "bb" not in pinned.parameters
    with pytest.raises(ValueError, match="not a parameter"):
        TF.tde_fallback_model(BANDS, Z, DL, pin={"nonsense": 1.0})
    with pytest.raises(ValueError, match="redshift"):
        TF.tde_fallback_model(BANDS)                  # neither fixed nor free


# =======================================================================================
# 4. float32
# =======================================================================================

def test_float32_bolometric_engine_is_usable_and_photometry_raises():
    """The log10 bolometric light curve in float32 vs float64 over 60 prior draws x 60 epochs:
    median 2.0e-6 dex, maximum 1.2e-4; the un-viscous engine alone on its dense grid: median
    2.6e-6, maximum 2.8e-5 dex, with the same epochs inside the nodes. The photometric chain
    carries L in erg/s, past float32's ceiling, so it raises and names float64 rather than
    returning every epoch at ``mag_floor``."""
    rng = np.random.default_rng(2)
    n = 60
    P = np.stack([rng.uniform(.01, 20, n), rng.uniform(.1, 30, n),
                  np.exp(rng.uniform(np.log(1e-3), np.log(1e2), n)), rng.uniform(0, 2, n),
                  np.exp(rng.uniform(np.log(1e-4), np.log(.4), n)), np.ones(n)], 1)
    t = np.geomspace(0.5, 1000.0, 60)

    dense = np.geomspace(1e-5, 1100.0, 1000)

    def engine(dtype):
        f = jax.jit(jax.vmap(lambda p: TF.tde_fallback_bolometric_log10(
            jnp.asarray(t, dtype=dtype), *[p[j] for j in range(6)])))
        return f(jnp.asarray(P, dtype=dtype))

    def unviscous(dtype):
        f = jax.jit(jax.vmap(lambda p: TF.fallback_engine(
            jnp.asarray(dense, dtype=dtype), p[0], p[1], p[3], p[4], p[5])[0]))
        return f(jnp.asarray(P, dtype=dtype))

    ref = np.asarray(engine(np.float64))
    ref_u = np.asarray(unviscous(np.float64))
    models = {"tde_fallback": TF.tde_fallback_model(["lsstg"], Z, DL),
              "tde_analytical": TA.tde_analytical_model(["lsstg"], Z, DL, constraint=None)}
    jax.config.update("jax_enable_x64", False)
    try:
        jax.clear_caches()
        got = engine(np.float32)
        assert got.dtype == np.float32 and np.isfinite(np.asarray(got)).all()
        diff = np.abs(np.asarray(got, dtype=float) - ref)
        assert diff.max() < 5e-4 and np.median(diff) < 1e-5, (diff.max(), np.median(diff))
        got_u = np.asarray(unviscous(np.float32), dtype=float)
        assert np.array_equal(got_u > -29, ref_u > -29)
        du = np.abs(got_u - ref_u)[ref_u > -29]
        assert du.max() < 1e-4 and np.median(du) < 1e-5, (du.max(), np.median(du))
        for name, model in models.items():
            p = model.default_prior.sample(rng)
            with pytest.raises(RuntimeError, match="float64"):
                model.predict(p, np.array([5.0, 10.0]), np.array(["lsstg", "lsstg"]))
    finally:
        jax.config.update("jax_enable_x64", True)
        jax.clear_caches()


# =======================================================================================
# 5. known answers
# =======================================================================================

def test_viscous_kernel_on_a_constant_source():
    """A constant luminosity through ``L(t) = int L exp(-(t - t')/tvisc) dt' / tvisc`` from the
    first node: ``L (1 - exp(-t/tvisc))``."""
    dense = jnp.geomspace(1e-5, 1100.0, 1000)
    log_l = jnp.full(dense.shape, 44.0)
    tvisc = 7.0
    t = jnp.array([0.5, 3.5, 7.0, 21.0, 70.0, 300.0])
    got = np.asarray(10.0 ** TF.viscous_log10(t, dense, log_l, tvisc))
    want = 1e44 * (1.0 - np.exp(-np.asarray(t) / tvisc))
    assert np.allclose(got, want, rtol=2e-3), got / want


@pytest.mark.parametrize("mstar,bb", [(1.0, 1.9), (0.5, 1.5), (10.0, 1.5)])
def test_late_time_fallback_rate_follows_t_to_the_minus_five_thirds(mstar, bb):
    """The rate of a full disruption declines as ``t^-5/3`` at late times (measured slope between
    1e3 and 1e4 days: -1.74, -1.64, -1.71) and returns about half the star (0.49, 0.50, 0.49)."""
    ts = np.geomspace(1e-5, 1e6, 6000)
    log_l, _, _, _ = TF.fallback_engine(jnp.asarray(ts), 1.0, mstar, bb, 0.1, 1e8)   # no cap
    log_l = np.asarray(log_l)
    sel = (ts > 1e3) & (ts < 1e4)
    slope = np.polyfit(np.log10(ts[sel]), log_l[sel], 1)[0]
    assert abs(slope + 5.0 / 3.0) < 0.1, slope
    assert 0.45 < _fallback_mass(1.0, mstar, bb) / mstar < 0.55


def _fallback_mass(mbh6, mstar, bb):
    """Mass returned to pericentre [solar masses]: the integral of the uncapped ``L / (eta c^2)``."""
    ts = np.geomspace(1e-5, 1e6, 6000)
    log_l = np.asarray(TF.fallback_engine(jnp.asarray(ts), mbh6, mstar, bb, 0.1, 1e8)[0])
    ok = log_l > -29.0
    return np.trapezoid(10.0 ** log_l[ok] / (0.1 * CLIGHT ** 2), ts[ok] * 86400.0) / MSUN


def test_partial_disruption_returns_less_mass():
    """``bb = 0.5`` strips a fraction of the star (measured 0.12 of it), far below the half of a
    full disruption (0.49 at ``bb = 1.5``)."""
    assert _fallback_mass(1.0, 1.0, 0.5) < 0.25 < 0.4 < _fallback_mass(1.0, 1.0, 1.5)


def test_eddington_cap_holds():
    """The harmonic cap keeps ``L`` below ``leddlimit * Ledd`` at every time, even for the largest
    efficiency (measured ``log10(L / Ledd)`` at most -2.6e-5)."""
    ts = np.geomspace(1e-5, 1e6, 6000)
    ledd_per_msun = 4.0 * np.pi * 6.6743e-8 * MSUN * CLIGHT / (0.2 * 1.74)
    for mbh6 in (0.1, 1.0, 10.0):
        log_l = np.asarray(TF.fallback_engine(jnp.asarray(ts), mbh6, 1.0, 1.0, 0.4, 1.0)[0])
        assert (log_l - np.log10(ledd_per_msun * mbh6 * 1e6)).max() < 0.0


def test_diffusion_returns_the_fallback_at_late_times():
    """With gamma-rays trapped (huge ``kappa_gamma``) and ``t`` far past the diffusion time
    (2.5 d here), the Arnett kernel returns the engine's own ``l0 / t^(5/3)``. redback's
    100-node trapezoid, which this kernel reproduces, is 1.1% high at every epoch here; the
    model's decline is the engine's slope to 0.02."""
    from whisper_cbpf.models.jax import supernova as S
    from whisper_cbpf.models.jax.tde import analytic_fallback

    l0, t0 = 1e55, 1e-3
    t = np.array([100.0, 200.0, 300.0, 400.0])
    grid = S.build_sn_grid(t)
    out = np.asarray(TA.tde_analytical_bolometric(grid, l0, t0, 0.05, 1e4, 0.1, 1e4))
    want = np.asarray(analytic_fallback(jnp.asarray(t), l0, t0))
    assert np.allclose(out, want, rtol=0.02), out / want
    slope = np.polyfit(np.log10(t), np.log10(out), 1)[0]
    assert abs(slope + 5.0 / 3.0) < 0.02, slope
