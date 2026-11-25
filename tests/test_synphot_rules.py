"""The band-integral rules of :mod:`whisper_cbpf.synphot`.

What the Gauss rule must be, checked on every filter the package ships:

1. **A Gaussian quadrature of T dlam/lam.** Positive weights, nodes inside the band, ``sum(w)``
   equal to ``int T dlam/lam`` to 1e-12, exact for every polynomial of degree <= 2N - 1.
2. **Accurate where it is used.** <= 0.02 mmag against the fine integral on blackbodies from
   300 K to 1e5 K and on 500 A-wide lines -- and the one measured weakness, a redback
   ``CutoffBlackbody`` kink inside the band, pinned at what it is (not what it was hoped to be).
3. **One object for both backends.** ``to_legacy`` hands the JAX kernels the same weights (round
   trip < 4.5e-16), the FilterSet pickles and saves by value with its hash, and the shipped sets
   ARE Gauss-16 on the installed sncosmo's curves.
4. **Labels mean one filter.** Bare letters are LSST (one warning per session), grouped labels
   raise, and the per-call / per-session overrides work.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

from whisper_cbpf import synphot
from whisper_cbpf.synphot import FilterSet, filter_set_for, gauss_rule, resolve_filter
from whisper_cbpf.synphot import labels as L
from whisper_cbpf.synphot.gauss_rule import _blackbody_fnu, _gauss_from_measure, fine_measure

SHIPPED = list(synphot.shipped_filter_names())


def _curve(name):
    sncosmo = pytest.importorskip("sncosmo")
    bp = sncosmo.get_bandpass(name)
    return np.asarray(bp.wave, dtype=float), np.asarray(bp.trans, dtype=float)


def _exact_int_T_over_lam(wave, trans):
    """Closed-form int T dlam/lam of the piecewise-linear curve (make_filter_set's formula)."""
    trans = np.clip(trans, 0.0, None)
    slope = np.diff(trans) / np.diff(wave)
    icpt = trans[:-1] - slope * wave[:-1]
    return float(np.sum(icpt * np.log(wave[1:] / wave[:-1]) + slope * np.diff(wave)))


# --- 1. a Gaussian quadrature of T dlam/lam ------------------------------------------------------

def test_shipped_sets_load_without_sncosmo_and_have_16_positive_nodes_inside_the_band():
    fs = filter_set_for(SHIPPED)
    assert fs.names == tuple(SHIPPED) and fs.rule == "gauss16"
    for name, x, w in zip(fs.names, fs.nodes, fs.weights):
        lo, hi = fs.provenance[name]["support"]
        assert x.size == 16 and np.all(w > 0), name
        assert lo < x.min() and x.max() < hi and np.all(np.diff(x) > 0), name


@pytest.mark.parametrize("name", SHIPPED)
def test_sum_of_weights_is_the_exact_filter_normalisation(name):
    wave, trans = _curve(name)
    w = filter_set_for([name]).weights[0]
    assert abs(w.sum() / _exact_int_T_over_lam(wave, trans) - 1.0) < 1e-12


@pytest.mark.parametrize("name", ["lsstu", "lssty", "ztfg", "sdssz"])
@pytest.mark.parametrize("n_nodes", [8, 16])
def test_exact_to_degree_2n_minus_1(name, n_nodes):
    """Every Legendre polynomial of degree <= 2N - 1 (in lam scaled to the band) is integrated to
    round-off; degree 2N is not, which shows the test can fail."""
    lam, m = fine_measure(*_curve(name))
    x, w = _gauss_from_measure(lam, m, n_nodes)
    c, h = 0.5 * (lam.min() + lam.max()), 0.5 * (lam.max() - lam.min())
    err = [abs(w @ P((x - c) / h) - m @ P((lam - c) / h)) / m.sum()
           for P in (np.polynomial.legendre.Legendre.basis(d) for d in range(2 * n_nodes + 1))]
    assert max(err[:-1]) < 1e-12
    assert err[-1] > 1e-6


# --- 2. accuracy against the fine integral ---------------------------------------------------------

@pytest.mark.parametrize("name", SHIPPED)
def test_within_0p02_mmag_of_the_fine_integral(name):
    """Blackbodies 300 K - 1e5 K (wherever the band flux is representable in float64) and a
    500 A-wide Gaussian line anywhere from 3000 to 11000 A (redback's type_1a line is 500 A)."""
    lam, m = fine_measure(*_curve(name))
    fs = filter_set_for([name])
    x, w = fs.nodes[0], fs.weights[0]
    temps = np.geomspace(300.0, 1e5, 60)
    fine, rule = m @ _blackbody_fnu(lam, temps), w @ _blackbody_fnu(x, temps)
    ok = (fine > 1e-300) & (rule > 1e-300)
    assert ok.sum() >= 40
    assert np.max(np.abs(2.5 * np.log10(rule[ok] / fine[ok]))) < 2e-5
    for centre in np.arange(3000.0, 11001.0, 250.0):
        line = lambda lam_: 1.0 + 3.0 * np.exp(-0.5 * ((lam_ - centre) / 500.0) ** 2)  # noqa: E731
        assert abs(2.5 * np.log10((w @ line(x) / w.sum()) / (m @ line(lam) / m.sum()))) < 2e-5


def test_a_cutoff_blackbody_kink_inside_the_band_is_the_rules_known_limit():
    """redback's ``CutoffBlackbody`` (slsn, general_magnetar_slsn, tde_analytical) multiplies the
    blackbody by (lam/lam_cut)^1 below ``lam_cut``: a KINK, which no polynomial rule integrates to
    round-off. Measured over every cutoff in 2500-7000 A and 5000-20000 K: Gauss-16 is off by up
    to 0.34 mmag in LSST g (1.1 mmag in SDSS u), Gauss-32 by 0.08 mmag -- still 1-2 orders below
    the monochromatic path it replaced. Pinned so a change in either direction is seen."""
    lam, m = fine_measure(*_curve("lsstg"))
    worst = {}
    for n in (16, 32):
        x, w = _gauss_from_measure(lam, m, n)
        err = 0.0
        for cut in np.arange(2500.0, 7000.0, 50.0):
            for temp in (5e3, 1e4, 2e4):
                sed = lambda lam_: (_blackbody_fnu(lam_, np.array([temp]))[:, 0]  # noqa: E731
                                    * np.where(lam_ < cut, lam_ / cut, 1.0))
                err = max(err, abs(2.5 * np.log10((w @ sed(x)) / (m @ sed(lam)))))
        worst[n] = err
    assert 1e-4 < worst[16] < 5e-4
    assert worst[32] < 1e-4


def test_self_check_warns_on_a_curve_the_rule_cannot_follow():
    """A 1000-30000 A top hat spans more blackbody curvature than 8 nodes follow (0.05 mmag; 16
    nodes: 3e-15 mag): the build-time self-check must say so, name the band, and the fix."""
    wave = np.array([999.0, 1000.0, 30000.0, 30001.0])
    trans = np.array([0.0, 1.0, 1.0, 0.0])
    with pytest.warns(RuntimeWarning, match="'white'.*n_nodes=16"):
        fs = gauss_rule(["white"], n_nodes=8, curves={"white": (wave, trans)})
    assert fs.provenance["white"]["selfcheck_max_mag"] > 2e-5
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gauss_rule(["lsstg"], curves={"lsstg": _curve("lsstg")})


# --- 3. one object for both backends -----------------------------------------------------------------

@pytest.fixture()
def _x64():
    jax = pytest.importorskip("jax")
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


def test_to_legacy_hands_the_jax_kernels_the_same_weights(_x64):
    from whisper_cbpf.synphot.grid_rule import ab_weights

    fs = filter_set_for(SHIPPED)
    leg = fs.to_legacy()
    assert leg["lam"].size == 16 * len(SHIPPED)
    w, norms = (np.asarray(a) for a in ab_weights(leg["lam"], leg["trans"]))
    for k, (x, wk) in enumerate(zip(fs.nodes, fs.weights)):
        i = np.searchsorted(leg["lam"], x)
        assert np.max(np.abs(w[k, i] - wk) / wk) < 4.5e-16
        assert np.count_nonzero(w[k]) == 16
    assert np.allclose(norms / 3631e-23, fs.norms, rtol=1e-15, atol=0)
    back = FilterSet.from_legacy(leg)
    assert all(np.array_equal(a, b) for a, b in zip(back.nodes, fs.nodes))


def test_filter_set_pickles_and_saves_with_its_hash(tmp_path):
    fs = filter_set_for(["lsstg", "ztfr", "sdssi"])
    back = pickle.loads(pickle.dumps(fs))
    assert back.hash == fs.hash and back == fs
    fs.save(tmp_path / "fs.npz")
    loaded = FilterSet.load(tmp_path / "fs.npz")
    assert loaded.hash == fs.hash and loaded.provenance == fs.provenance
    np.savez(tmp_path / "bad.npz", **{**dict(np.load(tmp_path / "fs.npz")),
                                      "weights_0": fs.weights[0] * (1 + 1e-12)})
    with pytest.raises(ValueError, match="hash"):
        FilterSet.load(tmp_path / "bad.npz")


def test_the_same_filter_set_on_cpu_and_gpu():
    """Both backends take the band integral from ``filter_set_for``; building it twice (in a
    worker process, on a GPU host) gives the same hash, because it is plain numpy."""
    a = filter_set_for(["lsstg", "lsstr"])
    b = gauss_rule(["lsstg", "lsstr"]) if _has("sncosmo") else a
    assert a.hash == filter_set_for(["lsstg", "lsstr"]).hash
    if b is not a:
        assert all(np.allclose(p, q, rtol=1e-13, atol=0) for p, q in zip(a.nodes, b.nodes))


def _has(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("stem", sorted(synphot.filterset.LIBRARY))
def test_shipped_sets_equal_gauss16_on_the_installed_sncosmo(stem):
    """The shipped files are what ``build_library`` makes from sncosmo: the same curves (sha256 of
    the float64 arrays) and the same nodes and weights to eigen-solver round-off."""
    pytest.importorskip("sncosmo")
    from whisper_cbpf.synphot.gauss_rule import curve

    shipped = FilterSet.load(synphot.filterset.LIBRARY_DIR / f"{stem}.npz")
    rebuilt = gauss_rule(shipped.names)
    for name in shipped.names:
        assert shipped.provenance[name]["sha256"] == curve(name)[2]["sha256"], name
        k = shipped.index(name)
        assert np.allclose(shipped.nodes[k], rebuilt.nodes[k], rtol=1e-13, atol=0), name
        assert np.allclose(shipped.weights[k], rebuilt.weights[k], rtol=1e-11, atol=0), name
        assert shipped.provenance[name]["release"], name


# --- 4. labels ---------------------------------------------------------------------------------------

@pytest.fixture
def fresh_labels(monkeypatch):
    monkeypatch.setattr(L, "_WARNED", False)
    monkeypatch.setattr(L, "_DEFAULT_SYSTEM", "lsst")
    monkeypatch.delenv(L.BAND_SYSTEM_ENV, raising=False)


def test_bare_letters_are_lsst_with_one_warning_per_session(fresh_labels):
    with pytest.warns(UserWarning, match="bare band letter 'g' read as LSST"):
        assert resolve_filter("g") == "lsstg"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert [resolve_filter(b) for b in "urizy"] == [f"lsst{b}" for b in "urizy"]


def test_the_bare_letter_system_can_change_per_session_and_per_call(fresh_labels):
    with warnings.catch_warnings():
        warnings.simplefilter("error")              # an explicit per-call choice needs no warning
        assert resolve_filter("g", default_system="sdss") == "sdssg"
        assert resolve_filter("g", aliases={"g": "ztfg"}) == "ztfg"
    import whisper_cbpf as wp
    old = wp.set_default_band_system("sdss")
    try:
        assert old == "lsst" and wp.default_band_system() == "sdss"
        with pytest.warns(UserWarning, match="read as SDSS"):
            assert resolve_filter("r") == "sdssr"
    finally:
        wp.set_default_band_system(old)
    with pytest.raises(ValueError, match="no 'u' filter"):
        resolve_filter("u", default_system="ztf")
    with pytest.raises(ValueError, match="unknown band system"):
        wp.set_default_band_system("hst")


def test_the_session_system_reaches_spawned_workers(fresh_labels):
    """The CPU samplers start workers with ``spawn``: a fresh interpreter. It must read bare letters
    the way the parent does, or the workers fit a different filter from the one the parent reports."""
    import multiprocessing as mp

    import whisper_cbpf as wp
    old = wp.set_default_band_system("sdss")
    try:
        with mp.get_context("spawn").Pool(1) as pool:
            assert pool.apply(_worker_resolves, ("g",)) == "sdssg"
    finally:
        wp.set_default_band_system(old)


def _worker_resolves(label):
    import warnings as w
    w.simplefilter("ignore")
    return resolve_filter(label)


def test_survey_labels_and_friendly_names_resolve():
    assert [resolve_filter(b) for b in ("ztfg", "zg", "ZTF_r", "lssty", "sdssu")] == \
        ["ztfg", "ztfg", "ztfr", "lssty", "sdssu"]
    assert resolve_filter("LSST/LSST.g") == "LSST/LSST.g"               # an SVO ID
    pytest.importorskip("redback")
    pytest.importorskip("sncosmo")
    assert [resolve_filter(b) for b in ("B", "J", "Ks", "uvot::uvw1")] == \
        ["bessellb", "2massj", "2massks", "uvot::uvw1"]


def test_grouped_and_curveless_labels_raise():
    with pytest.raises(ValueError, match="grouped effective-band label"):
        resolve_filter("g-band")
    with pytest.raises(ValueError, match="names no filter"):
        resolve_filter("not_a_band_xyz")
    pytest.importorskip("redback")
    with pytest.raises(ValueError, match="sncosmo has no transmission curve"):
        resolve_filter("wise::W1")          # in redback's filters.csv, not in sncosmo
