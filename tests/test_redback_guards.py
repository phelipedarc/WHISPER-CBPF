"""The redback adapter's configuration guards.

1.4 **Out-of-domain epochs are never silent.** Up to whisper 0.1.0 an epoch redback refused came
    back as exactly 0 Jy with no word, so AT2017GFO's two-component kilonova lost every epoch past
    redback's 6-day grid (18 of 211 at the published parameters, chi2/N ~ 1.2e7, AIC ~ 2.5e9 in
    every CPU sampler). Now:

    - a limit that is the same for every draw (a *model* limit, not a draw to reject) raises -- at
      registration when ``times=`` is given, else at the first ``predict`` on those epochs;
    - a parameter-dependent one (a TDE envelope that dies inside the window) still returns zero
      flux for the refused epochs, and warns once per model and span;
    - any exception redback raises rejects the draw (zero flux) instead of aborting the run, where
      only ``ValueError``/``IndexError`` were caught.

1.5 **A bolometric engine is refused.** redback's ``*_bolometric`` functions take ``**kwargs``, so
    ``output_format`` and ``frequency`` were swallowed and their erg/s fitted as Jy (median 1.2e37
    "Jy"). Now they raise at registration, naming the photometric wrapper; so does any model whose
    output does not change with frequency. ``redshift=`` goes through the same membership check as
    ``pin=``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from whisper_cbpf.models import redback_adapter as RA

KN2 = "two_component_kilonova_model"
Z_KN = 0.0098
#: Villar+2017's two-component AT2017GFO solution in redback's parameter names.
VILLAR = dict(mej_1=0.023, vej_1=0.256, temperature_floor_1=3983.0, kappa_1=0.5,
              mej_2=0.050, vej_2=0.149, temperature_floor_2=1151.0, kappa_2=3.65)
T_KN = np.linspace(0.5, 12.5, 25)                          # the ceiling is 6(1+z) = 6.06 d
B_KN = np.array([["ztfg", "ztfr", "ztfi"][i % 3] for i in range(T_KN.size)])


# --- 1.4: parameter-independent domain limits are configuration errors ---------------------------

def test_a_limit_every_draw_shares_raises_at_the_first_predict():
    """The bug: 13 of these 25 epochs silently came back as 0 Jy at every parameter set."""
    pytest.importorskip("redback")
    m = RA.redback_model(KN2, ["ztfg", "ztfr", "ztfi"], redshift=Z_KN)
    with pytest.raises(ValueError, match=r"every one of .* prior draws") as err:
        m.predict(VILLAR, T_KN, B_KN)
    msg = str(err.value)                    # names the span, the count and redback's own reason
    assert "0.5-6 d" in msg and "13 of 25" in msg and "518400" in msg, msg


def test_a_limit_every_draw_shares_raises_at_registration_with_times():
    pytest.importorskip("redback")
    with pytest.raises(ValueError, match=r"every one of .* prior draws"):
        RA.redback_model(KN2, ["ztfg", "ztfr", "ztfi"], redshift=Z_KN, times=T_KN)
    inside = T_KN[T_KN < 6.0]
    m = RA.redback_model(KN2, ["ztfg", "ztfr", "ztfi"], redshift=Z_KN, times=inside)
    flux = m.predict(VILLAR, inside, B_KN[:inside.size])
    assert np.all(flux > 0)


def test_times_given_to_the_registration_probe_need_band_names():
    pytest.importorskip("redback")
    with pytest.raises(ValueError, match="band_names"):
        RA.redback_model("arnett", redshift=0.05, times=[1.0, 2.0])


def _domain_ends_at_t_end(time, t_end, **kwargs):
    """A stand-in redback model whose domain ends at a PARAMETER, the way a TDE envelope's did
    (redback <= 1.15 raised past it; 1.20's ``cooling_envelope`` returns exact zeros there)."""
    if np.max(time) > t_end:
        raise ValueError(f"A value ({np.max(time)}) is above the grid's end ({t_end})")
    return 1e-3 * np.asarray(kwargs["frequency"], dtype=float) / 5e14 * np.ones_like(time)


def test_a_parameter_dependent_limit_zeroes_the_draw_and_warns_once_per_span(monkeypatch):
    """Its own epochs stay, the rest are zero, and the user is told once -- not on every
    likelihood call. The probe sees the limit move with ``t_end`` and does not raise."""
    from whisper_cbpf.priors import Uniform

    monkeypatch.setitem(RA._MODEL_CACHE, "_test_t_end", _domain_ends_at_t_end)
    monkeypatch.setitem(RA._PRIOR_CACHE, "_test_t_end", ({}, {}, {}))
    m = RA.redback_model("_test_t_end", ["ztfg"], prior={"t_end": Uniform(2.0, 20.0)})
    t = np.array([1.0, 3.0, 7.0])
    b = np.array(["ztfg"] * 3)
    getattr(RA, "_WARNED_SPANS", set()).clear()          # (absent before whisper 0.1.1)
    with pytest.warns(UserWarning, match=r"_test_t_end.*1 of 3 epochs \(7-7 d\).*1-3 d") as rec:
        out = m.predict({"t_end": 5.0}, t, b)
    assert out[0] > 0 and out[1] > 0 and out[2] == 0.0
    assert sum("epochs" in str(w.message) for w in rec) == 1
    with warnings.catch_warnings(record=True) as again:     # the same span again: silent
        warnings.simplefilter("always")
        m.predict({"t_end": 5.0}, t, b)
    assert not any("epochs" in str(w.message) for w in again)
    with pytest.warns(UserWarning, match=r"2 of 3 epochs"):  # a different span: told again
        m.predict({"t_end": 2.5}, t, b)


def test_whisper_warnings_survive_importing_redback():
    """redback's ``result.py`` runs ``warnings.simplefilter("ignore")`` on import,
    which silenced every warning above. The adapter imports redback and takes that filter out."""
    import subprocess
    import sys

    pytest.importorskip("redback")
    code = ("import warnings\n"
            "from whisper_cbpf.models import redback_adapter as RA\n"
            "RA._import_redback()\n"
            "import redback\n"
            "blanket = ('ignore', None, Warning, None, 0)\n"
            "assert blanket not in warnings.filters, warnings.filters[:3]\n"
            "with warnings.catch_warnings(record=True) as seen:\n"
            "    warnings.warn('still visible', UserWarning)\n"
            "assert [str(w.message) for w in seen] == ['still visible'], seen\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=600)
    assert run.returncode == 0, run.stderr[-1500:]


def _run_clean(code):
    """``python -c code`` with no ``PYTHONWARNINGS``, so only the filters under test apply."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=600, env=env)


def test_a_whisper_warning_after_registering_a_redback_model_reaches_stderr():
    """As a user meets it: bind a redback model (which imports redback), then do
    something whisper warns about. Up to whisper 0.1.0 the warning never printed -- nor did any of
    the package's others, the SNPE fallback notice included (0 times in 46 run logs where 39 runs
    fell back)."""
    pytest.importorskip("redback")
    run = _run_clean("import whisper_cbpf as wp\n"
                     "wp.register_redback('arnett', redshift=0.05)\n"
                     "wp.resolve_band('zzz_not_a_band', svo_fallback=False)\n")
    assert run.returncode == 0, run.stderr[-1500:]
    assert "Band 'zzz_not_a_band' is not in FILTER_LOOKUP" in run.stderr, run.stderr[-1500:]


def test_importing_redback_restores_the_callers_own_warning_filters():
    """``_import_redback`` snapshots ``warnings.filters`` and restores it. redback's
    ``simplefilter("ignore")`` MOVES an existing blanket filter to the front, so a caller who
    silences everything but ``UserWarning`` lost the exception too, even where the blanket was
    their own; only the filters the imported libraries register for their own warnings are kept."""
    pytest.importorskip("redback")
    run = _run_clean("import warnings\n"
                     "warnings.simplefilter('ignore')\n"
                     "warnings.filterwarnings('always', category=UserWarning)\n"
                     "mine = list(warnings.filters)\n"
                     "import whisper_cbpf as wp\n"
                     "wp.register_redback('arnett', redshift=0.05)\n"
                     "assert warnings.filters[-len(mine):] == mine, warnings.filters[:4]\n"
                     "wp.resolve_band('zzz_not_a_band', svo_fallback=False)\n")
    assert run.returncode == 0, run.stderr[-1500:]
    assert "Band 'zzz_not_a_band' is not in FILTER_LOOKUP" in run.stderr, run.stderr[-1500:]


# --- 1.4: any redback exception rejects the draw -------------------------------------------------

def _boom(time, **kwargs):
    raise RuntimeError("a redback failure that is not ValueError/IndexError")


def _late_boom(time, frequency=None, **kwargs):
    if np.max(time) > 5.0:
        raise ZeroDivisionError("past 5 d")
    return np.ones_like(np.asarray(time, dtype=float)) * 1e-3


def test_any_redback_exception_rejects_the_draw_instead_of_aborting(monkeypatch):
    """The bug: only ValueError/IndexError were caught, so any other redback failure on one draw
    aborted the whole run. Two stand-in redback models, looked up by name like any other."""
    monkeypatch.setitem(RA._MODEL_CACHE, "_test_boom", _boom)
    monkeypatch.setitem(RA._MODEL_CACHE, "_test_late_boom", _late_boom)
    t = np.array([1.0, 3.0, 7.0])
    b = np.array(["ztfg", "ztfr", "ztfi"])
    assert np.all(RA.redback_flux_jy("_test_boom", {}, t, b) == 0.0)
    out = RA.redback_flux_jy("_test_late_boom", {}, t, b)
    assert np.allclose(out, [1e-6, 1e-6, 0.0], rtol=1e-12)


# --- 1.5: bolometric engines and frequency-independent outputs are refused -----------------------

def test_a_bolometric_engine_is_refused_naming_its_photometric_wrapper():
    """The bug: this registered an 11-parameter model whose predictions are erg/s read as Jy."""
    pytest.importorskip("redback")
    with pytest.raises(ValueError, match=r"not a flux density.*'shock_cooling_and_arnett'"):
        RA.redback_model("shock_cooling_and_arnett_bolometric", ["ztfg", "ztfr"], redshift=0.05)
    with pytest.raises(ValueError, match=r"not a flux density.*'csm_shock_and_arnett'"):
        RA.register_redback("csm_shock_and_arnett_bolometric", ["ztfg"], overwrite=True)


def test_a_model_whose_output_ignores_frequency_is_refused():
    """``basic_magnetar`` is a luminosity engine without the suffix: the probe catches it."""
    pytest.importorskip("redback")
    with pytest.raises(ValueError, match=r"same value at .* Hz"):
        RA.redback_model("basic_magnetar", ["ztfg"])


def test_redshift_is_checked_like_pin():
    """``bazin_sne`` takes no redshift; ``redshift=`` used to pin one anyway, and redback's
    ``**kwargs`` swallowed it."""
    pytest.importorskip("redback")
    with pytest.raises(ValueError, match=r"'redshift'.*not a parameter of redback's 'bazin_sne'"):
        RA.redback_model("bazin_sne", redshift=0.1)


#: What the gate below refuses in redback 1.20: every ``*_bolometric`` engine (static rule), and
#: every other buildable model whose output is the same at two frequencies (the probe).
REFUSED_BY_THE_PROBE_1_20 = sorted([
    "basic_magnetar", "bazin_sne", "collapsing_magnetar", "collapsing_radiative_losses",
    "evolving_magnetar", "evolving_magnetar_only", "five_component_powerlaw",
    "four_component_powerlaw", "full_magnetar", "full_vacuum_dipole_magnetar",
    "general_magnetar", "gw_magnetar", "magnetar_luminosity_evolution", "magnetar_only",
    "piecewise_radiative_losses", "radiative_losses", "radiative_losses_mdr",
    "radiative_losses_smoothness", "six_component_powerlaw", "three_component_powerlaw",
    "two_component_powerlaw", "vacuum_dipole_magnetar_only", "villar_sne"])


def test_redback_gate_lists_exactly_which_models_are_refused():
    """Loop over ``all_models_dict``: the static rule refuses exactly the ``*_bolometric`` names
    that take no ``redshift``, and the frequency probe exactly the list above (redback 1.20)."""
    pytest.importorskip("redback")
    import inspect

    RA._import_redback()
    from redback.model_library import all_models_dict

    static, probed = [], []
    for name in sorted(all_models_dict):
        try:
            RA.redback_model(name)
        except ValueError as exc:
            msg = str(exc)
            if "not a flux density" in msg:
                (static if "bolometric engine" in msg else probed).append(name)
        except Exception:                   # noqa: BLE001 - other refusals are not this gate's
            pass
    expected_static = sorted(n for n, f in all_models_dict.items()
                             if n.endswith("_bolometric")
                             and "redshift" not in inspect.signature(f).parameters)
    assert static == expected_static
    assert len(static) == 55
    assert probed == REFUSED_BY_THE_PROBE_1_20
