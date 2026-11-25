import inspect
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from whisper_cbpf import load_lightcurve, plot_light_curve  # noqa: E402


def test_report_layout(at2017gfo_csv):
    lc = load_lightcurve(at2017gfo_csv, band_lookup=True)
    axes = plot_light_curve(lc, layout="report")
    fig = np.ravel(axes)[0].figure
    assert len(fig.axes) >= 2   # magnitude + flux panels
    assert axes.shape == (2,)
    plt.close(fig)


def test_grid_flux(at2017gfo_csv):
    lc = load_lightcurve(at2017gfo_csv, band_lookup=True)
    axes = plot_light_curve(lc, layout="grid", quantity="flux",
                            bands=["g-band", "r-band", "i-band"])
    assert axes.shape == (1, 3)
    plt.close(np.ravel(axes)[0].figure)


def test_absolute_mag_requires_redshift(at2017gfo_csv):
    lc = load_lightcurve(at2017gfo_csv, band_lookup=True)
    with pytest.raises(ValueError):
        plot_light_curve(lc, layout="grid", quantity="absolute_mag", bands=["g-band"])


def test_absolute_mag_with_redshift(at2017gfo_csv):
    lc = load_lightcurve(at2017gfo_csv, redshift=0.0099, band_lookup=True)
    axes = plot_light_curve(lc, layout="grid", quantity="absolute_mag",
                            bands=["g-band", "r-band"])
    assert axes.shape == (1, 2)
    plt.close(np.ravel(axes)[0].figure)


def test_upper_limit_markers(tmp_path):
    p = tmp_path / "ul.csv"
    p.write_text(
        "time,magnitude,e_magnitude,band,upper_limit\n"
        "1,19.0,0.05,g,False\n"   # detection -> circle
        "2,20.5,0.40,g,False\n"   # SNR < 3  -> up-triangle
        "3,21.0,0.30,g,True\n"    # upper limit -> down-triangle
    )
    lc = load_lightcurve(p)
    assert lc.upper_limit.sum() == 1
    plot_light_curve(lc, layout="report")
    plt.close("all")


# --- plot_corner -------------------------------------------------------------------------------

def _post(rng, names, shift=0.0, n=200):
    import pandas as pd
    return pd.DataFrame({k: rng.normal(0.0 + shift, 1.0, n) for k in names})


def test_plot_corner_overlays_and_legend():
    import numpy as np
    import whisper_cbpf as wp
    rng = np.random.default_rng(0)
    names = ["a", "b", "c"]
    posts = [_post(rng, names, s) for s in (0.0, 0.3, -0.3)]
    axes = wp.plot_corner(posts, labels=["x", "y", "z"], truths={"a": 0, "b": 0, "c": 0},
                          title="t")
    fig = np.ravel(axes)[0].figure
    assert axes.shape == (3, 3)
    assert (axes[2, 0].get_xlabel(), axes[2, 0].get_ylabel()) == ("a", "c")   # row-major, like corner
    assert len(fig.get_axes()) == 9          # 3x3 corner
    assert fig.legends                        # legend mapping colour -> label
    plt.close(fig)


def test_plot_corner_common_params_log_and_errors():
    import numpy as np
    import pandas as pd
    import whisper_cbpf as wp
    rng = np.random.default_rng(0)
    p1 = pd.DataFrame({"a": rng.uniform(1, 9, 100), "b": rng.normal(0, 1, 100)})
    p2 = pd.DataFrame({"a": rng.uniform(1, 9, 100), "c": rng.normal(0, 1, 100)})
    axes = wp.plot_corner([p1, p2], log_params=["a"])   # common param = ["a"], log axis
    assert axes.shape == (1, 1)
    plt.close(np.ravel(axes)[0].figure)
    # array input requires matching parameter names; empty input is an error
    arr = rng.normal(0, 1, (50, 2))
    axes2 = wp.plot_corner([arr], parameters=["a", "b"])
    assert axes2.shape == (2, 2)
    plt.close(np.ravel(axes2)[0].figure)
    with pytest.raises(ValueError):
        wp.plot_corner([])


def _ppc_setup():
    import numpy as np
    from whisper_cbpf import LightCurve, Prior, Uniform, fit_ABC, get_model
    m = get_model("gaussian_rise")
    truth = {"amplitude": 5.0, "t0": 8.0, "sigma_rise": 3.0, "tau_decay": 15.0}
    t = np.linspace(0.1, 30, 40)
    times = np.concatenate([t, t])
    bands = np.array(["g"] * 40 + ["r"] * 40)
    flux = m.predict(truth, times, bands)
    lc = LightCurve(time=times, band=bands, flux=flux + np.random.default_rng(0).normal(0, 0.1, 80),
                    flux_err=np.full_like(flux, 0.1), name="syn")
    prior = Prior({k: Uniform(0.5 * v, 1.5 * v) for k, v in truth.items()})
    res = fit_ABC(lc, "gaussian_rise", prior=prior, n_simulations=2000, quantile=0.05, n_jobs=1, seed=0)
    return lc, res


def test_plot_ppc_single_flux_grid_by_band():
    from whisper_cbpf import plot_ppc
    lc, res = _ppc_setup()
    axes = plot_ppc(res, lc, quantity="flux")           # single fit -> panel per band
    fig = np.ravel(axes)[0].figure
    assert len(fig.axes) >= 2                            # g + r panels
    assert axes.ndim == 2
    plt.close(fig)


def test_plot_ppc_multi_method_magnitude():
    from whisper_cbpf import plot_ppc
    lc, res = _ppc_setup()
    axes = plot_ppc({"a": res, "b": res}, lc, quantity="apparent_mag")   # per-method grid, mag axis
    fig = np.ravel(axes)[0].figure
    assert len(fig.axes) >= 2
    # magnitude panels are inverted (brighter=up): ylim descends
    ax = fig.axes[0]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]
    plt.close(fig)


def test_plot_ppc_panel_by_band_override():
    from whisper_cbpf import plot_ppc
    lc, res = _ppc_setup()
    axes = plot_ppc({"a": res, "b": res}, lc, panel_by="band", quantity="flux")
    fig = np.ravel(axes)[0].figure
    assert len(fig.axes) >= 2
    plt.close(fig)


def _time_axis_label(axes):
    """The time-axis text: the figure's supxlabel (grid layouts), else the bottom panel's xlabel."""
    return np.ravel(axes)[0].figure.get_supxlabel() or np.ravel(axes)[-1].get_xlabel()


def test_light_curve_time_axis_names_the_day_0_the_curve_declares(at2017gfo_csv):
    """Regression: every shifted curve read "days since explosion", including the
    demos' curves whose day 0 is the first detection."""
    lc = load_lightcurve(at2017gfo_csv, band_lookup=True).select_bands(["g-band", "r-band"])
    t_fd = float(lc.time.min())
    for curve, want in [
            (lc, "time [MJD]"),
            (lc.set_explosion_date(57982.529), "days since explosion (MJD 57982.529)"),
            (lc.set_time_reference(t_fd, "first detection"),
             f"days since first detection (MJD {t_fd:.3f})")]:
        for layout in ("report", "grid"):
            axes = plot_light_curve(curve, layout=layout)
            assert _time_axis_label(axes) == want
            plt.close(np.ravel(axes)[0].figure)


def test_ppc_time_axis_names_the_day_0_the_curve_declares():
    from whisper_cbpf import LightCurve, plot_ppc
    lc, res = _ppc_setup()
    t_ref = 60000.0             # the same curve recorded in MJD and shifted back, so the fit applies
    mjd = LightCurve(time=lc.time + t_ref, band=lc.band, flux=lc.flux, flux_err=lc.flux_err,
                     name=lc.name)
    axes = plot_ppc(res, mjd.set_time_reference(t_ref, "first detection"), quantity="flux")
    assert _time_axis_label(axes) == "days since first detection (MJD 60000.000)"
    plt.close(np.ravel(axes)[0].figure)


def test_binding_a_redback_model_leaves_the_session_plot_settings_alone(tmp_path):
    """Regression (bugs.md §4): importing redback turns ``text.usetex`` on for the whole session, so
    on a machine without LaTeX every later plot raised "latex could not be found".

    A fresh interpreter, because redback is imported once per process: after any earlier test has
    imported it, binding a model here could not show the side effect.
    """
    pytest.importorskip("redback")
    import os
    import subprocess
    import sys

    import whisper_cbpf

    root = str(Path(whisper_cbpf.__file__).parents[1])      # the package under test, not an install
    code = "\n".join([
        "import sys",
        f"sys.path.insert(0, {root!r})",
        "import matplotlib",
        "keys = ('text.usetex', 'font.family', 'text.latex.preamble')",
        "before = {k: matplotlib.rcParams[k] for k in keys}",
        "import whisper_cbpf as wp",
        "wp.register_redback('arnett', redshift=0.05)",
        "assert 'redback' in sys.modules",
        "after = {k: matplotlib.rcParams[k] for k in keys}",
        "assert after == before, (before, after)",
    ])
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path)},
                         cwd=str(tmp_path), timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]


def test_plot_calibration_multi_and_per_band():
    from whisper_cbpf import plot_calibration
    lc, res = _ppc_setup()
    ax = plot_calibration({"a": res}, lc, n_draws=200)           # overall curve + diagonal
    assert ax.get_xlabel() == "nominal credible level"
    plt.close(ax.figure)
    ax2 = plot_calibration(res, lc, per_band=True, n_draws=200)  # one line per band + overall
    plt.close(ax2.figure)


# --- one figure, drawn once in a notebook ------------------------------------------------------
# Returning the Figure drew every plot twice in Jupyter: once by IPython's displayhook as Out[n], and
# again by the inline backend's post-execute flush of every open pyplot figure.

_PLOT_FUNCTIONS = ["plot_light_curve", "plot_ppc", "plot_calibration", "plot_corner",
                   "plot_forecast", "plot_models", "plot_model_comparison", "plot_widths"]


def _table():
    """A comparison table in the shape wp.compare's has (the columns the ranking plot reads)."""
    import pandas as pd
    return pd.DataFrame({"model": ["arnett", "magnetar", "csm"], "delta": [0.0, 3.1, float("nan")],
                         "weight": [0.83, 0.17, float("nan")], "grade": ["", "positive", ""],
                         "status": ["ok", "ok", "left out"], "left_out_reason": ["", "", "k >= n"]})


def _call_plot(name, csv):
    import whisper_cbpf as wp
    from whisper_cbpf import plotting
    from whisper_cbpf.forecast import forecast
    if name == "plot_light_curve":
        return wp.plot_light_curve(load_lightcurve(csv, band_lookup=True))
    if name == "plot_corner":
        return wp.plot_corner([_post(np.random.default_rng(0), ["a", "b"])])
    if name == "plot_model_comparison":
        return plotting.plot_model_comparison(_table())
    lc, res = _ppc_setup()
    if name == "plot_ppc":
        return wp.plot_ppc(res, lc, quantity="flux")
    if name == "plot_forecast":
        return plotting.plot_forecast(forecast(res, np.linspace(30, 40, 6), ["g", "r"], lc=lc))
    if name == "plot_models":
        return plotting.plot_models({"a": res}, lc)
    if name == "plot_widths":
        return plotting.plot_widths(res)
    return wp.plot_calibration(res, lc, n_draws=200)


@pytest.mark.parametrize("name", _PLOT_FUNCTIONS)
def test_plot_returns_axes_of_one_open_figure_and_never_shows(name, at2017gfo_csv, monkeypatch):
    shows = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shows.append(1))
    before = set(plt.get_fignums())
    out = _call_plot(name, at2017gfo_csv)
    new = set(plt.get_fignums()) - before
    assert len(new) == 1                       # one figure, left open in pyplot
    assert not shows                           # the library never calls plt.show()

    axes = np.ravel(out)
    assert all(isinstance(ax, Axes) for ax in axes)
    assert {ax.figure.number for ax in axes} == new

    # The IPython half needs IPython; the guards above run without it (CI installs only [dev]).
    pytest.importorskip("IPython")
    from types import SimpleNamespace

    from IPython.core.formatters import DisplayFormatter
    from IPython.core.pylabtools import select_figure_formats

    shell = SimpleNamespace(display_formatter=DisplayFormatter())
    select_figure_formats(shell, {"png"})      # the PNG formatter `%matplotlib inline` registers
    data, _ = shell.display_formatter.format(out)
    assert "image/png" not in data             # so Out[n] is text; the flush draws the one image
    plt.close("all")


def test_bare_call_in_a_notebook_draws_one_image(at2017gfo_csv, tmp_path, monkeypatch):
    """Each plot function as the bare last line of a cell, run by a real kernel: exactly one PNG."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    import whisper_cbpf

    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    root = str(Path(whisper_cbpf.__file__).parents[1])      # the package under test, not an install
    setup = "\n".join([
        "%matplotlib inline",
        "import sys",
        f"sys.path.insert(0, {root!r})",
        "import numpy as np",
        "import whisper_cbpf as wp",
        "from whisper_cbpf.forecast import forecast",
        "from whisper_cbpf.plotting import (plot_forecast, plot_model_comparison, plot_models,",
        "                                   plot_widths)",
        inspect.getsource(_post),
        inspect.getsource(_ppc_setup),
        inspect.getsource(_table),
        f"lc = wp.load_lightcurve({str(at2017gfo_csv)!r}, band_lookup=True)",
        "syn, res = _ppc_setup()",
        "fc = forecast(res, np.linspace(30, 40, 6), ['g', 'r'], lc=syn)",
    ])
    calls = {
        "plot_light_curve": "wp.plot_light_curve(lc)",
        "plot_ppc": "wp.plot_ppc(res, syn, quantity='flux')",
        "plot_calibration": "wp.plot_calibration(res, syn, n_draws=200)",
        "plot_corner": "wp.plot_corner([_post(np.random.default_rng(0), ['a', 'b'])])",
        "plot_forecast": "plot_forecast(fc)",
        "plot_models": "plot_models({'a': res}, syn)",
        "plot_model_comparison": "plot_model_comparison(_table())",
        "plot_widths": "plot_widths(res)",
    }
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(setup)]
                                  + [nbformat.v4.new_code_cell(c) for c in calls.values()])
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}}).execute()

    images = {name: sum("image/png" in out.get("data", {}) for out in cell.outputs)
              for name, cell in zip(calls, nb.cells[1:])}
    assert images == dict.fromkeys(calls, 1)


# --- the plot kit: forecasts, models over the data, the ranking, posterior widths, intervals ----------

def _mjd_curve(lc, t_ref=60000.0):
    """The synthetic curve recorded in MJD and shifted back, so the fit's clock applies."""
    from whisper_cbpf import LightCurve
    mjd = LightCurve(time=lc.time + t_ref, band=lc.band, flux=lc.flux, flux_err=lc.flux_err,
                     name="syn")
    return mjd.set_time_reference(t_ref, "first detection")


def _fills(ax):
    from matplotlib.collections import PolyCollection
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def test_plot_forecast_shades_68_and_95_per_band_on_the_curve_clock():
    from whisper_cbpf.forecast import forecast
    from whisper_cbpf.plotting import plot_forecast
    lc, res = _ppc_setup()
    fc = forecast(res, np.linspace(30.0, 40.0, 6), ["g", "r"], lc=_mjd_curve(lc), survey_depth=2.0)
    ax = plot_forecast(fc)
    assert ax.get_xlabel() == "days since first detection (MJD 60000.000)"
    assert ax.get_ylabel() == "apparent magnitude (AB)" and ax.yaxis_inverted()
    assert len(_fills(ax)) == 4                                   # 68 % and 95 %, two bands
    depth_lines = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    assert len(depth_lines) == 2 and set(depth_lines[0].get_ydata()) == {2.0}
    labels = ax.get_legend_handles_labels()[1] + [t.get_text() for t in ax.get_legend().get_texts()]
    assert {"68 % interval", "95 % interval", "5-sigma depth", "most draws too faint"} <= set(labels)
    median = [ln for ln in ax.get_lines() if ln.get_label() == "g"][0]
    np.testing.assert_allclose(median.get_ydata(), fc.loc[fc["band"] == "g", "q50"])
    plt.close(ax.figure)


def test_plot_forecast_single_epoch_uses_error_bars_and_keeps_a_given_axis_inverted():
    from whisper_cbpf.forecast import forecast
    from whisper_cbpf.plotting import plot_forecast
    lc, res = _ppc_setup()
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    out = plot_forecast(forecast(res, [31.0], "g"), ax=ax)
    assert out is ax and ax.yaxis_inverted() and not _fills(ax)
    assert ax.get_xlabel() == "time (the fit's clock)"
    plt.close(fig)
    import pandas as pd
    with pytest.raises(ValueError, match=r"no \['mean_mag'\] column"):
        plot_forecast(pd.DataFrame({"time": [1.0], "band": ["g"]}))
    # a Comparison.forecast table (a 'model' column): one model at a time, and the error says how
    one = forecast(res, [31.0, 32.0], "g")
    both = pd.concat([one.assign(model="a"), one.assign(model="b")], ignore_index=True)
    with pytest.raises(ValueError, match=r"forecasts of 2 models.*fc\['model'\] == 'a'"):
        plot_forecast(both)
    ax = plot_forecast(both[both["model"] == "b"])
    assert ax.get_title().startswith("Forecast: b")
    plt.close(ax.figure)


def test_plot_models_draws_every_model_with_residuals_that_are_data_minus_median():
    from whisper_cbpf.forecast import _draw_magnitudes, _draws, _model_of
    from whisper_cbpf.plotting import plot_models
    lc, res = _ppc_setup()
    axes = plot_models({"a": res, "b": res}, lc)
    assert axes.shape == (2, 2)
    assert [ax.get_title() for ax in axes[0]] == ["g", "r"]
    assert axes[1, 0].get_ylabel() == "data - model [mag]"
    assert all(ax.yaxis_inverted() for ax in axes[0])
    labels = axes[0, 0].get_legend_handles_labels()[1]
    assert {"a", "b", "data"} <= set(labels)
    # residuals recomputed: data minus the median of the same draws at the detections
    full = lc.add_mag()
    g = np.asarray(full.band) == "g"
    m = _model_of(res, None)
    theta, _ = _draws(res, m, 200, 0)
    mags, dark, _ = _draw_magnitudes(m, theta, np.asarray(full.time)[g], np.asarray(full.band)[g])
    want = np.asarray(full.magnitude)[g] - np.median(mags, axis=0)
    np.testing.assert_allclose(axes[1, 0].lines[0].get_ydata(), want, atol=1e-9)
    assert np.ravel(axes)[0].figure.get_supxlabel() == "time [MJD]"
    plt.close(np.ravel(axes)[0].figure)


def test_plot_models_ranks_a_comparison_names_weights_and_left_out_fits():
    from types import SimpleNamespace

    import pandas as pd

    from whisper_cbpf.plotting import plot_models
    from whisper_cbpf.samplers.base import SamplerResult
    lc, res = _ppc_setup()
    empty = SamplerResult(sampler="abc", model="gaussian_rise", parameters=list(res.parameters),
                          samples=res.samples.iloc[:0], summary={}, best_params={}, n_data=80,
                          n_params=4, runtime_s=0.0)
    comp = SimpleNamespace(results={"m1": res, "m2": res, "m3": empty},
                           table=pd.DataFrame({"model": ["m1", "m2", "m3"], "delta": [1.7, 0.0, np.nan],
                                               "weight": [0.3, 0.7, np.nan]}))
    axes = plot_models(comp, lc, bands=["g"])
    leg = axes[0, 0].get_legend()
    texts = [t.get_text() for t in leg.get_texts()]
    assert texts[:2] == ["m2 (weight 0.70)", "m1 (weight 0.30)"]          # rank order
    assert leg.get_title().get_text() == "left out: m3 (no posterior draws)"
    plt.close(np.ravel(axes)[0].figure)
    fig, ax = plt.subplots()
    assert plot_models({"a": res}, lc, ax=ax, bands=["r"]) is ax             # one band, no residuals
    with pytest.raises(ValueError, match="with ax= plot_models draws one band"):
        plot_models({"a": res}, lc, ax=ax)
    plt.close("all")


def test_plot_models_draws_upper_limits_without_residuals():
    from whisper_cbpf import LightCurve
    from whisper_cbpf.plotting import plot_models
    _, res = _ppc_setup()
    t = np.array([2.0, 5.0, 8.0, 12.0, 20.0, 25.0])
    lc = LightCurve(time=t, band=["g"] * 6, magnitude=[9.5, 8.0, 7.4, 7.3, 7.9, 12.0],
                    magnitude_err=[0.1, 0.1, 0.1, 0.1, 0.1, np.nan],
                    upper_limit=[False] * 5 + [True], name="few points")
    axes = plot_models({"a": res}, lc)
    assert axes.shape == (2, 1)
    assert "upper limit" in axes[0, 0].get_legend_handles_labels()[1]
    assert len(axes[1, 0].lines[0].get_xdata()) == 5                 # detections only
    plt.close(np.ravel(axes)[0].figure)


def test_plot_models_evaluates_a_comparison_with_its_own_model_objects():
    """wp.compare binds families as Model objects it never registers by name, so a result's
    ``.model`` alone does not find them; the Comparison's ``.models`` does."""
    import dataclasses
    from types import SimpleNamespace

    import pandas as pd

    import whisper_cbpf as wp
    from whisper_cbpf.plotting import plot_models
    lc, res = _ppc_setup()
    unregistered = dataclasses.replace(res, model="bound_by_compare_only")
    comp = SimpleNamespace(results={"fam": unregistered},
                           models={"fam": wp.get_model("gaussian_rise")},
                           table=pd.DataFrame({"model": ["fam"], "delta": [0.0], "weight": [1.0]}))
    axes = plot_models(comp, lc)
    assert axes.shape == (2, 2)
    plt.close(np.ravel(axes)[0].figure)
    with pytest.raises(ValueError, match="not registered in this session.*comparison.forecast"):
        plot_models({"fam": unregistered}, lc)


def test_plot_model_comparison_ranks_and_follows_the_weights():
    from whisper_cbpf.plotting import plot_model_comparison
    table = _table()
    table.attrs["criterion"] = "BIC"
    ax = plot_model_comparison(table)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["arnett", "magnetar", "csm"]
    assert [round(p.get_width(), 2) for p in ax.patches] == [0.83, 0.17]
    texts = [t.get_text() for t in ax.texts]
    assert "delta 3.1  positive" in texts and "left out: k >= n" in texts
    assert ax.get_title().startswith("Ranking by BIC")
    plt.close(ax.figure)
    # the winner's grade is its preference over the runner-up, not a grade of a gap to itself
    won = _table().assign(grade=["positive", "positive", ""])
    won.attrs.update(criterion="BIC", winner="arnett")
    ax = plot_model_comparison(won)
    texts = [t.get_text() for t in ax.texts]
    assert "best  positive over the next" in texts and "delta 0.0  positive" not in texts
    plt.close(ax.figure)
    earlier = table.assign(weight=[0.4, 0.6, np.nan], delta=[0.8, 0.0, np.nan])
    a, b = plot_model_comparison(table, history={"+3 d": earlier, "+6 d": table})
    assert [t.get_text() for t in b.get_xticklabels()] == ["+3 d", "+6 d"]
    line = [ln for ln in b.get_lines() if ln.get_label() == "arnett"][0]
    np.testing.assert_allclose(line.get_ydata(), [0.4, 0.83])
    plt.close(a.figure)
    _, b = plot_model_comparison(table, history=[("+3 d", earlier)])       # the latest is appended
    assert [t.get_text() for t in b.get_xticklabels()] == ["+3 d", "latest"]
    plt.close("all")
    with pytest.raises(ValueError, match=r"no \['weight'\] column"):
        plot_model_comparison(table.drop(columns=["weight"]))
    with pytest.raises(TypeError, match="takes a Comparison"):
        plot_model_comparison(object())


def _width_fit():
    import pandas as pd

    from whisper_cbpf.priors import Fixed, LogUniform, Prior, Uniform
    from whisper_cbpf.results import prior_record
    from whisper_cbpf.samplers.base import SamplerResult
    rng = np.random.default_rng(0)
    prior = Prior({"a": Uniform(0.0, 1.0), "m": LogUniform(1e-3, 1.0), "z": Fixed(0.1)})
    samples = pd.DataFrame({"a": rng.uniform(0.0, 1.0, 20000),
                            "m": 10 ** rng.normal(-2.0, 0.1, 20000), "z": np.full(20000, 0.1)})
    res = SamplerResult(sampler="hand", model="not_registered", parameters=["a", "m", "z"],
                        samples=samples, summary={}, best_params={}, n_data=10, n_params=2,
                        runtime_s=0.0, provenance={"model": {"prior": prior_record(prior)}})
    return res, prior, samples


def test_plot_widths_numbers_are_posterior_over_prior_68_widths_in_the_prior_coordinate():
    from whisper_cbpf.plotting import plot_widths, posterior_width_ratios
    res, prior, s = _width_fit()
    want_a = np.subtract(*np.percentile(s["a"], [84, 16])) / 0.68
    want_m = np.subtract(*np.log10(np.percentile(s["m"], [84, 16]))) / (0.68 * 3.0)
    got = posterior_width_ratios(res)                        # the prior recorded with the fit
    assert set(got) == {"a", "m"}                            # a Fixed parameter has no width
    assert got["a"] == pytest.approx(want_a, rel=1e-12) and got["a"] == pytest.approx(1.0, abs=0.02)
    assert got["m"] == pytest.approx(want_m, rel=1e-12) and got["m"] == pytest.approx(0.098, abs=0.005)
    assert posterior_width_ratios(res, prior) == got
    ax = plot_widths(res)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "m (log10)"]
    np.testing.assert_allclose([p.get_width() for p in ax.patches], [got["a"], got["m"]])
    assert ax.get_xlabel() == "posterior 68 % width / prior 68 % width"
    plt.close(ax.figure)


def test_plot_widths_reads_the_same_numbers_and_threshold_as_the_facts_file():
    facts = pytest.importorskip("whisper_cbpf.facts")
    from whisper_cbpf.plotting import plot_widths, posterior_width_ratios
    lc, res = _ppc_setup()
    ratios = posterior_width_ratios(res)
    rec = facts.result_facts(res, lc)["parameters"]
    for name, value in ratios.items():
        assert value == pytest.approx(rec[name]["width_ratio"], rel=1e-12)
    ax = plot_widths(res)
    cut = facts.DEFAULT_THRESHOLDS["prior_dominated_ratio"]
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    assert len(dashed) == 1 and dashed[0].get_xdata()[0] == pytest.approx(cut)
    plt.close(ax.figure)


def test_plot_models_leaves_pre_event_rows_out_of_the_residuals():
    import warnings

    import whisper_cbpf as wp
    from whisper_cbpf.plotting import plot_models
    t = np.array([-2.0, -0.5, 1.0, 3.0, 6.0, 10.0, 15.0, 20.0])
    flux = wp.get_model("flare").predict({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t)
    lc = wp.LightCurve(time=t + 60000.0, band=["r"] * 8, flux=flux + 0.05,
                       flux_err=np.full(8, 0.1)).set_explosion_date(60000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wp.fit(lc, "flare", sampler="abc", n_simulations=500, quantile=0.1, seed=0)
    if res.info.get("excluded_pre_event") is None:
        pytest.skip("this installation has no pre-event rule")
    axes = plot_models(res, lc)
    top, bottom = axes[0, 0], axes[1, 0]
    assert "pre-event (not fitted)" in top.get_legend_handles_labels()[1]
    assert len(bottom.lines[0].get_xdata()) == 6                       # the six fitted rows
    curve = [ln for ln in top.get_lines() if ln.get_label() == "flare"][0]
    x, y = curve.get_xdata(), np.asarray(curve.get_ydata(), float)
    assert np.all(np.isnan(y[x <= 0.0])) and np.isfinite(y[x > 0.5]).all()   # nothing before day 0
    plt.close(np.ravel(axes)[0].figure)


def test_plot_widths_needs_a_prior_and_two_draws():
    from whisper_cbpf.plotting import plot_widths
    res, prior, _ = _width_fit()
    res.provenance = {}
    with pytest.raises(ValueError, match="prior is unknown.*pass prior="):
        plot_widths(res)
    res.samples = res.samples.iloc[:1]
    with pytest.raises(ValueError, match="not enough data: 1 posterior draw"):
        plot_widths(res, prior)


def test_plot_ppc_shades_68_and_95_by_default():
    from whisper_cbpf import plot_ppc
    lc, res = _ppc_setup()
    axes = plot_ppc(res, lc, bands=["g"])
    assert len(_fills(axes[0, 0])) == 2
    assert "68 % and 95 % bands" in axes[0, 0].get_legend().get_title().get_text()
    plt.close(np.ravel(axes)[0].figure)
    axes = plot_ppc(res, lc, bands=["g"], intervals=(95,))
    assert len(_fills(axes[0, 0])) == 1
    plt.close(np.ravel(axes)[0].figure)
    with pytest.raises(ValueError, match="strictly between 0 and 100"):
        plot_ppc(res, lc, intervals=(0.68,  100))


def test_plot_corner_puts_log_uniform_parameters_on_log_axes():
    import whisper_cbpf as wp
    res, _, s = _width_fit()
    axes = wp.plot_corner([res], parameters=["a", "m"])
    assert axes[1, 0].get_ylabel() == r"$\log_{10}\,$m" and axes[1, 0].get_xlabel() == "a"
    plt.close(np.ravel(axes)[0].figure)
    axes = wp.plot_corner([s[["a", "m"]]])                  # a bare table has no prior: linear
    assert axes[1, 0].get_ylabel() == "m"
    plt.close(np.ravel(axes)[0].figure)
    axes = wp.plot_corner([res], parameters=["a", "m"], log_params=None)
    assert axes[1, 0].get_ylabel() == "m"
    plt.close(np.ravel(axes)[0].figure)
    with pytest.raises(ValueError, match="log_params is 'auto', None or a list"):
        wp.plot_corner([res], log_params="m")


def test_the_plot_kit_docstring_examples_run():
    import doctest
    import io
    import warnings

    from whisper_cbpf import plotting
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    names = ["plot_forecast", "plot_models", "plot_model_comparison", "plot_widths",
             "posterior_width_ratios", "plot_ppc", "plot_corner"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in names:
            for test in doctest.DocTestFinder().find(getattr(plotting, name), name,
                                                     globs=dict(vars(plotting))):
                runner.run(test, out=io.StringIO().write)
    plt.close("all")
    assert runner.tries >= len(names) and runner.failures == 0
