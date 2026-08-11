import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from whisper_labia import load_lightcurve, plot_light_curve  # noqa: E402


def test_report_layout(at2017gfo_csv):
    lc = load_lightcurve(at2017gfo_csv, band_lookup=True)
    fig = plot_light_curve(lc, layout="report")
    assert len(fig.axes) >= 2   # magnitude + flux panels
    plt.close(fig)


def test_grid_flux(at2017gfo_csv):
    lc = load_lightcurve(at2017gfo_csv, band_lookup=True)
    fig = plot_light_curve(lc, layout="grid", quantity="flux",
                           bands=["g-band", "r-band", "i-band"])
    assert fig is not None
    plt.close(fig)


def test_absolute_mag_requires_redshift(at2017gfo_csv):
    lc = load_lightcurve(at2017gfo_csv, band_lookup=True)
    with pytest.raises(ValueError):
        plot_light_curve(lc, layout="grid", quantity="absolute_mag", bands=["g-band"])


def test_absolute_mag_with_redshift(at2017gfo_csv):
    lc = load_lightcurve(at2017gfo_csv, redshift=0.0099, band_lookup=True)
    fig = plot_light_curve(lc, layout="grid", quantity="absolute_mag",
                           bands=["g-band", "r-band"])
    assert fig is not None
    plt.close(fig)


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
    fig = plot_light_curve(lc, layout="report")
    plt.close(fig)


# --- plot_corner -------------------------------------------------------------------------------

def _post(rng, names, shift=0.0, n=200):
    import pandas as pd
    return pd.DataFrame({k: rng.normal(0.0 + shift, 1.0, n) for k in names})


def test_plot_corner_overlays_and_legend():
    import numpy as np
    import whisper_labia as wp
    rng = np.random.default_rng(0)
    names = ["a", "b", "c"]
    posts = [_post(rng, names, s) for s in (0.0, 0.3, -0.3)]
    fig = wp.plot_corner(posts, labels=["x", "y", "z"], truths={"a": 0, "b": 0, "c": 0},
                         title="t")
    assert len(fig.get_axes()) == 9          # 3x3 corner
    assert fig.legends                        # legend mapping colour -> label
    plt.close(fig)


def test_plot_corner_common_params_log_and_errors():
    import numpy as np
    import pandas as pd
    import whisper_labia as wp
    rng = np.random.default_rng(0)
    p1 = pd.DataFrame({"a": rng.uniform(1, 9, 100), "b": rng.normal(0, 1, 100)})
    p2 = pd.DataFrame({"a": rng.uniform(1, 9, 100), "c": rng.normal(0, 1, 100)})
    fig = wp.plot_corner([p1, p2], log_params=["a"])   # common param = ["a"], log axis
    assert len(fig.get_axes()) == 1
    plt.close(fig)
    # array input requires matching parameter names; empty input is an error
    arr = rng.normal(0, 1, (50, 2))
    fig2 = wp.plot_corner([arr], parameters=["a", "b"])
    assert len(fig2.get_axes()) == 4
    plt.close(fig2)
    with pytest.raises(ValueError):
        wp.plot_corner([])


def _ppc_setup():
    import numpy as np
    from whisper_labia import LightCurve, Prior, Uniform, fit_ABC, get_model
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
    from whisper_labia import plot_ppc
    lc, res = _ppc_setup()
    fig = plot_ppc(res, lc, quantity="flux")            # single fit -> panel per band
    assert len(fig.axes) >= 2                            # g + r panels
    plt.close(fig)


def test_plot_ppc_multi_method_magnitude():
    from whisper_labia import plot_ppc
    lc, res = _ppc_setup()
    fig = plot_ppc({"a": res, "b": res}, lc, quantity="apparent_mag")   # per-method grid, mag axis
    assert len(fig.axes) >= 2
    # magnitude panels are inverted (brighter=up): ylim descends
    ax = fig.axes[0]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]
    plt.close(fig)


def test_plot_ppc_panel_by_band_override():
    from whisper_labia import plot_ppc
    lc, res = _ppc_setup()
    fig = plot_ppc({"a": res, "b": res}, lc, panel_by="band", quantity="flux")
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_calibration_multi_and_per_band():
    from whisper_labia import plot_calibration
    lc, res = _ppc_setup()
    fig = plot_calibration({"a": res}, lc, n_draws=200)          # overall curve + diagonal
    ax = fig.axes[0]
    assert ax.get_xlabel() == "nominal credible level"
    plt.close(fig)
    fig2 = plot_calibration(res, lc, per_band=True, n_draws=200)  # one line per band + overall
    plt.close(fig2)
