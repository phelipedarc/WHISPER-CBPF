"""MOSFiT / redback ``tde_fallback``: fall-back of a disrupted star with viscous delay, in JAX.

Adapted from redback-jax (Nikhil Sarin et al.), https://github.com/nikhil-sarin/redback-jax,
commit 3431e2e, GPL-3.0, and from redback (Sarin et al.), https://github.com/nikhil-sarin/redback,
GPL-3.0; modified for whisper_cbpf. The fall-back engine (:func:`build_fallback_tables`,
:func:`fallback_engine`) is adapted from MOSFiT (James Guillochon et al.),
https://github.com/guillochon/MOSFiT, commit 0bc772e (version 2.1.1),
``mosfit/modules/engines/fallback.py``, MIT licence, Copyright (c) 2016-2026 James Guillochon (its
copyright and permission notice follow this docstring and are in ``THIRD_PARTY_NOTICES.md``);
redback's ``_initialize_mosfit_tde_model`` and ``_tde_mosfit_engine`` are the same code. The tables
are Guillochon & Ramirez-Ruiz's, as distributed with redback (byte-identical to MOSFiT's).

The model (Mockler, Guillochon & Ramirez-Ruiz 2019; MOSFiT, Guillochon et al. 2018): a star of mass
``mstar`` is disrupted by a black hole of ``mbh6`` million solar masses at impact parameter ``bb``
(mapped to the penetration factor ``beta``). The mass returns to pericentre at the rate ``dM/dt``
that Guillochon & Ramirez-Ruiz (2013) simulated for polytropes of index 4/3 and 5/3, tabulated on
``beta`` and rescaled to the black hole and star with the MOSFiT scalings and the Tout et al.
main-sequence radius. The luminosity is ``eta dM/dt c^2``, capped at ``leddlimit`` times the
Eddington luminosity by ``L Ledd / (L + Ledd)``, then smeared by an exponential viscous delay
``tvisc``: ``L(t) = int L'(t') exp(-(t - t') / tvisc) dt' / tvisc``. The photosphere is Mockler et
al.'s: it grows as a power of the rate, ``rph0`` and ``lphoto``, and emits as a blackbody
(:func:`whisper_cbpf.models.jax.tde.tde_photosphere`).

**Time.** Source-frame days since the first fall-back, which is the model's time zero. In
:func:`tde_fallback_model` the epochs are observer-frame, ``t_exp`` (or ``t_exp_days``) is that
zero, and an epoch at or before it is at ``mag_floor``.

**The engine is MOSFiT's, node for node.** ``Fallback.__init__``, which redback re-runs on every
call, runs once, offline, in :func:`build_fallback_tables`: every simulation's own nodes, split at
the peak of ``dM/dt`` into a rising and a falling branch, and for every pair of neighbouring
simulated betas and each branch the normalised log-time nodes of the curve with more nodes, with
the slope and intercept in ``beta`` of log10 ``dM/dt`` on them. The shipped
``whisper_cbpf/data/tde_fallback_nodes.npz`` (1.3 MB) is its output, equal bit for bit to
MOSFiT's and to redback's arrays. ``Fallback.process`` runs in the trace
(:func:`fallback_engine`): the pair is found by ``searchsorted``, a ``beta`` equal to a simulated
one takes that simulation's own nodes (MOSFiT's equality branch, a ``where``), the 4/3-5/3 blend
(stars between 0.3 and 1 and between 15 and 22 solar masses) maps the curve with fewer nodes onto
the other, and ``dM/dt`` is interpolated linearly in time, zero outside the nodes. Against
redback's ``_tde_mosfit_engine`` and MOSFiT's ``Fallback`` at the same epochs (634 parameter
sets: the 88 of ``tests/goldens/tde_mosfit_oracle.npz``, the 65 of
``tests/goldens/tde_redback_oracle.npz``, ``bb`` = 0, 1, 2, ``bb`` giving exactly a simulated beta
and one unit in the last place either side, stars in every gamma regime, on its limits and below
0.1 solar masses) the luminosity before the viscous delay agrees to a relative 1.64e-13 at worst;
``t_peak``, ``beta`` and the stellar radius to 3e-15.
Like MOSFiT's and redback's, the rate jumps in two places, so the model is piecewise smooth in
``bb`` and in ``mstar``: where a ``beta`` equals a simulated one (the equality branch; before the
peak by up to 1.47e-2 dex, the largest over every simulated beta a float ``bb`` reaches, at two
stars per gamma), and at the limits of the gamma regimes, where a pure gamma's own nodes meet the
blend's mapped ones (at 1e6 solar masses and ``bb`` = 0.25, 0.5, 0.77, 1, 1.5: up to 2.3e-2 dex
just above ``mstar`` = 0.3, 1.5e-2 just below 22, 8e-5 below 1 and 5e-5 above 15; up to 0.3.0
the engine was continuous there, to 2e-9). The gradient is finite everywhere; at a jump it is the
derivative of the side the ``where`` selects, not a one-sided limit: at a simulated beta it is
that simulation's own curve, so a star of one gamma has a derivative of exactly 0 in ``bb`` there
(at ``bb`` = 1 and 2, for instance).

**The rest is redback's.** The viscous convolution (500 + 500 log-mirror nodes on a geometric dense
grid ``geomspace(1e-5, t_last + 100, 1000)``), the Tout radius, the scalings and the cap.
``t_peak``, ``beta`` and the stellar radius reach the photosphere as in redback.

**Deviations from redback** (rules as in ``docs/PORTING_NOTES.md``, with the measured size):

1. **(c)** The viscous kernel interpolates log10 ``L`` between the dense nodes, redback ``L``
   itself (so that it runs in float32). This is all that is left of the difference: against
   redback's ``tde_fallback_bolometric`` over 60 prior draws and 60 epochs from 0.5 to 1000 d,
   median 1.1e-5 dex, 99th percentile 1.4e-4, maximum 2.2e-4 (2e-14 with redback's own kernel on
   this engine); over 200 prior draws and 701 epochs from 1e-3 to 1e3 d, maximum 2.6e-4; in band
   magnitudes (300 prior draws, LSST griz and ZTF gr at z = 0.05) median 2.2e-5 mag, 99th
   percentile 3.7e-4, maximum 7.9e-4.
2. **(b)** Before the first fall-back and after the last node redback's luminosity is exactly zero;
   here it is the floor of 1e-30 erg/s, which keeps every derivative finite.
3. **(b)** ``bb`` outside [0, 2], where redback raises (and MOSFiT reuses its previous call's
   ``beta``): the rate holds the table edge (the node lookup clips ``beta`` to the simulated
   range, so the first or the last simulation is used, unchanged), while the ``beta`` returned,
   which reaches the photosphere, is the unclipped mapping of ``bb`` (``bb`` = 2.5 at 5 solar
   masses: the rate of ``bb`` = 2, ``beta`` 5.075). ``tde_mosfit`` clips ``b`` to [0, 2] before
   the engine, so there the two agree.
4. **(b)** The ragged node arrays are padded to a fixed length by repeating their last node, with
   the number of valid nodes stored beside them: a repeated node is a zero-length segment, which
   changes no interpolated value.
5. **(a)** The products of the ``bb``-to-``beta`` mapping are kept apart from their sums
   (``jax.lax.optimization_barrier``), so that XLA cannot fuse them into one rounding: ``beta43``
   and ``beta53``, which MOSFiT's equality branch compares with the simulated betas, are MOSFiT's
   to the last bit. A second barrier holds the node times before the first one is subtracted, so
   that XLA cannot fuse the subtraction with the last product and the first node is exactly 0, as
   MOSFiT's strict ``times < time[0]`` needs (a guarantee: without it the first node was also 0 in
   2000 random sets on the CPU and on a GPU). In a gamma blend the ``beta`` returned is MOSFiT's
   to one unit in the last place (under ``jit`` it differs in 21 % of 2000 random blends on the
   CPU and 10 % on a GPU: XLA divides by a constant as a product with its reciprocal); it reaches
   only the photosphere.
   ``dM/dt`` is interpolated relative to its maximum and scaled in log10, the luminosity carried
   in log10: the same numbers (above), and float32-safe. The final interpolation's search is
   written out (``_search_right``) so that the compiled program does not grow with the batch;
   it finds the same nodes.

**Precision.** :func:`tde_fallback_bolometric_log10` works in log10 luminosity and runs in float32:
the largest difference from float64 is 1.2e-4 dex over 60 prior draws and 1.1e-4 over 300 (the
engine alone 2.8e-5 and 4.6e-5). These were measured with the viscous kernel's search; its
arithmetic segment search (``geometric=True``, the default) gives the same float32 values, bit for
bit, over 360 prior draws x 60 epochs. Two exceptions, for which use float64. In float32 the equality
branch compares float32 betas, so it fires at other ``bb`` than in float64, MOSFiT and redback,
and a pinned round ``bb`` can land on a simulated beta in one precision and not in the other: of
``bb`` = 0.00, 0.01, ..., 2.00 it does at 0.04, 0.08, 0.24, 0.88 and 0.96, which moved the rate by
up to 3.6e-3 dex (0.88, at 0.6 solar masses; at most the size of the jump), against 4.4e-5 at
the other 196. And the float32 derivative in ``mstar`` is wrong above about 11 solar masses (the
values are right): the Tout radius's quotient rule overflows float32 (``mass**19.5`` squared),
which makes reverse mode give 6.8 for a true 0.21 at 11 solar masses and forward mode NaN.
The photometric functions need float64: the black-body step carries ``L`` in erg/s, 1e43 to 1e46,
past float32's 3.4e38, and they raise rather than return a light curve at ``mag_floor``::

    import jax; jax.config.update("jax_enable_x64", True)   # before the first array exists

References
----------
Mockler, B., Guillochon, J., Ramirez-Ruiz, E. 2019, ApJ 872, 151
(https://ui.adsabs.harvard.edu/abs/2019ApJ...872..151M)

Guillochon, J., Ramirez-Ruiz, E. 2013, ApJ 767, 25
(https://ui.adsabs.harvard.edu/abs/2013ApJ...767...25G)

Guillochon, J., Nicholl, M., Villar, V. A. et al. 2018, ApJS 236, 6 (MOSFiT)
(https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G)

Tout, C. A., Pols, O. R., Eggleton, P. P., Han, Z. 1996, MNRAS 281, 257 (the stellar radius)

Sarin, N. et al. 2024, MNRAS 531, 1203 (redback; https://arxiv.org/abs/2308.12806)
"""
# MOSFiT's copyright and permission notice. It applies to the parts of this file adapted from
# MOSFiT (the table builder and the fall-back engine):
#
# MIT License
#
# Copyright (c) 2016-2026 James Guillochon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ._adopted_tde import (GRID_START, _at_epochs, _guard_epochs, _to_rows, assemble,
                           geometric_dense_times)
from .kilonova import DAY_TO_S, SOLAR_MASS, SPEED_OF_LIGHT
from .tde import GRAVITATIONAL_CONSTANT, _interp_geometric, ab_magnitude, tde_photosphere

__all__ = [
    "fallback_engine", "viscous_log10", "tde_fallback_bolometric_log10",
    "tde_fallback_ab_magnitude", "stellar_radius", "build_fallback_tables", "fallback_tables",
    "TABLE_FILE", "PARAMETERS", "default_prior", "fallback_prior",
    "tde_fallback_model", "register_tde_fallback",
]

#: MOSFiT's fall-back nodes (the products of ``Fallback.__init__``), shipped with the package and
#: written by :func:`build_fallback_tables`.
TABLE_FILE = Path(__file__).resolve().parents[2] / "data" / "tde_fallback_nodes.npz"
_GAMMAS = ("4-3", "5-3")
#: The arrays of :data:`TABLE_FILE` that hold one row per simulation or per pair, per branch.
_BRANCH_ARRAYS = ("premaptime", "premapdmdt", "mapped_time", "beta_slope", "beta_yinter")

_KAPPA_T = 0.2 * (1.0 + 0.74)                       # Thomson opacity, solar hydrogen fraction
_LOG10_E = math.log10(math.e)
_LOG10_C2 = 2.0 * math.log10(SPEED_OF_LIGHT)
#: log10 of the Eddington luminosity per solar mass of black hole, erg/s.
_LOG10_EDD_PER_MSUN = math.log10(
    4.0 * math.pi * GRAVITATIONAL_CONSTANT * SOLAR_MASS * SPEED_OF_LIGHT / _KAPPA_T)
#: Luminosity floor: 1e-30 erg/s stands for redback's exact zero (outside the table). The rate
#: outside the table is set far below it so that the floor, not the rate, decides.
_LOG10_L_FLOOR = -30.0
_LOG10_RATE_OUTSIDE = -60.0
#: Newton's constant as MOSFiT and redback read it, astropy's ``G.cgs`` (6.6743e-11 SI times 1e3,
#: one unit in the last place below 6.6743e-8). Used to build the tables, which are then MOSFiT's
#: bit for bit; with 6.6743e-8 the slopes differ by 7e-13 dex.
_G_MOSFIT = 6.6743e-11 * 1e3

#: redback's order for ``tde_fallback``, minus ``time`` and ``redshift``.
PARAMETERS = ("mbh6", "mstar", "tvisc", "bb", "eta", "leddlimit", "rph0", "lphoto")


# --- the tables: MOSFiT's Fallback.__init__, run once, offline (NumPy) ------------------------

def _read_simulation(path, mh_base):
    """One simulation file as MOSFiT reads it: ``(log10 t [s], log10 dM/dt [g/s])``, each split
    at the peak of ``dM/dt`` into the rising branch and the falling branch (which holds the peak)."""
    # ORIGINAL (mosfit/modules/engines/fallback.py:110-136, and :148-172 for every further file):
    #     e, d = np.loadtxt(dmdedir + sim_beta_files[0])
    #     ebound = e[e < 0]
    #     dmdebound = d[e < 0]
    #     dedt = (1.0 / 3.0) * (-2.0 * ebound) ** (5.0 / 2.0) / \
    #         (2.0 * np.pi * G * Mhbase)
    #     time['lo'] = np.log10((2.0 * np.pi * G * Mhbase) *
    #                           (-2.0 * ebound) ** (-3.0 / 2.0))
    #     dmdt['lo'] = np.log10(dmdebound * dedt)
    #     ipeak['lo'] = np.argmax(dmdt['lo'])
    #     time['lo'] = np.array([time['lo'][:ipeak['lo']], time['lo'][ipeak['lo']:]], dtype=object)
    #     dmdt['lo'] = np.array([dmdt['lo'][:ipeak['lo']], dmdt['lo'][ipeak['lo']:]], dtype=object)
    g = _G_MOSFIT
    e, d = np.loadtxt(path)
    ebound = e[e < 0]
    dmdebound = d[e < 0]
    dedt = (1.0 / 3.0) * (-2.0 * ebound) ** (5.0 / 2.0) / (2.0 * np.pi * g * mh_base)
    time = np.log10((2.0 * np.pi * g * mh_base) * (-2.0 * ebound) ** (-3.0 / 2.0))
    dmdt = np.log10(dmdebound * dedt)
    ipeak = int(np.argmax(dmdt))
    return (time[:ipeak], time[ipeak:]), (dmdt[:ipeak], dmdt[ipeak:])


def _map_pair(time_lo, dmdt_lo, time_hi, dmdt_hi, beta_lo, beta_hi):
    """MOSFiT's beta interpolation between two neighbouring simulations, one branch: the denser
    curve's mapped times and, on them, the slope and intercept in beta of log10 dM/dt."""
    # ORIGINAL (mosfit/modules/engines/fallback.py:185-236, for branch j of the pair i - 1, i):
    #     if len(time['lo'][j]) < len(time['hi'][j]):
    #         interp = 'lo'; nointerp = 'hi'
    #     else:                    # will also catch case where they have the same lengths
    #         interp = 'hi'; nointerp = 'lo'
    #     mapped_time[nointerp].append(
    #         1. / (time[nointerp][j][-1] - time[nointerp][j][0]) *
    #         (time[nointerp][j] - time[nointerp][j][0]))
    #     mapped_time[interp].append(
    #         1. / (time[interp][j][-1] - time[interp][j][0]) *
    #         (time[interp][j] - time[interp][j][0]))
    #     mapped_time[interp][j][0] = 0
    #     mapped_time[interp][j][-1] = 1
    #     mapped_time[nointerp][j][0] = 0
    #     mapped_time[nointerp][j][-1] = 1
    #     dmdtinterp = _lin_interp(mapped_time[nointerp][j], mapped_time[interp][j], dmdt[interp][j])
    #     if interp == 'hi':
    #         slope = ((dmdtinterp - dmdt['lo'][j]) / (self._sim_beta[g][i] - self._sim_beta[g][i - 1]))
    #     else:
    #         slope = ((dmdt['hi'][j] - dmdtinterp) / (self._sim_beta[g][i] - self._sim_beta[g][i - 1]))
    #     yinter1 = (dmdt[nointerp][j] - self._beta_slope[g][-1][j] * self._sim_beta[g][i - 1])
    #     yinter2 = (dmdtinterp - self._beta_slope[g][-1][j] * self._sim_beta[g][i])
    #     self._beta_yinter[g][-1].append((yinter1 + yinter2) / 2.0)
    #     self._mapped_time[g][-1].append(np.array(mapped_time[nointerp][j]))
    time = {"lo": time_lo, "hi": time_hi}
    dmdt = {"lo": dmdt_lo, "hi": dmdt_hi}
    interp, nointerp = ("lo", "hi") if len(time_lo) < len(time_hi) else ("hi", "lo")
    mapped = {}
    for key in (nointerp, interp):
        mapped[key] = 1. / (time[key][-1] - time[key][0]) * (time[key] - time[key][0])
        mapped[key][0] = 0
        mapped[key][-1] = 1
    dmdtinterp = np.interp(mapped[nointerp], mapped[interp], dmdt[interp])
    if interp == "hi":
        slope = (dmdtinterp - dmdt["lo"]) / (beta_hi - beta_lo)
    else:
        slope = (dmdt["hi"] - dmdtinterp) / (beta_hi - beta_lo)
    yinter1 = dmdt[nointerp] - slope * beta_lo
    yinter2 = dmdtinterp - slope * beta_hi
    return np.array(mapped[nointerp]), slope, (yinter1 + yinter2) / 2.0


def _pad(rows, width):
    """Ragged rows as one ``(len(rows), width)`` array, each row padded by repeating its last value."""
    return np.stack([np.pad(r, (0, width - r.size), mode="edge") for r in rows])


def build_fallback_tables(table_dir):
    """MOSFiT's ``Fallback.__init__`` on the Guillochon & Ramirez-Ruiz files: the arrays of
    :data:`TABLE_FILE`.

    ``table_dir`` holds the ``4-3`` and ``5-3`` folders of ``<beta>.dat`` files (energy and
    ``dM/dE`` rows, cgs, for a 1e6 solar-mass black hole and a solar star; the files MOSFiT and
    redback distribute, byte-identical). For each file, as MOSFiT: keep the bound debris
    (``E < 0``), convert to ``(log10 t, log10 dM/dt)`` with ``t = 2 pi G M (-2E)^(-3/2)`` and
    ``dM/dt = dM/dE dE/dt``, and split at the peak of ``dM/dt`` into a rising and a falling branch
    (``premaptime``, ``premapdmdt``). For each pair of neighbouring simulated betas and each branch:
    the normalised log-time nodes of the curve with more nodes (``mapped_time``) and, on them, the
    slope and intercept in beta of log10 dM/dt (``beta_slope``, ``beta_yinter``), the other curve
    being interpolated onto those nodes.

    Returns ``{name: array}``: per gamma ``k`` (``4_3``, ``5_3``) ``sim_beta_k`` (n,);
    ``premaptime_k_j`` and ``premapdmdt_k_j`` (n, width_j) for branch ``j`` (0 rising, 1 falling);
    ``mapped_time_k_j``, ``beta_slope_k_j`` and ``beta_yinter_k_j`` (n - 1, width_j);
    ``premap_count_k`` (n, 2) and ``mapped_count_k`` (n - 1, 2), the number of valid nodes of each
    row; and ``provenance`` (JSON: method, files and their SHA-256). Each row is padded to the
    longest branch ``width_j`` of either gamma by repeating its last node (a repeated node is a
    zero-length segment, which linear interpolation passes over); a row's validity mask is
    ``arange(width_j) < count``. The shipped file is
    ``np.savez_compressed(TABLE_FILE, **build_fallback_tables(<redback>/tables/guillochon_tde_data))``.
    """
    table_dir = Path(table_dir)
    mh_base = 1.0e6 * SOLAR_MASS
    raw, files = {}, {}
    for gamma in _GAMMAS:
        paths = sorted((table_dir / gamma).glob("*.dat"), key=lambda p: float(p.stem))
        sim = [float(p.stem) for p in paths]
        sims = [_read_simulation(p, mh_base) for p in paths]
        pairs = [[_map_pair(sims[i - 1][0][j], sims[i - 1][1][j], sims[i][0][j], sims[i][1][j],
                            sim[i - 1], sim[i]) for j in (0, 1)] for i in range(1, len(sim))]
        raw[gamma] = (sim, sims, pairs)
        files[gamma] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    width = [max(s[0][j].size for _, sims, _ in raw.values() for s in sims) for j in (0, 1)]
    out = {}
    for gamma, (sim, sims, pairs) in raw.items():
        k = gamma.replace("-", "_")
        out[f"sim_beta_{k}"] = np.asarray(sim)
        out[f"premap_count_{k}"] = np.array([[s[0][j].size for j in (0, 1)] for s in sims])
        out[f"mapped_count_{k}"] = np.array([[p[j][0].size for j in (0, 1)] for p in pairs])
        for j in (0, 1):
            out[f"premaptime_{k}_{j}"] = _pad([s[0][j] for s in sims], width[j])
            out[f"premapdmdt_{k}_{j}"] = _pad([s[1][j] for s in sims], width[j])
            for m, name in enumerate(("mapped_time", "beta_slope", "beta_yinter")):
                out[f"{name}_{k}_{j}"] = _pad([p[j][m] for p in pairs], width[j])
    out["provenance"] = np.array(json.dumps({
        "method": "MOSFiT 2.1.1 Fallback.__init__ (mosfit/modules/engines/fallback.py:90-239), "
                  "whisper_cbpf.models.jax.tde_fallback.build_fallback_tables",
        "padding": "each row padded to the longest branch by repeating its last node",
        "files_sha256": files}, sort_keys=True))
    return out


@lru_cache(maxsize=1)
def fallback_tables():
    """``{gamma: {name: array}}`` for ``"4-3"`` and ``"5-3"``: MOSFiT's fall-back nodes, as NumPy.

    ``sim_beta`` (n,), ``premap_count`` (n, 2), ``mapped_count`` (n - 1, 2), and per branch a pair
    ``(rising, falling)`` of ``premaptime`` and ``premapdmdt`` (n, width) and of ``mapped_time``,
    ``beta_slope`` and ``beta_yinter`` (n - 1, width); see :func:`build_fallback_tables`. Read
    from :data:`TABLE_FILE` on first use, so importing this module reads nothing.
    """
    with np.load(TABLE_FILE) as z:
        out = {}
        for gm in _GAMMAS:
            k = gm.replace("-", "_")
            tab = {name: z[f"{name}_{k}"] for name in ("sim_beta", "premap_count", "mapped_count")}
            for name in _BRANCH_ARRAYS:
                tab[name] = (z[f"{name}_{k}_0"], z[f"{name}_{k}_1"])
            out[gm] = tab
        return out


# --- the engine: MOSFiT's Fallback.process, in the trace --------------------------------------

def _beta_nodes(beta, gamma):
    """MOSFiT's beta step for one gamma: per branch, the ``(log10 t, log10 dM/dt)`` nodes at
    ``beta`` and the number of valid ones. ``beta`` is held in the simulated range."""
    # ORIGINAL (mosfit/modules/engines/fallback.py:312-372):
    #     for i in range(len(self._sim_beta[g])):
    #         if self._betas[g] == self._sim_beta[g][i]:
    #             beta_interp = False; interp_index_low = i; break
    #         if self._betas[g] < self._sim_beta[g][i]:
    #             interp_index_high = i; interp_index_low = i - 1; beta_interp = True; break
    #     if beta_interp:
    #         dmdt = np.array([
    #             self._beta_yinter[g][interp_index_low][0] +
    #             self._beta_slope[g][interp_index_low][0] * self._betas[g], ...[1]...])
    #         for i in [0, 1]:
    #             time_betalo = (self._mapped_time[g][interp_index_low][i] *
    #                 (self._premaptime[g][interp_index_low][i][-1] -
    #                  self._premaptime[g][interp_index_low][i][0]) +
    #                 self._premaptime[g][interp_index_low][i][0])
    #             time_betahi = (self._mapped_time[g][interp_index_low][i] *
    #                 (self._premaptime[g][interp_index_high][i][-1] -
    #                  self._premaptime[g][interp_index_high][i][0]) +
    #                 self._premaptime[g][interp_index_high][i][0])
    #             time.append(time_betalo + (time_betahi - time_betalo) *
    #                 (self._betas[g] - self._sim_beta[g][interp_index_low]) /
    #                 (self._sim_beta[g][interp_index_high] - self._sim_beta[g][interp_index_low]))
    #     elif not beta_interp:
    #         timedict[g] = np.copy(self._premaptime[g][interp_index_low])
    #         dmdtdict[g] = np.copy(self._premapdmdt[g][interp_index_low])
    # The loop is a searchsorted, the two branches a where. A beta outside the simulations (only
    # for bb outside [0, 2]) is held at the edge, where MOSFiT would index the pair -1 or fail.
    tab = fallback_tables()[gamma]
    sim = jnp.asarray(tab["sim_beta"])
    n = sim.shape[0]
    b = jnp.clip(beta, sim[0], sim[-1])
    k = jnp.clip(jnp.searchsorted(sim, b, side="left"), 0, n - 1)
    exact = sim[k] == b                              # the equality branch: that file's own nodes
    lo = jnp.clip(k - 1, 0, n - 2)
    hi = lo + 1
    count = jnp.where(exact, jnp.asarray(tab["premap_count"])[k],
                      jnp.asarray(tab["mapped_count"])[lo])
    out = []
    for j in (0, 1):
        pt, pd, mt, sl, yi = (jnp.asarray(tab[name][j]) for name in _BRANCH_ARRAYS)
        time_betalo = mt[lo] * (pt[lo, -1] - pt[lo, 0]) + pt[lo, 0]
        time_betahi = mt[lo] * (pt[hi, -1] - pt[hi, 0]) + pt[hi, 0]
        time = time_betalo + (time_betahi - time_betalo) * (b - sim[lo]) / (sim[hi] - sim[lo])
        dmdt = yi[lo] + sl[lo] * b
        out.append((jnp.where(exact, pt[k], time), jnp.where(exact, pd[k], dmdt), count[j]))
    return out


def _search_right(xp, x):
    """``jnp.searchsorted(xp, x, side="right")`` for sorted, NaN-free ``xp``: its binary search,
    unrolled (``method="scan_unrolled"``), with the first step taken at the middle node by a static
    index. ``scan_unrolled`` starts every query from the same ``(low, high)``, so under ``vmap`` XLA
    folds the first step's gather indices into a constant of batch x epochs entries, and the
    compiled program grew with the batch (on a GPU 2.0 MB at one parameter set, 18.4 MB at 2048);
    with this it stays at 2.0 MB. The same indices, so the same values."""
    n = xp.shape[0]
    mid = n // 2
    go_left = x < xp[mid]
    lo, hi = jnp.where(go_left, 0, mid), jnp.where(go_left, mid, n)
    for _ in range(int(np.ceil(np.log2(n + 1))) - 1):
        mid = lo + (hi - lo) // 2
        go_left = x < xp[mid]
        lo, hi = jnp.where(go_left, lo, mid), jnp.where(go_left, mid, hi)
    return hi


def _interp(x, xp, fp, left=None, right=None):
    """``jnp.interp(x, xp, fp, left, right)``: the same arithmetic (under ``jit``, over 48 prior
    draws, the engine's values and gradient are those with ``jnp.interp``, bit for bit), with the
    unrolled search :func:`_search_right`, for the final evaluation (1000 dense epochs on 1091
    nodes). On a GPU the default search loop costs a kernel launch per step: one evaluation at one
    parameter set (30 epochs x 6 bands, median of 300) took 0.79 to 0.97 ms with it and 0.74 to
    0.81 ms unrolled for ``tde_fallback``, 0.97 to 1.25 and 0.78 to 0.80 ms for ``tde_mosfit``.
    The unrolled search compiles longer, so the gamma blend keeps ``jnp.interp``."""
    x, xp, fp = jnp.asarray(x), jnp.asarray(xp), jnp.asarray(fp)
    i = jnp.clip(_search_right(xp, x), 1, xp.shape[0] - 1)
    df = fp[i] - fp[i - 1]
    dx = xp[i] - xp[i - 1]
    delta = x - xp[i - 1]
    flat = jnp.abs(dx) <= np.spacing(np.finfo(dx.dtype).eps)    # a repeated node: no division
    f = jnp.where(flat, fp[i - 1], fp[i - 1] + (delta / jnp.where(flat, 1, dx)) * df)
    f = jnp.where(x < xp[0], fp[0] if left is None else left, f)
    return jnp.where(x > xp[-1], fp[-1] if right is None else right, f)


def _mapped(log_t, count):
    """A padded curve's log times mapped onto [0, 1], first node 0 and every node from the last
    valid one on 1, as MOSFiT maps them."""
    idx = jnp.arange(log_t.shape[-1])
    m = 1. / (log_t[-1] - log_t[0]) * (log_t - log_t[0])
    return jnp.where(idx >= count - 1, 1.0, jnp.where(idx == 0, 0.0, m))


def _gamma_blend(nodes43, nodes53, gfrac):
    """MOSFiT's 4/3-5/3 blend, per branch: the sparser curve mapped onto the denser one's nodes,
    then log t and log dM/dt interpolated by ``gfrac`` (0 for 4/3, 1 for 5/3)."""
    # ORIGINAL (mosfit/modules/engines/fallback.py:376-440):
    #     if len(timedict['4-3'][j]) < len(timedict['5-3'][j]):
    #         interp = '4-3'; nointerp = '5-3'
    #     else:                    # will also catch case where they have the same lengths
    #         interp = '5-3'; nointerp = '4-3'
    #     mapped_time[nointerp].append(1. / (timedict[nointerp][j][-1] - timedict[nointerp][j][0]) *
    #                                  (timedict[nointerp][j] - timedict[nointerp][j][0]))
    #     mapped_time[interp].append(...same for interp...)
    #     mapped_time[interp][j][0] = 0; mapped_time[interp][j][-1] = 1
    #     mapped_time[nointerp][j][0] = 0; mapped_time[nointerp][j][-1] = 1
    #     dmdtdict[interp][j] = _lin_interp(mapped_time[nointerp][j], mapped_time[interp][j],
    #                                       dmdtdict[interp][j])
    #     if interp == '5-3':
    #         time53 = (mapped_time['4-3'][j] * (timedict['5-3'][j][-1] - timedict['5-3'][j][0]) +
    #                   timedict['5-3'][j][0])
    #         time.extend(10 ** (timedict['4-3'][j] + (time53 - timedict['4-3'][j]) * gfrac))
    #     else:
    #         time43 = (mapped_time['5-3'][j] * (timedict['4-3'][j][-1] - timedict['4-3'][j][0]) +
    #                   timedict['4-3'][j][0])
    #         time.extend(10 ** (time43 + (timedict['5-3'][j] - time43) * gfrac))
    #     dmdt.extend(10 ** (dmdtdict['4-3'][j] + (dmdtdict['5-3'][j] - dmdtdict['4-3'][j]) * gfrac))
    # A where on the two node counts picks which curve is mapped (the arrays have the same padded
    # size); the result has the denser curve's nodes and count. The powers of ten are taken by the
    # caller, on the concatenated branches.
    out = []
    for (t43, d43, n43), (t53, d53, n53) in zip(nodes43, nodes53):
        on53 = n43 < n53                      # interp == '4-3': the 5/3 curve is the denser
        t_den, n_den = jnp.where(on53, t53, t43), jnp.where(on53, n53, n43)
        t_sp, d_sp = jnp.where(on53, t43, t53), jnp.where(on53, d43, d53)
        m_den, m_sp = _mapped(t_den, n_den), _mapped(t_sp, jnp.where(on53, n43, n53))
        d_on = jnp.interp(m_den, m_sp, d_sp)              # dmdtdict[interp][j], mapped
        t_on = m_den * (t_sp[-1] - t_sp[0]) + t_sp[0]     # time43 or time53
        t_43, t_53 = jnp.where(on53, t_on, t43), jnp.where(on53, t53, t_on)
        d_43, d_53 = jnp.where(on53, d_on, d43), jnp.where(on53, d53, d_on)
        out.append((t_43 + (t_53 - t_43) * gfrac, d_43 + (d_53 - d_43) * gfrac, n_den))
    return out


def stellar_radius(mstar):
    """Main-sequence radius [solar radii] at solar-like metallicity (Z = 0.0134), Tout et al.

    Masses below 0.1 solar masses are held at 0.1, as in redback.
    """
    mass = jnp.maximum(mstar, 0.1)
    lz = math.log10(0.0134 / 0.02)
    theta = 1.71535900 + 0.62246212 * lz - 0.92557761 * lz**2 - 1.16996966 * lz**3 - 0.30631491 * lz**4
    ll = 6.59778800 - 0.42450044 * lz - 12.13339427 * lz**2 - 10.73509484 * lz**3 - 2.51487077 * lz**4
    kpa = 10.08855000 - 7.11727086 * lz - 31.67119479 * lz**2 - 24.24848322 * lz**3 - 5.33608972 * lz**4
    lbda = 1.01249500 + 0.32699690 * lz - 0.00923418 * lz**2 - 0.03876858 * lz**3 - 0.00412750 * lz**4
    mu = 0.07490166 + 0.02410413 * lz + 0.07233664 * lz**2 + 0.03040467 * lz**3 + 0.00197741 * lz**4
    nu = 0.01077422
    eps = 3.08223400 + 0.94472050 * lz - 2.15200882 * lz**2 - 2.49219496 * lz**3 - 0.63848738 * lz**4
    oo = 17.84778000 - 7.45345690 * lz - 48.9606685 * lz**2 - 40.05386135 * lz**3 - 9.09331816 * lz**4
    pi = 0.00022582 - 0.00186899 * lz + 0.00388783 * lz**2 + 0.00142402 * lz**3 - 0.00007671 * lz**4
    return ((theta * mass**2.5 + ll * mass**6.5 + kpa * mass**11 + lbda * mass**19
             + mu * mass**19.5)
            / (nu + eps * mass**2 + oo * mass**8.5 + mass**18.5 + pi * mass**19.5))


def _fallback_nodes(mbh6, mstar, bb):
    """MOSFiT's final fall-back nodes for one star: ``(t_days, log10 dM/dt [g/s], valid, rstar,
    t_peak, beta)``. ``t_days`` (since the first fall-back) and ``log10 dM/dt`` hold the rising
    branch and then the falling branch, each padded by repeating its last node; ``valid`` marks
    MOSFiT's nodes (``t_days[valid]`` and ``10 ** log10_dmdt[valid]`` are its ``time`` and ``dmdt``
    arrays just before the final interpolation)."""
    # ORIGINAL (mosfit/modules/engines/fallback.py:256-269):
    #     if 0 <= self._b < 1:
    #         beta43 = 0.6 + 1.25 * self._b  # 0.6 + (1.85 - 0.6)*b
    #         beta53 = 0.5 + 0.4 * self._b  # 0.5 + (0.9 - 0.5)*b
    #     elif 1 <= self._b <= 2:
    #         beta43 = 1.85 + 2.15 * (self._b - 1)
    #         beta53 = 0.9 + 1.6 * (self._b - 1)
    # The products are held apart from the sums: under jit, XLA on the CPU fuses a product and a
    # sum into one rounding (a fused multiply-add), which moves 18 % of the betas by one unit in
    # the last place, and the equality branch compares beta with the simulated values exactly.
    prod = jax.lax.optimization_barrier((1.25 * bb, 0.4 * bb, 2.15 * (bb - 1.0), 1.6 * (bb - 1.0)))
    beta43 = jnp.where(bb < 1.0, 0.6 + prod[0], 1.85 + prod[2])
    beta53 = jnp.where(bb < 1.0, 0.5 + prod[1], 0.9 + prod[3])
    # ORIGINAL (mosfit/modules/engines/fallback.py:280-307):
    #     if self._Mstar <= 0.3 or self._Mstar >= 22:
    #         gammas = [self._gammas[1]]; self._beta = self._betas['5-3']
    #     elif 1 <= self._Mstar <= 15:
    #         gammas = [self._gammas[0]]; self._beta = self._betas['4-3']
    #     elif 0.3 < self._Mstar < 1:
    #         gamma_interp = True; gfrac = (self._Mstar - 1.) / (0.3 - 1.)
    #         self._beta = self._betas['5-3'] + (self._betas['4-3'] - self._betas['5-3']) * (1. - gfrac)
    #     elif 15 < self._Mstar < 22:
    #         gamma_interp = True; gfrac = (self._Mstar - 15.) / (22. - 15.)
    #         self._beta = self._betas['5-3'] + (self._betas['4-3'] - self._betas['5-3']) * (1. - gfrac)
    only53 = (mstar <= 0.3) | (mstar >= 22.0)
    only43 = (mstar >= 1.0) & (mstar <= 15.0)
    # In a blend the beta returned is MOSFiT's to one unit in the last place, not to the bit: XLA
    # computes the division by a constant as a product with its reciprocal and may fuse the sum.
    # It reaches only the photosphere; beta43 and beta53, which the equality branch compares, are
    # exact.
    gfrac = jnp.where(mstar < 1.0, (mstar - 1.) / (0.3 - 1.), (mstar - 15.) / (22. - 15.))
    beta = jnp.where(only53, beta53, jnp.where(only43, beta43,
                                               beta53 + (beta43 - beta53) * (1. - gfrac)))
    nodes43 = _beta_nodes(beta43, "4-3")
    nodes53 = _beta_nodes(beta53, "5-3")
    mixed = _gamma_blend(nodes43, nodes53, gfrac)
    # ORIGINAL (mosfit/modules/engines/fallback.py:442-451), without a gamma blend:
    #     time = np.concatenate((timedict[g][0], timedict[g][1])); time = 10 ** time
    #     dmdt = np.concatenate((dmdtdict[g][0], dmdtdict[g][1])); dmdt = 10 ** dmdt
    # Each branch is padded on its own, so the padding of the rising branch repeats its last node
    # before the falling branch begins.
    picked = [[jnp.where(only53, c53[i], jnp.where(only43, c43[i], cm[i])) for i in range(3)]
              for c43, c53, cm in zip(nodes43, nodes53, mixed)]
    log_t = jnp.concatenate([branch[0] for branch in picked])
    log_d = jnp.concatenate([branch[1] for branch in picked])
    valid = jnp.concatenate([jnp.arange(branch[0].shape[-1]) < branch[2] for branch in picked])

    rstar = stellar_radius(mstar)
    mh = mbh6 * 1.0e6
    # ORIGINAL (mosfit/modules/engines/fallback.py:479-486 and :564-566, resttexplosion = 0; redback
    # tde_models.py:1088-1096 is the same):
    #     dmdt = (dmdt * np.sqrt(Mhbase / self._Mh) *
    #             (self._Mstar / Mstarbase) ** 2.0 * (Rstarbase / Rstar) ** 1.5)
    #     time = (time * np.sqrt(self._Mh / Mhbase) * (Mstarbase / self._Mstar) *
    #             (Rstar / Rstarbase) ** 1.5)
    #     time = time / DAY_CGS
    #     tfallback = np.copy(time[0])
    #     time = time - tfallback + self._rest_t_explosion
    #     tpeak = time[np.argmax(dmdt)]
    # dM/dt is scaled in log10 (it is 1e20 to 1e28 g/s).
    t_days = jnp.power(10.0, log_t) * jnp.sqrt(mh / 1.0e6) * (1.0 / mstar) * rstar ** 1.5 / DAY_TO_S
    # held apart from the subtraction, which XLA could otherwise fuse with the last product into
    # one rounding: the first node could then sit at a few 1e-16 d instead of MOSFiT's exact 0
    t_days = jax.lax.optimization_barrier(t_days)
    t_days = t_days - t_days[0]                      # time since the first fall-back
    log_d = log_d + jnp.log10(jnp.sqrt(1.0e6 / mh) * mstar ** 2.0 * (1.0 / rstar) ** 1.5)
    t_peak = t_days[jnp.argmax(log_d)]
    return t_days, log_d, valid, rstar, t_peak, beta


def fallback_engine(times, mbh6, mstar, bb, eta, leddlimit):
    """Fall-back luminosity at source-frame ``times`` [days since first fall-back], and the
    quantities the photosphere needs: MOSFiT's ``Fallback.process`` (redback's
    ``_tde_mosfit_engine``) on the nodes of :func:`fallback_tables`.

    Returns ``(log10 L [erg/s], star_radius [Rsun], t_peak [d], beta)``: the Eddington-capped
    ``eta dM/dt c^2`` before any viscous delay (floored at 1e-30 erg/s outside the nodes, where
    MOSFiT and redback give 0), the Tout radius, the time of the maximum rate and the physical
    penetration factor. ``dM/dt`` is linear in time between the nodes, as in MOSFiT; it is carried
    relative to its maximum and the luminosity in log10, so the engine holds in float32. ``bb``
    should lie in [0, 2] (redback raises outside it; here the rate holds the table edge, and the
    ``beta`` returned is the unclipped mapping of ``bb``). As in MOSFiT and redback the rate jumps
    where a beta equals a simulated one (the equality branch) and at the limits of the gamma
    regimes, ``mstar`` = 0.3, 1, 15 and 22, so it is piecewise smooth in ``bb`` and ``mstar``. The
    gradient is finite everywhere; at a jump it is the derivative of the side that is selected, not
    a one-sided limit: at a simulated beta that simulation's own curve, which for a star of one
    gamma does not depend on ``bb`` (derivative 0 there).
    """
    t_days, log_d, _, rstar, t_peak, beta = _fallback_nodes(mbh6, mstar, bb)
    # ORIGINAL (mosfit/modules/engines/fallback.py:568-578; redback tde_models.py:1098-1108 with
    # interp1d):
    #     lengthpretimes = int(np.count_nonzero(times_eval < time[0]))
    #     lengthposttimes = int(np.count_nonzero(times_eval > time[-1]))
    #     mid = times_eval[lengthpretimes:n_times - lengthposttimes]
    #     dmdt2 = _lin_interp(mid, time, dmdt) if mid.size else np.empty(0)
    #     dmdtnew = np.concatenate((np.zeros(lengthpretimes), dmdt2, np.zeros(lengthposttimes)))
    # Linear in dM/dt, which is carried relative to its maximum (a constant of the gradient: the
    # result does not depend on it); zero outside the nodes becomes the floor. The padding repeats
    # a node, a zero-length segment that changes no interpolated value.
    log_top = jax.lax.stop_gradient(jnp.max(log_d))
    rate = _interp(times, t_days, jnp.power(10.0, log_d - log_top), left=0.0, right=0.0)
    inside = rate > 0.0
    log_rate = jnp.where(inside, jnp.log10(jnp.where(inside, rate, 1.0)) + log_top,
                         _LOG10_RATE_OUTSIDE)
    # ORIGINAL (mosfit/modules/engines/fallback.py:580-589):
    #     luminosities = (self._efficiency * dmdtnew * c.c.cgs.value * c.c.cgs.value)
    #     Ledd = (FOUR_PI * c.G.cgs.value * self._Mh * M_SUN_CGS * C_CGS / kappa_t)
    #     ledd_cap = self._Leddlim * Ledd
    #     luminosities = luminosities * ledd_cap / (luminosities + ledd_cap)
    # in log10, the sum taken relative to its larger term.
    mh = mbh6 * 1.0e6
    log_free = jnp.log10(eta) + log_rate + _LOG10_C2
    log_cap = jnp.log10(leddlimit) + _LOG10_EDD_PER_MSUN + jnp.log10(mh)
    mx = jnp.maximum(log_free, log_cap)
    log_sum = mx + jnp.log10(jnp.power(10.0, log_free - mx) + jnp.power(10.0, log_cap - mx))
    return jnp.maximum(log_free + log_cap - log_sum, _LOG10_L_FLOOR), rstar, t_peak, beta


def _require_segments(n_nodes, what):
    """``geometric=True`` needs at least two dense nodes: with one, the arithmetic segment search
    divides by ``n - 1 = 0`` and every value is NaN. Called only when ``geometric`` is True, on a
    static count, so it adds nothing to a traced program."""
    if n_nodes < 2:
        raise ValueError(f"{what}: geometric=True needs at least 2 dense nodes (dense_resolution "
                         f">= 2; redback's is 1000), got {n_nodes}: with fewer the arithmetic "
                         f"segment search has no segment and gives NaN. Pass a larger "
                         f"dense_resolution, or geometric=False.")


def viscous_log10(times, dense_times, log10_luminosity, tvisc, *, geometric=False):
    """redback's exponential viscous convolution, evaluated at ``times``, in log10.

    ``L(t) = int L'(t') exp((t' - t) / tvisc) dt' / tvisc`` by the trapezoid rule on redback's nodes:
    ``t' = tb + (t - tb) x`` with ``x`` the sorted union of ``logspace(log10(tvisc / t_end) - 3, 0,
    500)`` and its mirror about 1, ``tb = dense_times[0]``, clipped to the dense grid. The integrand
    is carried relative to its maximum, so nothing overflows in float32; the maximum is a constant
    of the gradient (the result does not depend on it).

    ``log10_luminosity`` is interpolated linearly between the dense nodes. ``geometric=False``
    (default) finds each node's segment by a search (``jnp.interp``). ``geometric=True`` finds it
    by arithmetic, ``floor(log(t' / dense_times[0]) / log step)``
    (:func:`whisper_cbpf.models.jax.tde._interp_geometric`, as the supernova diffusion does), and
    then needs ``dense_times`` from :func:`geometric_dense_times` (geometric to round-off, its
    endpoints exact). It is porting rule (a): the same piecewise-linear interpolant, computed with
    ``jnp.interp``'s arithmetic in the same order. The value is identical where the same segment is
    chosen, and differs by round-off where a node within rounding of a dense node takes the
    neighbouring segment.

    The default stays False here, while both factories and the functions that build their own
    grid with :func:`geometric_dense_times` (:func:`tde_fallback_bolometric_log10`,
    :func:`tde_fallback_ab_magnitude` and, in ``tde_mosfit``, the curve, the magnitude, the
    constraint score and the flux density) default to True: this function takes the caller's
    ``dense_times``, and on a valid grid that is not geometric (any increasing one) the arithmetic
    picks wrong segments and gives a wrong number. Pass True only with a grid from
    :func:`geometric_dense_times`, of at least 2 nodes (``ValueError`` otherwise).
    """
    if geometric:
        _require_segments(np.shape(dense_times)[0], "viscous_log10")
    tb = dense_times[0]
    lsp = jnp.logspace(jnp.log10(tvisc / dense_times[-1]) - 3.0, 0.0, 500)
    x = jnp.sort(jnp.concatenate((lsp, 1.0 - lsp)))
    nodes = jnp.clip(tb + (times[:, None] - tb) * x[None, :], tb, dense_times[-1])
    log_l = (_interp_geometric(nodes, dense_times, log10_luminosity) if geometric
             else jnp.interp(nodes, dense_times, log10_luminosity))
    arg = log_l + (nodes - nodes[:, -1:]) / tvisc * _LOG10_E
    scale = jax.lax.stop_gradient(jnp.max(arg, axis=1, keepdims=True))
    integral = jnp.trapezoid(jnp.power(10.0, arg - scale), nodes, axis=1) / tvisc
    return scale[:, 0] + jnp.log10(jnp.maximum(integral, 1e-30))


def tde_fallback_bolometric_log10(time, mbh6, mstar, tvisc, bb, eta, leddlimit, *,
                                  dense_resolution=1000, geometric=True):
    """log10 of redback's ``tde_fallback_bolometric`` [erg/s] at ``time``.

    ``time`` is source-frame days since the first fall-back, every entry positive; the largest
    sizes the dense engine grid (redback uses ``time[-1] + 100`` d). ``mbh6`` in 1e6 solar masses,
    ``mstar`` in solar masses, ``tvisc`` in days, ``bb`` the impact-parameter mapping in [0, 2],
    ``eta`` the radiative efficiency, ``leddlimit`` the multiple of the Eddington luminosity at
    which the rate is capped. Traceable in every argument; float32 is enough for the values, except
    at a pinned round ``bb``, and not for the derivative in ``mstar`` above about 11 solar masses
    (see the module's **Precision**). ``geometric`` (default True) finds the viscous kernel's
    segments by arithmetic on the geometric dense grid this function builds (rule (a), the same
    interpolant; see :func:`viscous_log10`); False finds them by a search (``jnp.interp``), the
    function before the option existed.
    """
    time = jnp.asarray(time)
    dense = geometric_dense_times(jnp.max(time), dense_resolution)
    log_l, _, _, _ = fallback_engine(dense, mbh6, mstar, bb, eta, leddlimit)
    return viscous_log10(time, dense, log_l, tvisc, geometric=geometric)


def tde_fallback_ab_magnitude(tau_days, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
                              mbh6, mstar, tvisc, bb, eta, leddlimit, rph0, lphoto, *,
                              epoch_idx=None, dense_resolution=1000, geometric=True,
                              mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None,
                              xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99"):
    """AB magnitude of redback's ``tde_fallback`` per observation. Traced; **float64 only**.

    ``tau_days`` are SOURCE-frame days since the first fall-back (``(t - t_exp) / (1 + z)``); an
    epoch at or before 0 is at ``mag_floor``. ``weights, norms`` come from
    :func:`whisper_cbpf.models.jax.kilonova.ab_weights` and ``lam_obs_ang`` is its grid, as for
    :func:`whisper_cbpf.models.jax.tde.ab_magnitude`, whose ``mag_floor``, ``dilation`` and
    extinction arguments these are. ``rph0`` and ``lphoto`` are the photosphere's
    (:func:`whisper_cbpf.models.jax.tde.tde_photosphere`).

    ``epoch_idx`` is ``None`` (every observation evaluated) or ``(first, inv)``, int32, from
    ``predict_jax.epoch_index(times[, size])``. With it, the engine, the viscous integral and the
    photosphere run at ``tau[first]`` and ``T``, ``R`` are copied back with ``inv`` before the band
    integral: the same numbers, to round-off (porting rule (a); redback's and MOSFiT's ``Viscous``
    do the same). A map that does not describe ``tau`` makes every magnitude NaN. The derivative
    with respect to ``tau_days`` itself is then per distinct epoch, carried by its first
    observation.

    ``geometric`` (default True) finds the viscous kernel's segments by arithmetic on the geometric
    dense grid this function builds (rule (a), the same interpolant; see :func:`viscous_log10`);
    False finds them by a search (``jnp.interp``). ``epoch_idx=None, geometric=False`` is the code
    before these options existed.
    """
    from ._adopted_tde import require_x64

    require_x64("tde_fallback")
    tau = jnp.asarray(tau_days)
    alive = tau > 0.0
    tau_e = _at_epochs(tau, epoch_idx)                        # the distinct epochs, or tau
    alive_e = alive if epoch_idx is None else tau_e > 0.0
    t_eval = jnp.where(alive_e, jnp.maximum(tau_e, GRID_START), 1.0)     # finite when dead
    dense = geometric_dense_times(jnp.max(jnp.where(alive_e, tau_e, 0.0)), dense_resolution)
    log_l, rstar, t_peak, beta = fallback_engine(dense, mbh6, mstar, bb, eta, leddlimit)
    lum = jnp.power(10.0, viscous_log10(t_eval, dense, log_l, tvisc, geometric=geometric))
    temp, rad, _ = tde_photosphere(t_eval, lum, mbh6 * 1.0e6, mstar, rstar, t_peak, beta,
                                   rph0, lphoto)
    # the band integral stays per observation: T and R copied to the rows of their epoch
    mag = ab_magnitude(_to_rows(temp, epoch_idx), _to_rows(rad, epoch_idx), band_idx, weights,
                       norms, lam_obs_ang, redshift, dl_cm,
                       mag_floor=mag_floor, dilation=dilation, ebv_mw=ebv_mw, ebv_host=ebv_host,
                       xi_mw=xi_mw, xi_host=xi_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)
    mag = jnp.where(alive, mag, jnp.asarray(mag_floor, mag.dtype))
    return _guard_epochs(mag, tau, epoch_idx)


# --- priors ---------------------------------------------------------------------------------

#: Transcribed from redback 1.20's ``tde_fallback.prior``, for when redback is not importable.
#: ``leddlimit = 1`` is a fixed value there, so it is pinned, not sampled.
_FALLBACK_PRIOR = (
    {"mbh6": ("Uniform", 0.01, 20.0), "mstar": ("Uniform", 0.1, 30.0),
     "tvisc": ("LogUniform", 1e-3, 1e2), "bb": ("Uniform", 0.0, 2.0),
     "eta": ("LogUniform", 1e-4, 0.4), "rph0": ("LogUniform", 1e-4, 1e4),
     "lphoto": ("Uniform", 0.0, 2.0)},
    {"leddlimit": 1.0},
)


def fallback_prior():
    """:func:`default_prior` without redback: the transcription of its 1.20 prior file."""
    from ...priors import LogUniform, Prior, Uniform

    dists, pinned = _FALLBACK_PRIOR
    cls = {"Uniform": Uniform, "LogUniform": LogUniform}
    return Prior({k: cls[c](lo, hi) for k, (c, lo, hi) in dists.items()}), dict(pinned)


def default_prior():
    """``(Prior, pinned)`` of redback's ``tde_fallback``: its prior file, read from redback where
    possible (``redshift`` dropped: it is the factory's argument), else :func:`fallback_prior`.

    ``mbh6`` Uniform(0.01, 20) 1e6 Msun, ``mstar`` Uniform(0.1, 30) Msun, ``tvisc``
    LogUniform(1e-3, 100) d, ``bb`` Uniform(0, 2), ``eta`` LogUniform(1e-4, 0.4), ``rph0``
    LogUniform(1e-4, 1e4), ``lphoto`` Uniform(0, 2); pinned ``leddlimit = 1``.
    """
    from .tde import redback_prior

    try:
        return redback_prior("tde_fallback")
    except ImportError:
        return fallback_prior()


# --- the whisper model ----------------------------------------------------------------------

def tde_fallback_model(band_names, redshift=None, dl_cm=None, *, name="tde_fallback_jax",
                       n_wave=None, dense_resolution=1000, pin=None, prior=None, mag_floor=40.0,
                       dilation=True, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1,
                       law="f99", filter_set=None, band_aliases=None, default_system=None,
                       t_exp_days=0.0, free=None, redshift_prior=None, unique_epochs=True,
                       geometric=True):
    """Bind dataset context to redback's ``tde_fallback`` and return a whisper ``Model``.

    Photometric, so a factory rather than a registry entry: ``predict(parameters, times, bands)``
    carries no filter set, redshift or distance, and a guessed distance would rescale every fitted
    black-hole mass. See :func:`whisper_cbpf.models.jax.tde_model`, whose arguments these follow.

    Parameters
    ----------
    band_names : sequence of str
        Filter names or labels, in the order the light curve's ``bands`` column maps to (bare
        letters are LSST unless ``default_system=``).
    redshift, dl_cm : float
        Fixed for the dataset; leave both ``None`` when the redshift is fitted
        (``free=["redshift"]``).
    name : str
        The model's name (default ``"tde_fallback_jax"``).
    dense_resolution : int
        Points of the engine grid (redback's 1000); at least 2 with ``geometric=True``
        (``ValueError`` here otherwise).
    pin : dict or None
        Override which parameters are held fixed: ``{"bb": 1.0}`` pins one, ``{"leddlimit": None}``
        frees redback's pinned ``leddlimit`` (and needs ``prior=`` for it).
    prior : whisper_cbpf.priors.Prior or None
        Distributions overriding redback's, per parameter. The default is redback's own prior
        (:func:`default_prior`), not modified.
    mag_floor, dilation, ebv_mw, ebv_host, r_v_mw, r_v_host, law, filter_set, n_wave, band_aliases, default_system :
        As :func:`whisper_cbpf.models.jax.tde_model` (Gauss-16 band integral by default, the
        ``(1 + z)`` flux factor of redback's TDE path, optional extinction).
    t_exp_days : float
        Time of the first fall-back on the light curve's own clock (the model's time zero),
        subtracted on the host in float64. Default 0: ``times`` are days since it.
    free : list of str or None
        ``"t_exp"`` and/or ``"redshift"`` fitted as parameters, appended in that order; see
        :func:`whisper_cbpf.models.jax.tde_model`. ``t_exp`` needs ``prior=Prior({"t_exp": ...})``.
    redshift_prior : dict, distribution or None
        The fitted redshift's prior when ``prior`` has none.
    unique_epochs : bool
        Evaluate the engine, the viscous delay and the photosphere once per distinct epoch and copy
        them to the observations of that epoch (default True; exact to round-off, see
        PORTING_NOTES). False evaluates every observation, with ``geometric=False`` the code before
        these options existed: the same numbers to round-off, slower on multi-band light curves.
    geometric : bool
        Default True: the viscous kernel finds its segments on the geometric dense grid by
        arithmetic instead of a search (porting rule (a), the same interpolant; see
        :func:`viscous_log10`). False keeps the search. ``unique_epochs=False, geometric=False`` is
        the code before these options existed. ``geometric`` enters the model's state (and so its
        ``model_record`` identity) only when True: a ``geometric=False`` model keeps the identity
        it had before the option existed, and the default model is identified as geometric.

    Returns
    -------
    Model
        ``parameters`` ``["mbh6", "mstar", "tvisc", "bb", "eta", "rph0", "lphoto"]`` (plus the freed
        context values). redback declares no constraints for this model, so there is no wall.

    References
    ----------
    Mockler, Guillochon & Ramirez-Ruiz 2019, ApJ 872, 151; Guillochon & Ramirez-Ruiz 2013, ApJ 767,
    25; Guillochon et al. 2018, ApJS 236, 6 (MOSFiT); Tout et al. 1996, MNRAS 281, 257 (the stellar
    radius); Sarin et al. 2024, MNRAS 531, 1203 (redback). Full list in the module docstring.

    Notes
    -----
    **float64 is required** (``jax.config.update("jax_enable_x64", True)`` before the first array
    exists); the guard raises at the first ``predict``, not here. A light curve depends on the
    other epochs only through the end of the engine grid, as in redback (about 1e-3 mag).

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> from whisper_cbpf.models.jax.tde_fallback import tde_fallback_model
    >>> m = tde_fallback_model(["lsstg", "lsstr"], redshift=0.05, dl_cm=7.1e26)
    >>> m.parameters
    ['mbh6', 'mstar', 'tvisc', 'bb', 'eta', 'rph0', 'lphoto']
    """
    if geometric:
        _require_segments(dense_resolution, "tde_fallback_model")
    return assemble(
        "tde_fallback", "tde_fallback", band_names, redshift, dl_cm, name=name, n_wave=n_wave,
        pin=pin, prior=prior, mag_floor=mag_floor, dilation=dilation, ebv_mw=ebv_mw,
        ebv_host=ebv_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law, filter_set=filter_set,
        band_aliases=band_aliases, default_system=default_system, t_exp_days=t_exp_days, free=free,
        redshift_prior=redshift_prior, constraint=None,
        # `geometric` enters the state only when True (the default): a geometric=False model keeps
        # the state (and so the `model_record` identity) it had before the option existed, and
        # `DirectTDEPredict` reads the absent key as geometric=False, whatever the default here
        engine_kw=dict(dense_resolution=dense_resolution,
                       **({"geometric": True} if geometric else {})),
        summary="MOSFiT fall-back (Guillochon tables) with viscous delay and Mockler photosphere",
        caller="tde_fallback_model", unique_epochs=unique_epochs)


def register_tde_fallback(band_names, redshift=None, dl_cm=None, *, name="tde_fallback_jax",
                          **kwargs):
    """Build a ``tde_fallback`` ``Model`` for this dataset and register it under ``name``.

    Parameters
    ----------
    band_names, redshift, dl_cm, **kwargs
        As :func:`tde_fallback_model`.
    name : str, default "tde_fallback_jax"
        The registry name (an existing one is replaced).

    Returns
    -------
    Model

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.jax.tde_fallback import register_tde_fallback
    >>> m = register_tde_fallback(["lsstg", "lsstr"], redshift=0.05, dl_cm=7.1e26)
    >>> m.name in wp.list_models()
    True
    """
    from ._factories import _register

    return _register(tde_fallback_model(band_names, redshift, dl_cm, name=name, **kwargs))
