"""Many alerts per GPU call: K independent emcee ensembles in one compiled on-device loop.

``emcee_jax`` used to drive the emcee library from Python: every half-step returned to the host,
and each light curve was its own compiled density. For a night of alerts both costs dominated: the
first supernova alert paid 24.3 s of compile and fit, and removing the return to Python between
the 20 000 half-steps alone took a fit from 45 s to 17 s. :func:`fit_batch` keeps the whole chain
on the device, inside one compiled loop, and advances K alerts together, so each half-step is ONE
evaluation of K x nwalkers/2 models with ONE compile for every alert of the same model and bucket
size (:func:`~whisper_cbpf.samplers.jax._adapters.log_density` passes the data as arguments).
``emcee_jax`` now runs on this engine too (:func:`_run_ensembles`), with its K ensembles fitting
one light curve.

The move is emcee's ``StretchMove`` (Goodman & Weare 2010, a = 2) with emcee's randomised red-blue
split: each step the walkers are shuffled into two halves; each walker of one half proposes
``y = c + z (x - c)`` towards a random walker ``c`` of the other half, ``z`` drawn from
``g(z) ~ 1/sqrt(z)`` on ``[1/a, a]``, and is accepted with probability
``min(1, z^(D-1) p(y) / p(x))``. An alert's walkers only ever see that alert's walkers, and each
alert draws from its own random stream, so a fit does not depend on what else is in the batch.

No JAX at module scope: importing this module on a CPU-only install is free.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

__all__ = ["fit_batch", "STRETCH_A"]

#: emcee's default stretch scale ``a``.
STRETCH_A = 2.0

#: The sampler name every result of :func:`fit_batch` carries.
SAMPLER_NAME = "emcee_batch"


def _alert_seeds(seed, k):
    """One integer seed per alert: ``seed + i`` for an int, or the given list."""
    if np.ndim(seed) == 0:
        return [int(seed) + i for i in range(k)]
    seeds = [int(s) for s in seed]
    if len(seeds) != k:
        raise ValueError(f"fit_batch: seed has {len(seeds)} entries for {k} light curves; pass "
                         f"one int (alert i then uses seed + i) or one seed per light curve.")
    return seeds


def _alert_priors(prior, k):
    if isinstance(prior, (list, tuple)):
        if len(prior) != k:
            raise ValueError(f"fit_batch: prior has {len(prior)} entries for {k} light curves; "
                             f"pass one Prior for all, or one per light curve.")
        return list(prior)
    return [prior] * k


def _make_loop(program, n_walkers, n_dim, burnin, n_keep, thin, tail, dtype, *, width=None,
               pilot=None, segments=False):
    """``run(keys, x0, data) -> (xs, lps, lls, n_accepted, n_nan, lp0)``, one compiled loop.

    ``keys`` (K, 2), ``x0`` (K, W, D), ``data`` a pytree of (K, ...) arrays. ``xs`` is
    (n_keep, K, W, D): the state after step ``burnin + j * thin + thin - 1`` for j < n_keep --
    exactly the draws ``emcee``'s ``get_chain(discard=burnin, thin=thin)`` keeps. ``lps`` / ``lls``
    are the log posterior / log likelihood there, ``n_accepted`` (K, W) the accepted moves per
    walker, ``n_nan`` (K,) the proposals whose density was NaN (rejected, as -inf).

    Every step runs inside ONE ``lax.fori_loop`` and the kept states are written into
    preallocated buffers, so the model evaluation appears once in the compiled program (a loop
    per phase -- burn-in, the kept blocks, the tail -- put it there three times, and XLA compiled
    each copy).

    ``width`` evaluates each half-ensemble in ``lax.map`` blocks of that many walkers (padded);
    ``None`` is one vmap. ``pilot`` (a dict with ``steps``, ``burnin``, ``keep`` and ``thin``) runs
    that many steps first, keeping ``keep`` states every ``thin`` steps after ``burnin``; then
    every ensemble restarts on its own random draws, without replacement, of the kept pilot states
    of the ensemble that reached the highest log likelihood, and the run above follows. With a
    pilot, ``run`` also returns the pilot's kept states, log posteriors and log likelihoods
    ((keep, K, W, ...)), the index of the ensemble chosen, and the accepted moves per walker over
    the pilot.

    ``segments=True`` returns ``(advance, score, start)`` instead, for a run in chunks (emcee_jax's
    ``until_converged``): ``start(keys, x0, data)`` is the carry before step 0, and
    ``advance(carry, t_begin, t_end, keep_from, n_now, data) -> (carry, (xs, lps, lls))`` runs the
    absolute steps ``t_begin .. t_end - 1`` with the same body (pilot and restart included), keeping
    ``n_now <= n_keep`` states every ``thin`` steps after ``keep_from``. The bounds are arguments,
    so every chunk is a call of the same compiled program; the carry (state, pilot buffers, pilot
    acceptance, restart keys) stays on the device between calls.
    """
    import jax
    import jax.numpy as jnp

    a = STRETCH_A
    n0 = (n_walkers + 1) // 2                 # emcee: arange(W) % 2 -> group 0 has ceil(W / 2)
    n_total = int(burnin) + int(n_keep) * int(thin) + int(tail)
    p_steps = 0 if not pilot else int(pilot["steps"])
    if p_steps:
        p_burn, p_keep, p_thin = int(pilot["burnin"]), int(pilot["keep"]), int(pilot["thin"])
        if p_keep < 1 or p_burn + p_keep * p_thin > p_steps:
            raise ValueError(f"_make_loop: the pilot keeps {p_keep} states every {p_thin} steps "
                             f"after {p_burn}, beyond its {p_steps} steps.")

    def one_block(x, d):
        return jax.vmap(program, in_axes=(0, None))(x, d)

    def evaluate(x, d):
        m = x.shape[0]
        if width is None or m <= int(width):
            return one_block(x, d)
        w = int(width)
        n_blocks = -(-m // w)
        pad = n_blocks * w - m
        xp = jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0) if pad else x
        lp, ll = jax.lax.map(lambda b: one_block(b, d), xp.reshape(n_blocks, w, x.shape[-1]))
        return lp.reshape(-1)[:m], ll.reshape(-1)[:m]

    def lnp(x, d):
        lp, ll = evaluate(x, d)
        nan = jnp.isnan(lp)
        return (jnp.where(nan, -jnp.inf, lp), jnp.where(nan, -jnp.inf, ll),
                jnp.sum(nan).astype(jnp.int32))

    def half(key, x, lp, ll, moving, other, d):
        kz, kr, ku = jax.random.split(key, 3)
        ns, nc = moving.shape[0], other.shape[0]
        xs, xc = x[moving], x[other]
        z = ((a - 1.0) * jax.random.uniform(kz, (ns,), dtype) + 1.0) ** 2 / a
        c = xc[jax.random.randint(kr, (ns,), 0, nc)]
        y = c - (c - xs) * z[:, None]
        lpy, lly, n_nan = lnp(y, d)
        lnr = (n_dim - 1.0) * jnp.log(z) + lpy - lp[moving]
        acc = jnp.log(jax.random.uniform(ku, (ns,), dtype)) < lnr
        x = x.at[moving].set(jnp.where(acc[:, None], y, xs))
        lp = lp.at[moving].set(jnp.where(acc, lpy, lp[moving]))
        ll = ll.at[moving].set(jnp.where(acc, lly, ll[moving]))
        n_acc = jnp.zeros(x.shape[0], jnp.int32).at[moving].set(acc.astype(jnp.int32))
        return x, lp, ll, n_acc, n_nan

    def step_one(key, x, lp, ll, n_acc, n_nan, d):
        key, kp, k0, k1 = jax.random.split(key, 4)
        perm = jax.random.permutation(kp, n_walkers)          # emcee's randomize_split
        g0, g1 = perm[:n0], perm[n0:]
        x, lp, ll, a0, b0 = half(k0, x, lp, ll, g0, g1, d)
        x, lp, ll, a1, b1 = half(k1, x, lp, ll, g1, g0, d)
        return key, x, lp, ll, n_acc + a0 + a1, n_nan + b0 + b1

    step = jax.vmap(step_one)

    def put(buf, j, do, value):
        return buf.at[j].set(jnp.where(do, value, buf[j]))

    def kept(t, start, n, every):
        """(write?, slot) for step t of a phase that keeps n states every `every` after start."""
        s = t - start
        write = (s >= 0) & (s % every == every - 1) & (s // every < n)
        return write, jnp.clip(s // every, 0, n - 1)

    def best_ensemble(plls):
        return jnp.argmax(jnp.max(plls, axis=(0, 2)))

    def restart(state, pbufs, rkeys):
        keys, x, lp, ll, n_acc, n_nan = state
        pxs, plps, plls = pbufs
        b = best_ensemble(plls)
        pool_x = pxs[:, b].reshape(-1, n_dim)
        pool_lp, pool_ll = plps[:, b].reshape(-1), plls[:, b].reshape(-1)

        def pick(k):
            idx = jax.random.choice(k, pool_x.shape[0], (n_walkers,), replace=False)
            return pool_x[idx], pool_lp[idx], pool_ll[idx]

        x, lp, ll = jax.vmap(pick)(rkeys)
        return keys, x, lp, ll, jnp.zeros_like(n_acc), n_nan

    def run(keys, x0, data):
        lp0, ll0, _ = jax.vmap(lnp)(x0, data)
        k = x0.shape[0]
        state = (keys, x0, lp0, ll0, jnp.zeros((k, n_walkers), jnp.int32),
                 jnp.zeros((k,), jnp.int32))
        bufs = (jnp.zeros((n_keep, k, n_walkers, n_dim), x0.dtype),
                jnp.zeros((n_keep, k, n_walkers), lp0.dtype),
                jnp.zeros((n_keep, k, n_walkers), ll0.dtype))
        if p_steps:
            # the restart's own random stream, apart from the steps'
            rkeys = jax.vmap(lambda kk: jax.random.fold_in(kk, 0x5EED))(keys)
            pbufs = (jnp.zeros((p_keep, k, n_walkers, n_dim), x0.dtype),
                     jnp.zeros((p_keep, k, n_walkers), lp0.dtype),
                     jnp.zeros((p_keep, k, n_walkers), ll0.dtype))
        else:
            pbufs = ()

        def body(t, c):
            state, bufs, pbufs, p_acc = c
            if p_steps:
                p_acc = jnp.where(t == p_steps, state[4], p_acc)
                state = jax.lax.cond(t == p_steps, lambda s, p: restart(s, p, rkeys),
                                     lambda s, p: s, state, pbufs)
            state = step(*state, data)
            now = (state[1], state[2], state[3])
            do, j = kept(t, p_steps + burnin, n_keep, thin)
            bufs = tuple(put(b, j, do, v) for b, v in zip(bufs, now))
            if p_steps:
                do, j = kept(t, p_burn, p_keep, p_thin)
                do = do & (t < p_steps)
                pbufs = tuple(put(b, j, do, v) for b, v in zip(pbufs, now))
            return state, bufs, pbufs, p_acc

        p_acc0 = jnp.zeros((k, n_walkers), jnp.int32)
        state, bufs, pbufs, p_acc = jax.lax.fori_loop(0, p_steps + n_total, body,
                                                      (state, bufs, pbufs, p_acc0))
        out = (*bufs, state[4], state[5], lp0)
        if p_steps:
            out = out + (*pbufs, best_ensemble(pbufs[2]), p_acc)
        return out

    def score(x, data):                                    # (K, W, D) -> (K, W) log posterior
        return jax.vmap(lnp)(x, data)[0]

    if not segments:
        return jax.jit(run), jax.jit(score)

    # --- segments: the same steps, run in chunks by repeated calls of ONE compiled program -----
    def start(keys, x0, data):
        """The carry before step 0. The log densities are filled by the first ``advance``."""
        k = x0.shape[0]
        lp_t = jax.eval_shape(lambda x, d: jax.vmap(lnp)(x, d)[0], x0, data)
        ll_t = jax.eval_shape(lambda x, d: jax.vmap(lnp)(x, d)[1], x0, data)
        lp, ll = jnp.zeros(lp_t.shape, lp_t.dtype), jnp.zeros(ll_t.shape, ll_t.dtype)
        rkeys = jax.vmap(lambda kk: jax.random.fold_in(kk, 0x5EED))(keys)
        pbufs = ((jnp.zeros((p_keep, k, n_walkers, n_dim), x0.dtype),
                  jnp.zeros((p_keep, k, n_walkers), lp.dtype),
                  jnp.zeros((p_keep, k, n_walkers), ll.dtype)) if p_steps else ())
        z = jnp.zeros((k, n_walkers), jnp.int32)
        return ((keys, x0, lp, ll, z, jnp.zeros((k,), jnp.int32)), pbufs, z, rkeys)

    def advance(carry, t_begin, t_end, keep_from, n_now, data):
        """Steps ``t_begin .. t_end - 1`` (absolute, pilot included): the new carry, and the
        states after steps ``keep_from + j * thin + thin - 1`` for ``j < n_now`` (n_now <= n_keep)
        in buffers of ``n_keep`` rows."""
        state, pbufs, p_acc, rkeys = carry
        keys, x, lp, ll, n_acc, n_nan = state
        lp, ll = jax.lax.cond(t_begin == 0,
                              lambda o: tuple(jax.vmap(lnp)(o[0], o[1])[:2]),
                              lambda o: (o[2], o[3]), (x, data, lp, ll))
        state = (keys, x, lp, ll, n_acc, n_nan)
        k = x.shape[0]
        bufs = (jnp.zeros((n_keep, k, n_walkers, n_dim), x.dtype),
                jnp.zeros((n_keep, k, n_walkers), lp.dtype),
                jnp.zeros((n_keep, k, n_walkers), ll.dtype))

        def body(t, c):
            state, bufs, pbufs, p_acc = c
            if p_steps:
                p_acc = jnp.where(t == p_steps, state[4], p_acc)
                state = jax.lax.cond(t == p_steps, lambda s, p: restart(s, p, rkeys),
                                     lambda s, p: s, state, pbufs)
            state = step(*state, data)
            now = (state[1], state[2], state[3])
            do, j = kept(t, keep_from, n_now, thin)
            bufs = tuple(put(b, j, do, v) for b, v in zip(bufs, now))
            if p_steps:
                do, j = kept(t, p_burn, p_keep, p_thin)
                do = do & (t < p_steps)
                pbufs = tuple(put(b, j, do, v) for b, v in zip(pbufs, now))
            return state, bufs, pbufs, p_acc

        state, bufs, pbufs, p_acc = jax.lax.fori_loop(t_begin, t_end, body,
                                                      (state, bufs, pbufs, p_acc))
        return (state, pbufs, p_acc, rkeys), bufs

    return jax.jit(advance), jax.jit(score), jax.jit(start)


def _prior_draws(prior, names, n, seed):
    """``_diagnostics.prior_draws``, vectorised per parameter: the same draws, to the bit.

    ``prior_draws`` calls each distribution's scalar ``rescale`` once per value (``init="prior"``
    redraws for every light curve of the batch). The same uniform ``u`` go through the same
    formulas here, a column at a time; an unknown family keeps the scalar path.
    """
    from ...priors._numpy import family

    rng = np.random.default_rng(seed)
    u = rng.uniform(1e-6, 1.0 - 1e-6, size=(int(n), len(names)))
    out = np.empty_like(u)
    for j, nm in enumerate(names):
        d = prior.distributions[nm]
        kind = family(d)
        if kind == "Uniform":
            out[:, j] = d.low + u[:, j] * (d.high - d.low)
        elif kind == "LogUniform":
            out[:, j] = np.exp(d._lnlow + u[:, j] * (d._lnhigh - d._lnlow))
        elif kind in ("Normal", "TruncatedNormal"):
            out[:, j] = d.ppf(u[:, j])
        elif kind == "Fixed":
            out[:, j] = d.value
        else:
            out[:, j] = [d.rescale(x) for x in u[:, j]]
    return out


def _row_scorer(one, d, width, dt):
    """``rows (m, D) -> (m,)`` log posteriors of one light curve, through the compiled ``one``
    (``(width, D)`` rows and that light curve's data as arguments): no compile per light curve."""
    import jax.numpy as jnp

    def score(rows):
        rows = np.asarray(rows, dtype=float)
        m = rows.shape[0]
        pad = (-m) % width
        full = np.concatenate([rows, np.repeat(rows[-1:], pad, axis=0)]) if pad else rows
        out = [one(jnp.asarray(full[j:j + width], dtype=dt), d)
               for j in range(0, full.shape[0], width)]
        return np.concatenate([np.asarray(o, dtype=float) for o in out])[:m]

    return score


class _Lockstep:
    """K threads, each running one light curve's start search, scored together.

    A thread's ``score(rows)`` waits until every thread still running has asked for its rows; the
    last one to ask evaluates them all at once -- ``width`` rows of every light curve per call of
    ``score_batch`` ((K, width, D) rows and the stacked data -> (K, width)) -- and wakes the rest.
    Each light curve's rows sit at the same places in the same blocks whatever the others ask, so
    its numbers do not depend on the batch. A thread that ends (or fails) stops being waited for.
    """

    def __init__(self, score_batch, data, k, width, ndim, dt):
        import threading

        self.score_batch, self.data, self.dt = score_batch, data, dt
        self.k, self.width, self.ndim = k, width, ndim
        self.cv = threading.Condition()
        self.running = k
        self.asked, self.answers = {}, {}

    def scorer(self, i):
        def score(rows):
            rows = np.asarray(rows, dtype=float).reshape(-1, self.ndim)
            with self.cv:
                self.asked[i] = rows
                if len(self.asked) == self.running:
                    self._evaluate()
                else:
                    self.cv.wait_for(lambda: i in self.answers)
                out = self.answers.pop(i)
            if isinstance(out, BaseException):
                raise out
            return out
        return score

    def leave(self):
        with self.cv:
            self.running -= 1
            if self.asked and len(self.asked) == self.running:
                self._evaluate()

    def _evaluate(self):                                   # called holding the lock
        asked, self.asked = self.asked, {}
        try:
            m = max(r.shape[0] for r in asked.values())
            x = None
            if m:
                x = np.empty((self.k, m, self.ndim))
                x[:] = next(r for r in asked.values() if r.shape[0])[0]    # idle slots: any row
                for i, r in asked.items():
                    if r.shape[0]:
                        x[i, :r.shape[0]] = r
                        x[i, r.shape[0]:] = r[-1]
            vals = (np.empty((self.k, 0)) if x is None else
                    _score_blocks(self.score_batch, x, self.data, self.width, self.dt))
            for i, r in asked.items():
                self.answers[i] = vals[i, :r.shape[0]]
        except Exception as exc:                          # noqa: BLE001 - raised in every thread
            for i in asked:
                self.answers[i] = exc
        self.cv.notify_all()


def _score_blocks(score_batch, x, data, width, dt):
    """``(K, m, D)`` rows -> ``(K, m)`` log posteriors through the compiled ``(K, width, D)``
    ``score_batch``, the last block padded with copies of each light curve's last row."""
    import jax
    import jax.numpy as jnp

    m = x.shape[1]
    pad = (-m) % width
    if pad:
        x = np.concatenate([x, np.repeat(x[:, -1:], pad, axis=1)], axis=1)
    blocks = [score_batch(jnp.asarray(x[:, b:b + width], dtype=dt), data)
              for b in range(0, x.shape[1], width)]
    return np.concatenate([np.asarray(v, dtype=float) for v in jax.device_get(blocks)],
                          axis=1)[:, :m]


def _scan_starts(score_batch, data, priors, names, seeds, n_walkers, n_scan, width, dt, *,
                 sampler="fit_batch", who=None):
    """``init="prior_scan"``: ``emcee_jax``'s default start, for every light curve at once.

    ``_diagnostics.scan_starts`` per light curve: score ``n_scan`` prior draws, climb the best 32
    (the derivative-free climb, whose only need is a scorer), keep the climbed points that reach
    the best basin, and spread the walkers two conditional sd around them. The K searches run in
    threads in lockstep (:class:`_Lockstep`), so every round of every search is ONE batched call
    of the chain's own compiled evaluation, with the data as arguments: nothing compiles per light
    curve, and a model whose evaluation is latency-bound (the TDE's sequential integrator) pays
    its latency once per round, not once per light curve and round.

    Picking the best ``n_walkers`` raw draws instead (the first version) left 18-27 of 60 walkers
    stuck at three of the four sn2026jkr LSST alerts, and best fits up to 6.9 lower in ln L than
    ``emcee_jax``'s after 5000 steps; this start lands within 0.12 of ``emcee_jax`` on all four.
    """
    import threading

    from . import _diagnostics as _dg

    k = len(priors)
    lock = _Lockstep(score_batch, data, k, width, len(names), dt)
    found = [None] * k

    def search(i):
        try:
            found[i] = _dg.scan_starts(None, priors[i], names, n_walkers, seeds[i], dt,
                                       n_scan=n_scan, score=lock.scorer(i))
        except BaseException as exc:                      # noqa: BLE001 - re-raised below
            found[i] = exc
        finally:
            lock.leave()

    threads = [threading.Thread(target=search, args=(i,), daemon=True) for i in range(k)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, sc in enumerate(found):
        if isinstance(sc, ValueError):
            name = f"light curve {i}" if who is None else who[i]
            raise ValueError(f"{sampler}: {name}: {sc}") from None
        if isinstance(sc, BaseException):
            raise sc
    starts = np.stack([sc["starts"] for sc in found])                  # (K, W, D)
    lps = _score_blocks(score_batch, starts, data, width, dt)
    details = [{"n_draws": int(len(sc["draws"])), "n_climbed": sc["n_climbed"],
                "n_reaching_best": sc["n_good"], "plateau_fraction": sc["plateau_fraction"],
                "climb_error": sc["climb_error"], "best_log_prob": sc["reference"],
                "worst_start_log_prob": float(np.min(lp))} for sc, lp in zip(found, lps)]
    return starts, lps, details


def _prior_starts(score, data, priors, names, seeds, n_walkers, dt, *, sampler="fit_batch"):
    """``init="prior"``: independent prior draws, each redrawn while its density is -inf/NaN."""
    import jax.numpy as jnp

    k = len(priors)
    rngs = [np.random.default_rng([s, 1]) for s in seeds]
    x = np.stack([_prior_draws(p, names, n_walkers, int(r.integers(2**31)))
                  for p, r in zip(priors, rngs)])
    lp = np.asarray(score(jnp.asarray(x, dtype=dt), data), dtype=float)
    for _ in range(100):
        bad = ~np.isfinite(lp)
        if not bad.any():
            break
        for i in np.flatnonzero(bad.any(axis=1)):
            rows = np.flatnonzero(bad[i])
            x[i, rows] = _prior_draws(priors[i], names, rows.size, int(rngs[i].integers(2**31)))
        lp = np.asarray(score(jnp.asarray(x, dtype=dt), data), dtype=float)
    stuck = [int(i) for i in np.flatnonzero((~np.isfinite(lp)).any(axis=1))]
    if stuck:
        raise ValueError(f"{sampler}: init='prior' found no finite start for some walkers of "
                         f"{'light curves' if sampler == 'fit_batch' else 'ensembles'} {stuck} "
                         f"after 100 redraws. Use init='prior_scan' or pass "
                         f"starts.")
    return x, lp, [{} for _ in range(k)]


def _explicit_starts(inits, one, data_list, priors, names, seeds, n_walkers, dt, *,
                     sampler="fit_batch", who=None):
    """Per-alert ``init=`` (a point, (point, scale), one start per walker, a previous result),
    through the shared ``_diagnostics.explicit_starts``; scored alert by alert on one compiled
    (W, D) program, which takes each alert's data as an argument."""
    from . import _diagnostics as _dg

    starts, lps, labels, details = [], [], [], []
    for i, (init, prior, seed, d) in enumerate(zip(inits, priors, seeds, data_list)):
        score = _row_scorer(one, d, n_walkers, dt)
        name = f"light curve {i}" if who is None else who[i]
        p0, lp, label, detail = _dg.explicit_starts(init, prior, names, n_walkers, seed, score,
                                                    f"{sampler} ({name})")
        starts.append(p0)
        lps.append(lp)
        labels.append(label)
        details.append(detail)
    return np.stack(starts), np.stack(lps), labels, details


def _run_ensembles(program, data, data_list, priors, names, seeds, inits, mode, is_log, nwalkers,
                   burnin, n_keep, thin, tail, dt, sampler, who, *, width=None, pilot=None,
                   until=None, first_keep=None):
    """Start, compile and run K ensembles of ``program`` in one compiled loop.

    The engine of :func:`fit_batch` and ``emcee_jax``. ``program(theta, d) -> (lp, ll)`` is in
    parameters; ``data`` is the K data stacked, ``data_list`` the K data apart; ``priors``,
    ``seeds`` and ``who`` (names for errors) have one entry per ensemble, ``inits`` is ``None``
    (``mode`` "prior_scan" or "prior") or one ``init=`` per ensemble. ``width`` and ``pilot`` are
    :func:`_make_loop`'s. Returns a dict with the kept chain in walker coordinates (``xs``,
    ``lps``, ``lls``: (n_keep, K, W, ...)), ``n_acc``, ``n_nan``, ``labels``, ``details``,
    ``compile_s``, ``start_s``, ``run_s``, and with a pilot ``pilot`` (its kept states, log
    likelihoods, the chosen ensemble and its accepted moves).

    ``until`` runs the chain in chunks of one compiled program (:func:`_make_loop` with
    ``segments=True``) instead of one call: ``n_keep`` is then the compiled buffer, the first chunk
    is the pilot, the burn-in and ``first_keep`` (default ``n_keep``) kept states; then
    ``until(xs, lps, lls, main_steps, call_s)`` -- the kept chunks so far (lists of host arrays),
    the main-run steps so far (burn-in included) and the last call's seconds -- returns how many
    kept states the next chunk draws (at most ``n_keep``; 0 stops). ``tail`` is not run in this
    mode. The dict then also has ``main_steps`` and ``calls`` (per call: its steps and seconds).
    """
    import jax
    import jax.numpy as jnp

    from . import _diagnostics as _dg

    k, ndim = len(priors), len(names)
    walker_program, param_score = _walker_programs(is_log)
    made = _make_loop(walker_program(program), nwalkers, ndim, burnin, n_keep, thin, tail,
                      dt, **({"width": width} if width is not None else {}),
                      **({"pilot": pilot} if pilot else {}),
                      **({"segments": True} if until is not None else {}))
    run, score = made[0], made[1]

    t_c = time.perf_counter()
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
    x_spec = jax.ShapeDtypeStruct((k, nwalkers, ndim), dt)
    if until is None:
        compiled = run.lower(keys, x_spec, data).compile()
    else:
        start = made[2]
        i32 = jax.ShapeDtypeStruct((), jnp.int32)
        carry_spec = jax.eval_shape(start, keys, x_spec, data)
        compiled = run.lower(carry_spec, i32, i32, i32, i32, data).compile()
    half = (nwalkers + 1) // 2
    if inits is not None:
        # one light curve's (W, D) rows with its data as an argument: the starts' scorer
        one = jax.jit(jax.vmap(lambda th, d: program(th, d)[0], in_axes=(0, None)))
        one = one.lower(jax.ShapeDtypeStruct((nwalkers, ndim), dt), data_list[0]).compile()
    elif mode == "prior":
        score_c = score.lower(x_spec, data).compile()
    else:
        # every light curve's rows at once, half an ensemble each: the chain's own
        # evaluation shape, so the start needs no more device memory than a half-step
        score_c = score.lower(jax.ShapeDtypeStruct((k, half, ndim), dt), data).compile()
    compile_s = time.perf_counter() - t_c

    t_i = time.perf_counter()
    if inits is not None:
        p0, lp0, labels, details = _explicit_starts(inits, one, data_list, priors, names, seeds,
                                                    nwalkers, dt, sampler=sampler, who=who)
    elif mode == "prior":
        p0, lp0, details = _prior_starts(param_score(score_c), data, priors, names, seeds,
                                         nwalkers, dt, sampler=sampler)
        labels = ["prior"] * k
    else:
        p0, lp0, details = _scan_starts(param_score(score_c), data, priors, names, seeds,
                                        nwalkers, max(1000, 4 * nwalkers), half, dt,
                                        sampler=sampler, who=who)
        labels = ["prior_scan"] * k
    bad = [who[j] for j in range(k) if not np.all(np.isfinite(lp0[j]))]
    if bad:
        raise ValueError(f"{sampler}: {', '.join(bad)}: starts at a -inf or NaN log-density; "
                         f"every walker must start where the density is finite.")
    y0 = _dg.to_walker(p0, is_log)
    for j in range(k):
        _dg.refuse_collapsed(y0[j], f"{sampler} ({who[j]})", f"init={labels[j]!r}")
    start_s = time.perf_counter() - t_i

    if until is not None:
        return _run_segments(compiled, start, keys, y0, data, dt, burnin, n_keep, thin, pilot,
                             until, first_keep,
                             {"labels": labels, "details": details, "compile_s": compile_s,
                              "start_s": start_s})
    t_r = time.perf_counter()
    out = compiled(keys, jnp.asarray(y0, dtype=dt), data)
    out = jax.device_get(jax.block_until_ready(out))
    run_s = time.perf_counter() - t_r
    g = {"xs": out[0], "lps": out[1], "lls": out[2], "n_acc": out[3], "n_nan": out[4],
         "labels": labels, "details": details, "compile_s": compile_s, "start_s": start_s,
         "run_s": run_s, "pilot": None}
    if pilot:
        g["pilot"] = {"xs": out[6], "lps": out[7], "lls": out[8], "best": int(out[9]),
                      "n_acc": out[10]}
    return g


def _run_segments(compiled, start, keys, y0, data, dt, burnin, n_keep, thin, pilot, until,
                  first_keep, g):
    """:func:`_run_ensembles`' chunked run: call ``compiled`` (the ``advance`` program) until
    ``until`` returns 0. Only the kept states of each chunk come back to the host; the carry stays
    on the device."""
    import jax
    import jax.numpy as jnp

    p_steps = int(pilot["steps"]) if pilot else 0
    t_r = time.perf_counter()
    carry = start(keys, jnp.asarray(y0, dtype=dt), data)
    t_begin, keep_from = 0, p_steps + int(burnin)
    n_now = int(n_keep if first_keep is None else first_keep)
    xs, lps, lls, calls = [], [], [], []
    while n_now > 0:
        if n_now > n_keep:
            raise RuntimeError(f"_run_segments: a chunk of {n_now} kept states exceeds the "
                               f"compiled buffer of {n_keep}.")
        t_end = keep_from + n_now * int(thin)
        t_c = time.perf_counter()
        carry, bufs = compiled(carry, np.int32(t_begin), np.int32(t_end), np.int32(keep_from),
                               np.int32(n_now), data)
        bufs = jax.device_get(jax.block_until_ready(bufs))
        call_s = time.perf_counter() - t_c
        xs.append(bufs[0][:n_now])
        lps.append(bufs[1][:n_now])
        lls.append(bufs[2][:n_now])
        calls.append({"steps": int(t_end - t_begin), "kept": int(n_now), "call_s": float(call_s)})
        main_steps = int(t_end - p_steps)
        n_next = int(until(xs, lps, lls, main_steps, call_s))
        t_begin = keep_from = t_end
        n_now = n_next
    state, pbufs, p_acc, _ = jax.device_get(carry)
    run_s = time.perf_counter() - t_r
    g = {**g, "xs": np.concatenate(xs), "lps": np.concatenate(lps), "lls": np.concatenate(lls),
         "n_acc": state[4], "n_nan": state[5], "run_s": run_s, "pilot": None,
         "main_steps": main_steps, "calls": calls}
    if pilot:
        g["pilot"] = {"xs": pbufs[0], "lps": pbufs[1], "lls": pbufs[2],
                      "best": int(np.argmax(np.max(np.asarray(pbufs[2]), axis=(0, 2)))),
                      "n_acc": p_acc}
    return g


def _record(lc, model, prior, settings, wall_s, batch):
    """Provenance of one alert's fit, in the form ``BaseSampler`` records (results.record_call)."""
    import sys

    from ...results import _call_record, _environment

    try:
        out = {"recorded_at": "fit_batch", **_environment()}
        out.update(_call_record(SAMPLER_NAME, "whisper_cbpf.samplers.jax.batch.fit_batch", lc,
                                model, {**settings, "prior": prior}, {}))
        jax = sys.modules.get("jax")
        if jax is not None:
            out["jax_devices"] = {"backend": jax.default_backend(),
                                  "devices": [str(d) for d in jax.devices()]}
        out["wall_s"] = float(wall_s)
        out["batch"] = batch
        return out
    except Exception as exc:                                  # noqa: BLE001 - never fail a fit
        return {"recorded_at": "fit_batch", "wall_s": float(wall_s),
                "error": f"{type(exc).__name__}: {exc}"}


def _result(lc, model, ld, chain, lp_chain, ll_chain, info, runtime, metrics,
            sampler=SAMPLER_NAME, all_names=None):
    """One fit's ``SamplerResult`` from its kept chain ``(n_keep, walkers, D)`` in parameters.

    ``ld`` needs ``names``, ``fixed``, ``n_data`` and ``space`` (a ``LogDensity`` has them)."""
    from ...samplers.base import (
        SamplerResult,
        _keep_model,
        aic_bic,
        attach_band_metrics,
        attach_predictive_metrics,
        summarize_posterior,
    )
    from ._diagnostics import fill_fixed

    names, fixed = list(ld.names), dict(ld.fixed)
    all_names = _sampling_order(model, ld) if all_names is None else list(all_names)
    flat = chain.reshape(-1, chain.shape[-1])               # emcee's flat order: step-major
    ll_flat = ll_chain.reshape(-1)
    samples = pd.DataFrame(fill_fixed(flat, names, fixed, all_names), columns=all_names)
    best = int(np.nanargmax(ll_flat))
    best_params = {nm: float(flat[best, j]) for j, nm in enumerate(names)}
    best_params = {nm: best_params.get(nm, fixed.get(nm)) for nm in all_names}
    max_ll = float(ll_flat[best])
    k, n = len(names), int(ld.n_data)
    aic, bic = aic_bic(max_ll, k, n)
    if metrics:
        attach_band_metrics(info, lc, model, best_params, ld.space)
    result = SamplerResult(
        sampler=sampler, model=model.name, parameters=all_names, samples=samples,
        summary=summarize_posterior(samples, all_names), best_params=best_params, n_data=n,
        n_params=k, runtime_s=float(runtime), info=info, max_log_likelihood=max_ll, aic=aic,
        bic=bic)
    if metrics:
        attach_predictive_metrics(result, lc, ld.space, model=model)
    _keep_model(result, model)
    result.samples_by_chain = fill_fixed(np.swapaxes(chain, 0, 1), names, fixed, all_names)
    return result


def _walker_programs(is_log):
    """``(walker_program, param_score)`` for walkers that move in log for the ``is_log`` columns.

    ``walker_program(program)`` turns ``program(theta, d) -> (lp, ll)`` of parameters into the
    same density of walker coordinates (lp plus the log-Jacobian); ``param_score(score_c)`` turns a
    compiled walker-coordinate scorer back into a scorer of parameters (the starts' scale).
    """
    import jax.numpy as jnp

    is_log = np.asarray(is_log, dtype=bool)
    il = jnp.asarray(is_log)

    def walker_program(program):
        if not is_log.any():
            return program

        def shared(y, d):
            lp, ll = program(jnp.where(il, jnp.exp(jnp.where(il, y, 0.0)), y), d)
            return lp + jnp.sum(jnp.where(il, y, 0.0)), ll
        return shared

    def param_score(score_c):
        if not is_log.any():
            return score_c

        def score(x, d):
            x = jnp.asarray(x)
            y = jnp.where(il, jnp.log(jnp.where(il, x, 1.0)), x)
            return score_c(y, d) - jnp.sum(jnp.where(il, y, 0.0), axis=-1)
        return score

    return walker_program, param_score


def _autocorr(chain_w, thin):
    """emcee's integrated autocorrelation time per parameter of a kept ``(n_keep, W, D)`` chain,
    in steps (scaled by ``thin``); ``[nan]`` when the chain is too short to estimate it."""
    from emcee.autocorr import integrated_time

    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.asarray(integrated_time(chain_w, c=5, tol=0), dtype=float) * thin
    except Exception:                                   # noqa: BLE001 - too short a chain
        return np.array([np.nan])


def _kept_chain_health(chain_w, lp_c, thin, nsteps, n_ensembles=1):
    """``(taus, taus_by_ensemble, health)`` of a kept chain in walker coordinates.

    ``chain_w`` is ``(n_keep, walkers, D)`` and ``lp_c`` ``(n_keep, walkers)``; with
    ``n_ensembles`` > 1 the walkers are that many ensembles of equal size, one after the other.
    The autocorrelation time is estimated per ensemble and the largest is kept per parameter.
    ``health`` is :func:`_diagnostics.walker_health`'s, plus ``frozen_walkers`` (walkers that
    never moved across the kept draws: their autocorrelation time is undefined, so the fit is
    not converged).
    """
    from . import _diagnostics as _dg

    n_keep, n_w = chain_w.shape[0], chain_w.shape[1]
    per = n_w // int(n_ensembles)
    by = [_autocorr(chain_w[:, e * per:(e + 1) * per], thin) for e in range(int(n_ensembles))]
    if len(by) == 1:
        taus = by[0]
    else:
        shaped = [t for t in by if t.size == chain_w.shape[2]]
        stack = np.stack(shaped) if shaped else np.full((1, 1), np.nan)
        finite = np.isfinite(stack)
        taus = np.where(finite.any(axis=0),
                        np.max(np.where(finite, stack, -np.inf), axis=0), np.nan)
        if len(shaped) < len(by):              # an ensemble too short to estimate: not a pass
            taus = np.full_like(taus, np.nan)
    # A walker that never moved across the kept draws makes emcee's estimate 0 / 0 (NaN, and one
    # numpy RuntimeWarning per parameter): named below instead.
    frozen = (np.flatnonzero(np.all(chain_w == chain_w[:1], axis=(0, 2))) if n_keep > 1
              else np.array([], dtype=int))
    health = _dg.walker_health(np.asarray(lp_c).T, taus, nsteps)
    health["frozen_walkers"] = [int(w) for w in frozen]
    if frozen.size:
        health["convergence_problems"].append(
            f"{frozen.size} of {n_w} walker(s) never moved across the kept draws "
            f"(walkers {', '.join(str(w) for w in frozen[:10])}"
            f"{', ...' if frozen.size > 10 else ''}): no accepted move in the "
            f"{(n_keep - 1) * thin} steps from the first kept draw to the last, so the "
            f"autocorrelation time is undefined. "
            f"Run longer, or start the walkers closer to the posterior (init=)")
        health["converged"] = False
    return taus, by, health


def _sampling_order(model, ld):
    """``model.parameters`` plus the free-scatter column, the order the samplers report in."""
    order = list(model.parameters)
    for nm in list(ld.names) + list(ld.fixed):
        if nm not in order:
            order.append(nm)
    return order


def fit_batch(lcs, model, *, sampler="emcee_jax", nwalkers=32, nsteps=5000, burnin=1000, thin=10,
              seed=0, prior=None, space="auto", likelihood="auto", init="prior_scan",
              metrics=True, walker_coordinates="own"):
    """Fit one model to many light curves at once: K emcee ensembles in one compiled GPU loop.

    Each light curve gets its own ensemble of ``nwalkers`` walkers, run with emcee's stretch move
    (a = 2, emcee's randomised red-blue split) for ``nsteps`` steps. All K ensembles advance
    together inside one compiled ``lax`` loop on the device, so every half-step is one evaluation of
    K x nwalkers/2 models, and the whole batch is one compile. Walkers never see another light
    curve's walkers, and light curve i draws from its own random stream (``seed + i``), so its
    result does not depend on what else is in the batch: it samples the same density, and
    ``fit_batch([lc_a, lc_b], m, seed=0)[1]`` equals ``fit_batch([lc_b], m, seed=1)[0]`` draw for
    draw when the padded shapes agree (the same length bucket and, for the fall-back TDEs, the
    same distinct-epoch count). Otherwise the density agrees to round-off, and the chains can part:
    the default ``init="prior_scan"`` climbs its starts on that density, which can move them by far
    more than round-off, so the chains then differ from the first step; with ``init="prior"`` they
    can part after an accept/reject decision at round-off. Either way they sample the same density.

    The density is :func:`~whisper_cbpf.samplers.jax._adapters.log_density`'s -- the prior
    plus the likelihood ``wp.fit`` would use -- with every light curve padded to the bucket of the
    longest, and, for a model that evaluates each distinct epoch once (``tde_fallback``,
    ``tde_mosfit``), its distinct epochs to the largest count in the batch: one light curve with
    many distinct epochs (one band per visit) sets the cost for all. Group alerts by epoch count
    for speed. A model that needs concrete epochs (see ``log_density``) cannot share one program:
    its light curves then run one after the other, one compile each (``info["batch"]["program"]``
    says which).

    Parameters
    ----------
    lcs : sequence of LightCurve
        The alerts. Each needs the bands the model was bound to.
    model : str or Model
        A JAX model (``predict_jax``), for example ``supernova_model("arnett", bands,
        free=["t_exp", "redshift"], ...)``, whose epochs can be traced.
    sampler : {"emcee_jax"}
        The algorithm: emcee's ensemble sampler, the only one implemented on the device.
    nwalkers, nsteps, burnin, thin : int
        As for ``emcee_jax`` (the same defaults). The kept draws are the states after steps
        ``burnin + thin - 1, burnin + 2 thin - 1, ...``, as emcee's ``get_chain``.
    seed : int or sequence of int
        Light curve i uses ``seed + i``, or ``seed[i]`` for a list.
    prior : Prior or sequence of Prior, optional
        One prior for all, or one per light curve (a different redshift prior per host, say).
        Per-light-curve priors must use the same distribution family per parameter; their
        numbers are data, so they share one program. Default: the model's.
    space, likelihood : str
        As for :func:`~whisper_cbpf.samplers.jax._adapters.log_density`.
    init : "prior_scan", "prior", array or sequence
        ``"prior_scan"`` (default): ``emcee_jax``'s start, per light curve -- score
        ``max(1000, 4 nwalkers)`` prior draws, climb the best 32, and spread the walkers two
        conditional sd around the climbed points that reach the best basin. The climb is the
        derivative-free one; every light curve's climb round is one batched call of the chain's
        compiled evaluation (``emcee_jax``'s gradient climb would compile once per light curve).
        ``"prior"``: independent prior draws, each redrawn while its density is -inf. A
        ``(K, nwalkers, ndim)`` array: the starts. A list of K per-light-curve starts, each any
        form ``emcee_jax`` takes (a point, ``(point, scale)``, one start per walker, a previous
        ``SamplerResult`` such as an ABC fit).
    metrics : bool
        Attach the per-band and posterior-predictive metrics to each result, as every sampler
        does (CPU work, per light curve). ``False`` skips them for speed.
    walker_coordinates : {"own", "linear"}
        Where the walkers move, as for ``emcee_jax``: ``"own"`` (default) in each prior's own
        coordinate (the natural log of a LogUniform parameter, with the log-Jacobian added, so the
        posterior is the same), ``"linear"`` in the parameters themselves (whisper 0.1.x).

    Returns
    -------
    list of SamplerResult
        One per light curve, in order, with ``sampler="emcee_batch"``: draws, summary, best fit,
        AIC/BIC from the maximum log-likelihood of the kept draws, ``samples_by_chain``
        (walkers, draws, parameters), and ``info`` with the ``emcee_jax`` diagnostics
        (``converged``, ``convergence_problems``, ``stuck_walkers``, ``max_autocorr_time``,
        ``mean_acceptance_fraction``) plus ``info["batch"]`` (the batch's size, this light
        curve's index, bucket, ``epoch_bucket`` (the distinct epochs' padded count, ``None`` for
        a model that evaluates every observation) and this light curve's own
        ``n_distinct_epochs``, and the compile, start and run seconds of the whole batch).
        ``runtime_s`` and ``info["compile_time_s"]`` are this light curve's share: the batch's
        time divided by the number of light curves it held. ``result.diagnostics()`` works as
        for ``emcee_jax``.

    Raises
    ------
    ValueError
        An unknown sampler, ``burnin >= nsteps``, a thinning that keeps no draw, fewer than
        ``2 * ndim`` walkers, mismatched ``seed`` / ``prior`` / ``init`` lengths, or a light curve
        whose density is -inf at nearly every prior draw.

    Notes
    -----
    The autocorrelation time is estimated on the kept (thinned, post-burn-in) chain and scaled by
    ``thin``, so it cannot resolve times shorter than ``thin`` steps; it errs long, never short. A
    walker that never moves across the kept draws leaves it undefined; such walkers are listed in
    ``info["frozen_walkers"]`` and the fit is marked not converged. A proposal whose density is
    NaN is rejected like -inf (emcee would raise) and counted in ``info["n_nan_proposals"]``. One
    warning names the light curves whose fits are not converged, and one
    :class:`~whisper_cbpf.samplers.ZeroFluxWarning` those whose first detection is consistent
    with zero flux.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> flare = wp.get_model("flare_jax")
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> lcs = []
    >>> for t0 in (8.0, 12.0, 16.0):
    ...     truth = {"log_amp": 1.0, "log_sigma": 0.5, "log_tau": 1.5, "t0": t0}
    ...     lcs.append(wp.LightCurve(time=t, band=["r"] * 30, flux=flare.predict(truth, t),
    ...                              flux_err=np.full(30, 0.1)))
    >>> fits = wp.fit_batch(lcs, "flare_jax", nsteps=3000, burnin=1000, metrics=False)
    >>> [round(f.summary["t0"]["median"]) for f in fits]
    [8, 12, 16]
    >>> fits[0].info["batch"]["n_alerts"], fits[0].sampler
    (3, 'emcee_batch')
    """
    from ...io.schema import LightCurve
    from ...models import get_model
    from ..base import (ZERO_FLUX_WARNED, ZeroFluxWarning, _pre_event_plan,
                        _warn_user, _zero_flux_batch_message, _zero_flux_given, check_burnin)
    from ._adapters import bucket_size, float_dtype, log_density

    if str(sampler) != "emcee_jax":
        raise ValueError(f"fit_batch: sampler={sampler!r} is not implemented on the device. Only "
                         f"'emcee_jax' (emcee's ensemble sampler) is; for another sampler loop "
                         f"over wp.fit, or use wp.run_jobs.")
    if isinstance(lcs, LightCurve):
        raise TypeError("fit_batch takes a LIST of light curves; wrap one as [lc], or use "
                        "wp.fit(lc, model, sampler='emcee_jax').")
    lcs = list(lcs)
    if not lcs:
        raise ValueError("fit_batch: no light curves were given.")
    check_burnin(burnin, nsteps)
    nwalkers, nsteps, burnin, thin = int(nwalkers), int(nsteps), int(burnin), int(thin)
    if thin < 1:
        raise ValueError(f"fit_batch: thin must be >= 1; got {thin}.")
    n_keep = (nsteps - burnin) // thin
    if n_keep < 1:
        raise ValueError(f"fit_batch: thin={thin} keeps no draw of the {nsteps - burnin} steps "
                         f"after burnin={burnin}. Lower thin or raise nsteps.")
    tail = (nsteps - burnin) - n_keep * thin

    import jax
    import jax.numpy as jnp

    t_wall = time.perf_counter()
    model = get_model(model)
    k_all = len(lcs)
    seeds = _alert_seeds(seed, k_all)
    priors = _alert_priors(prior, k_all)
    if isinstance(init, str) or init is None:
        inits = None
        mode = "prior_scan" if init is None else str(init)
        if mode not in ("prior_scan", "prior"):
            raise ValueError(f"fit_batch: unknown init {init!r}. Use 'prior_scan' (default), "
                             f"'prior', a (K, nwalkers, ndim) array, or one start per light curve.")
    else:
        if isinstance(init, (list, tuple)):
            inits = list(init)
        else:
            try:
                arr = np.asarray(init, dtype=float)
            except (TypeError, ValueError):
                raise TypeError(
                    f"fit_batch: init must be 'prior_scan', 'prior', a (K, nwalkers, ndim) array, "
                    f"or a list with one start per light curve (a point, (point, scale), a "
                    f"(nwalkers, ndim) array or a previous result); got {type(init).__name__}."
                ) from None
            if arr.ndim != 3 or arr.shape[:2] != (k_all, nwalkers):
                raise ValueError(f"fit_batch: an init array must be (K, nwalkers, ndim) = "
                                 f"({k_all}, {nwalkers}, ndim); got {arr.shape}. Pass a list of "
                                 f"per-light-curve starts for any other form.")
            inits = list(arr)
        if len(inits) != k_all:
            raise ValueError(f"fit_batch: init has {len(inits)} entries for {k_all} light curves.")
        mode = "explicit"

    t_setup = time.perf_counter()
    # The pre-event rule, as in every fit: each alert's rows at or before its event are left out
    # and, for a free t_exp, its data set the explosion-time prior (one warning for the batch).
    plans = [_pre_event_plan(lc, model, p, likelihood=likelihood) for lc, p in zip(lcs, priors)]
    cut = [i for i, pl in enumerate(plans) if pl.message]
    if cut:
        _warn_user(
            f"fit_batch: pre-event data are not fitted: {len(cut)} of {k_all} light curves "
            f"(indices {cut[:20]}{', ...' if len(cut) > 20 else ''}) had rows left out or an "
            f"explosion-time prior set from their data. Each result's info['pre_event'] and "
            f"info['t_exp_prior'] say what.")
    fit_lcs = [pl.lc for pl in plans]
    priors = [pl.prior if pl.prior is not None else p for pl, p in zip(plans, priors)]
    width = bucket_size(max(int(len(lc.time)) for lc in fit_lcs))
    epoch_of = getattr(model.predict_jax, "epoch_index", None)
    # Distinct epochs padded EXACTLY to the batch's largest count: the K data must stack, and the
    # loop is compiled once per call anyway, so a bucket would only add rows.
    n_distinct = (None if epoch_of is None else
                  [int(epoch_of(np.asarray(lc.time, dtype=float))[0].size) for lc in fit_lcs])
    ewidth = "auto" if epoch_of is None else max(n_distinct)
    # One zero-flux warning for the batch, not one per light curve (log_density would give each).
    zf_msg = _zero_flux_batch_message([i for i, pl in enumerate(plans) if pl.zero_flux], k_all)
    if zf_msg and not ZERO_FLUX_WARNED.get():
        _warn_user(zf_msg, ZeroFluxWarning)
    with _zero_flux_given():
        lds = [log_density(lc, model, space=space, likelihood=likelihood, prior=p, bucket=width,
                           epoch_bucket=ewidth)
               for lc, p in zip(fit_lcs, priors)]
    names = list(lds[0].names)
    ndim = len(names)
    if nwalkers < 2 * ndim:
        raise ValueError(f"fit_batch: nwalkers={nwalkers} is fewer than twice the {ndim} sampled "
                         f"parameters; the stretch move needs at least {2 * ndim}.")
    dt = float_dtype()
    # Group the light curves that share a compiled program (all of them when the data are
    # arguments); a model that closes over its epochs gives one group per light curve.
    groups = {}
    for i, ld in enumerate(lds):
        if list(ld.names) != names:
            raise ValueError(f"fit_batch: light curve {i} samples {ld.names}, light curve 0 "
                             f"samples {names}. Every light curve must fit the same parameters "
                             f"(the same Fixed parameters and scatter column).")
        groups.setdefault(id(ld.shared), []).append(i)
    setup_s = time.perf_counter() - t_setup

    from . import _diagnostics as _dg

    free_priors = [_dg.split_fixed(ld.prior, _sampling_order(model, ld), "fit_batch")[0]
                   for ld in lds]
    # The walkers move in each prior's own coordinate (log for a LogUniform); every light curve's
    # prior has the same family per parameter, so one set of log columns serves the batch.
    is_log = _dg.walker_log_columns(free_priors[0], names, walker_coordinates, "fit_batch")
    walker_program, param_score = _walker_programs(is_log)

    results = [None] * k_all
    settings = {"nwalkers": nwalkers, "nsteps": nsteps, "burnin": burnin, "thin": thin,
                "space": space, "likelihood": likelihood,
                "init": mode if inits is None else "explicit", "metrics": bool(metrics),
                "walker_coordinates": walker_coordinates}
    n_unconverged, few_data = [], []
    for members in groups.values():
        k = len(members)
        program = lds[members[0]].shared
        data = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[lds[i].data for i in members])
        g = _run_ensembles(
            program, data, [lds[i].data for i in members], [free_priors[i] for i in members],
            names, [seeds[i] for i in members],
            None if inits is None else [inits[i] for i in members], mode, is_log, nwalkers,
            burnin, n_keep, thin, tail, dt, "fit_batch", [f"light curve {i}" for i in members])
        xs, lps, lls, n_acc, n_nan = g["xs"], g["lps"], g["lls"], g["n_acc"], g["n_nan"]
        labels, details = g["labels"], g["details"]
        compile_s, start_s, run_s = g["compile_s"], g["start_s"], g["run_s"]

        first = lds[members[0]]
        batch_info = {"n_alerts": k, "n_alerts_in_call": k_all, "bucket": int(first.bucket),
                      "epoch_bucket": first.epoch_bucket,
                      "program": ("data as arguments" if first.data_as_argument else
                                  "epochs closed over: " + first.reason),
                      "compile_s": float(compile_s), "start_s": float(start_s),
                      "run_s": float(run_s), "run_s_per_alert": float(run_s / k),
                      "setup_s": float(setup_s),
                      "evaluations_per_half_step": int(k * ((nwalkers + 1) // 2))}
        for j, i in enumerate(members):
            t_p = time.perf_counter()
            chain_w = np.asarray(xs[:, j], dtype=float)       # (n_keep, W, D), walker coordinates
            chain = _dg.from_walker(chain_w, is_log)
            lp_c = np.asarray(lps[:, j], dtype=float)
            ll_c = np.asarray(lls[:, j], dtype=float)
            taus, _, health = _kept_chain_health(chain_w, lp_c, thin, nsteps)
            ld = lds[i]
            info = {
                "nwalkers": nwalkers, "nsteps": nsteps, "burnin": burnin, "thin": thin,
                "space": ld.space, "seed": int(seeds[i]),
                "init": labels[j], "init_detail": details[j],
                "init_time_s": float(start_s / k),
                "walker_coordinates": {"coordinates": walker_coordinates,
                                       "log": [nm for nm, lg in zip(names, is_log) if lg]},
                "compile_time_s": float(compile_s / k),
                "likelihood": ld.likelihood,
                "scatter_param": next((nm for nm in ld.names if nm not in model.parameters), None),
                "x64": bool(jax.config.jax_enable_x64), "fixed": dict(ld.fixed),
                "max_log_likelihood_is": "likelihood",
                "mean_acceptance_fraction": float(np.mean(n_acc[j]) / nsteps),
                "acceptance_fraction": [float(v) for v in np.asarray(n_acc[j]) / nsteps],
                "mean_autocorr_time": (float(np.nanmean(taus)) if np.isfinite(taus).any()
                                       else float("nan")),
                "autocorr_time": [float(v) for v in taus],
                "autocorr_from": f"kept chain (post-burn-in, thinned by {thin}) x {thin}",
                "n_nan_proposals": int(n_nan[j]),
                **health,
                "backend": "fit_batch: emcee stretch move on the device (one compiled lax loop), "
                           f"{k} light curves per call",
                "batch": {**batch_info, "alert_index": int(i),
                          "n_distinct_epochs": ld.n_distinct_epochs},
            }
            runtime = (start_s + run_s) / k
            res = _result(fit_lcs[i], model, ld, chain, lp_c, ll_c, info, runtime, metrics)
            plans[i].stamp(res)
            res.info["postprocess_s"] = float(time.perf_counter() - t_p)
            res.provenance = _record(lcs[i], model, priors[i],
                                     {**settings, "seed": int(seeds[i])},
                                     (time.perf_counter() - t_wall) / k_all,
                                     {"n_alerts": k_all, "alert_index": int(i)})
            if not health["converged"]:
                n_unconverged.append(i)
            if ld.n_data <= ndim:
                few_data.append(i)
            results[i] = res

    if n_unconverged:
        shown = ", ".join(str(i) for i in n_unconverged[:20])
        _warn_user(
            f"fit_batch: {len(n_unconverged)} of {k_all} fits are not converged (light curves "
            f"{shown}{', ...' if len(n_unconverged) > 20 else ''}). Each result's "
            f"info['convergence_problems'] says why; result.diagnostics() gives the full report.",
           )
    if few_data:
        _warn_user(
            f"fit_batch: {len(few_data)} light curves have no more points than the {ndim} free "
            f"parameters (light curves {few_data[:20]}): not enough data, so their posteriors "
            f"mostly restate the prior.")
    return results
