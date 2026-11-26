# Rescuing SVGD from divergence

## What goes wrong

Two suite tests (`test_model_selection::test_log_likelihood_independent_of_regularization`
and `test_svgd_exposure::test_exposure_shifts_posterior_inverse_to_alpha`) fail
mid-`optimize()` — **not** at an assertion, but with a hard crash:

```
jaxlib._jax.XlaRuntimeError: INTERNAL: CpuCallback error calling callback:
RuntimeError: Expected vertex ... outgoing rate divided by granularity <= 1.
Rate is '9.88e31' ... Increase the granularity
```

A single SVGD particle wanders (during the gradient-ascent updates) to a θ whose
implied transition rate reaches ~1e31. The PMF/PDF callback
(`_compute_pmf_and_moments_cached` → `builder.compute_pmf_and_moments` →
`ptd_probability_distribution_context_create`, `src/c/phasic.c:11475`) cannot
build a phase-type distribution for such a rate (see the companion analysis:
uniformization needs `granularity ≥ rate`, and the forward-step count
`granularity·time` is hard-capped at 1e9 in `api/cpp/phasiccpp.h:1313`). It
raises, the exception crosses the `jax.pure_callback` boundary
(`src/phasic/__init__.py:8685`), and **the whole optimization dies** — one bad
particle out of 60 takes down the entire run.

Increasing MPFR "multiple precision" cannot help: precision widens arithmetic
bit-width, but the failure is a *discretization-resolution* limit, not
round-off. The real fix is to stop particles reaching rate 1e31, and to make a
stray one non-fatal.

The three failure ingredients, each independently addressable:

1. **A particle diverges** — the optimizer lets θ explode.
2. **The model is unbounded** — rate = θ (or `softplus(θ)`) has no ceiling, so a
   large θ maps to an astronomically large rate.
3. **The compute path throws** — a bad θ crashes instead of returning a finite
   "very unlikely" value the optimizer can step away from.

Fixing **any one** of these makes the run survive; fixing all three makes SVGD
robust. Ordered by leverage:

---

## A. Make the compute path fail soft (highest leverage, smallest blast radius)

The single most valuable change: when a θ produces an uncomputable
distribution, return a **finite penalty** (PMF ≈ 0 → log-likelihood ≈ a large
negative constant) instead of raising across the callback boundary. The
optimizer then sees a terrible-but-finite loss at that θ and its gradient pushes
the particle back toward feasible regions — exactly the desired behavior.

Where: the callback closures in `src/phasic/__init__.py`
(`_compute_pmf_and_moments_cached` ~8603, `callback_fn` ~8658). Wrap the native
call:

```python
def _compute_pmf_and_moments_cached(theta_np, times_np, rewards_np=None):
    try:
        return _native_compute(theta_np, times_np, rewards_np)   # existing body
    except RuntimeError as e:
        # Uncomputable θ (rate too large / diverged particle). Return a finite
        # "impossible observation" so the optimizer steps away instead of the
        # whole vmap/pmap batch crashing. PMF≈0 → logpdf≈large-negative.
        if _is_rate_blowup(e):        # match the granularity/overflow messages
            pmf = np.zeros_like(times_np)
            moments = np.zeros((nr_moments,), dtype=np.float64)
            return pmf, moments
        raise
```

Notes/caveats:
- Return a **small positive floor** (e.g. `1e-300`) rather than exactly 0 if the
  loss takes `log(pmf)` without an epsilon, to avoid `-inf` → `nan` gradients.
  (`test_svgd_exposure`'s loss already adds `1e-12`; `model_selection`'s
  likelihood path should be checked.)
- Keep it **narrow**: only swallow the rate-blowup/overflow errors (now clearly
  labeled after the int64-overflow fix), not every `RuntimeError`, so genuine
  bugs still surface.
- This is behavior-preserving for all non-diverged runs (the `try` only
  triggers on θ that previously crashed).
- Do the same in the gradient callback path if one exists, so the FD/analytic
  gradient at a blown-up θ is finite (e.g. zero) rather than propagating `nan`.

---

## B. Keep particles from diverging (SVGD-level numerical safeguards)

In `src/phasic/svgd.py` (`SVGD.optimize()` and the per-step update), add standard
robustness that trades a little bias for a lot of stability:

1. **Per-step update clipping / trust region.** Cap the L2 norm (or per-coordinate
   magnitude) of each particle's φ-update so no single step can jump orders of
   magnitude:
   ```python
   step = lr * phi_grad
   max_step = trust_radius              # e.g. 1.0 in φ-space
   norm = jnp.linalg.norm(step, axis=-1, keepdims=True)
   step = step * jnp.minimum(1.0, max_step / (norm + 1e-12))
   phi = phi + step
   ```
   This is the most direct cure for "a particle escaped in one bad step."

2. **Global gradient clipping.** Clip the log-posterior gradient norm before it
   enters the SVGD kernel combination (`optax.clip_by_global_norm` composes
   cleanly with the existing optax integration).

3. **Non-finite guard + particle rejuvenation.** After each step, detect
   particles with non-finite φ or loss and either (a) reset them to a resample
   from the current finite particle cloud, or (b) freeze them at their last
   finite value. Prevents one `nan` from contaminating the kernel (the SVGD
   kernel couples all particles, so one `nan` can poison the whole cloud on the
   next iteration).

4. **Clamp φ to a sane box.** A soft clip `phi = jnp.clip(phi, -CAP, CAP)` in
   unconstrained space bounds the implied rate. With the softplus/exp transforms
   already used, a φ-cap of e.g. ±30 keeps rates well below the 1e9-ish
   feasibility ceiling while leaving the physical parameter range untouched.

5. **Step-size schedule sanity.** The failing tests use
   `ExpStepSize(first_step=0.01, last_step=0.001, tau=500)`. `first_step` is
   modest, but with a weak prior and a wide kernel bandwidth the *effective*
   step can still be large. Consider a warmup (smaller first steps) and/or
   coupling the step size to the gradient norm.

---

## C. Bound the model so a large θ can't imply an impossible rate

- **Bounded transforms.** Where a parameter is a rate, map it through a
  transform with a finite ceiling for large φ (e.g. a scaled sigmoid to
  `[rate_min, rate_max]`) instead of `θ` (unbounded) or `softplus(θ)`
  (unbounded above). phasic already supports per-dimension transforms and
  priors (`BetaPrior`, `LogGaussPrior`, `svgd_config.py`); a bounded transform
  makes rate 1e31 unreachable by construction.
- **Informative priors / stronger regularization.** The tests use a very weak
  prior (`-0.5·(φ/10)²`). A tighter Gaussian (or `LogGaussPrior(ci=[...])`)
  supplies restoring force toward plausible θ. Note `regularization=` in
  `model_selection` regularizes the moment-matching term, not θ magnitude — it
  does not by itself prevent divergence.
- **Rescale units.** The C error suggests it directly: if rates and observation
  times live on very different scales, rescale so the working rate is O(1).

---

## D. Diagnostics (so divergence is visible, not silent-until-crash)

- Track and log per-iteration `max |φ|`, `max implied rate`, and any
  non-finite loss/gradient counts; emit a warning when a particle crosses a
  threshold (well before the 1e9 feasibility ceiling).
- Optionally expose a `on_divergence=('penalize'|'clip'|'raise')` policy on
  `SVGD`/`Graph.svgd` so users choose between the fail-soft (A), clip (B), and
  strict behaviors.

---

## Recommended minimum

For the two failing tests specifically, **A + B.1** is the smallest change that
makes them robust without altering converged results: fail soft in the callback
so a stray particle yields a finite penalty, and clip the per-step update so
particles can't leap to rate 1e31 in one move. **C** (bounded transform / tighter
prior) is the principled longer-term fix; **D** turns the next divergence into a
warning instead of a crash.

Note: the int64-overflow mislabel in `ptd_probability_distribution_context_create`
has been fixed separately — the crash now reports "maximum outgoing rate … too
large … model diverged or rescale units" instead of the misleading "Increase the
granularity", which makes the `_is_rate_blowup(e)` match in (A) straightforward.
