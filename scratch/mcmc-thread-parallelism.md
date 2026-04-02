# MCMC Thread Parallelism for Likelihood Correction

## Summary

When MCMC uses a `likelihood_correction` (e.g., BFFG importance weights), the correction cannot be JIT-compiled because it involves stochastic C path sampling. To still benefit from multi-core execution, the correction is evaluated in parallel across chains using Python's `ThreadPoolExecutor`. The C path sampling releases the GIL, giving real parallelism (~2.7x speedup with 4 threads measured).

## Code Changes

### `src/phasic/mcmc.py`

#### 1. JIT auto-detection (line ~765)

JIT is only enabled when `_log_prob` is fully traceable — model+data mode with no `log_prob_fn` and no `likelihood_correction`:

```python
can_jit = (self.jit_enabled
           and self._log_prob_fn is None
           and self.likelihood_correction is None)
```

#### 2. Parallel auto-detection (line ~410)

`vmap` parallelism is only used when the log_prob is fully traceable:

```python
can_vmap = (log_prob_fn is None and likelihood_correction is None
            and model is not None)
if parallel is None:
    if can_vmap and n_chains > 1:
        self.parallel = 'vmap'
    else:
        self.parallel = None
```

#### 3. Correction outside JIT in `_run_chain` (line ~500, ~555)

The `likelihood_correction` is applied after the JIT'd `log_prob_fn` call, not inside `_log_prob`:

```python
# Initial evaluation
lp_current = float(log_prob_fn(phi_current))
if self.likelihood_correction is not None:
    theta_current = self.param_transform(phi_current) if self.param_transform else phi_current
    lp_current += float(self.likelihood_correction(theta_current))

# Per-iteration evaluation
lp_proposed = float(log_prob_fn(phi_proposed))
if self.likelihood_correction is not None:
    theta_proposed = self.param_transform(phi_proposed) if self.param_transform else phi_proposed
    lp_proposed += float(self.likelihood_correction(theta_proposed))
```

#### 4. Threaded corrections in `_run_chains_vmap` (line ~690)

When vmap evaluates all chains' model log-probs in parallel, the corrections are added via `ThreadPoolExecutor`:

```python
if self.likelihood_correction is not None:
    from concurrent.futures import ThreadPoolExecutor
    def _eval_correction(c):
        theta_c = self.param_transform(phi_proposed[c]) if self.param_transform else phi_proposed[c]
        return float(self.likelihood_correction(theta_c))
    with ThreadPoolExecutor(max_workers=n_chains) as pool:
        corrections = list(pool.map(_eval_correction, range(n_chains)))
    lp_proposed = lp_proposed + jnp.array(corrections)
```

Same pattern for initial `lp_all` computation.

## What This Enables

- **model + observed_data** (no correction): JIT + vmap, full parallelism via JAX
- **log_prob_fn** mode: no JIT, sequential chains
- **model + likelihood_correction**: JIT for model (vmap across chains), threaded correction

## When to Remove

This thread-based parallelism is a workaround for the correction not being JIT-compatible. Once the full JAX FFI plan (`docs/bffg-jit-plan.md`) is implemented, the correction will be JIT+vmap compatible and the threading code becomes unnecessary.
