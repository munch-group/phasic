# BFFG JAX Vision

## End Goal

Full JAX support for BFFG inference using the standard phasic UI:

```python
model, correction = bffg_log_prob(...)

mcmc = MCMC(
    model=model,
    observed_data=observed_data,
    likelihood_correction=correction,
    ...
)
```

Where the entire `_log_prob` (model + correction + prior) is:

- **`jax.jit`** compilable — single compilation, fast repeated evaluation
- **`jax.vmap`** compatible — parallel chains on a single device
- **`jax.pmap`** compatible — one chain per device (multi-GPU, multi-CPU)
- **Sharding** compatible — distribute across devices via JAX mesh
- **Multi-node** compatible — distribute across machines on SLURM via `jax.distributed`

## What This Means

The `model + observed_data + likelihood_correction` interface is the standard phasic pattern for all inference (SVGD, MCMC, future methods). The BFFG correction is just an optional additive term. All three components must be JAX-traceable:

1. **model(theta, data)**: Already JIT-compatible via `compute_sojourn_times_ffi` / `pmf_from_graph_joint_index`
2. **likelihood_correction(theta)**: Requires FFI for conditioned path sampling + pure JAX weight computation (see `docs/bffg-jit-plan.md`)
3. **prior(theta)**: Already JAX-compatible (GaussPrior etc.)

## Implementation Path

1. **Current state**: Model is JIT+vmap. Correction is non-JIT (C path sampling + Python loops). Thread-based workaround for multi-core (`docs/mcmc-thread-parallelism.md`).

2. **Next**: Implement `docs/bffg-jit-plan.md` — FFI for conditioned path sampling, pure JAX weight computation. Remove thread workaround.

3. **Then**: pmap support in MCMC (one chain per device). Follows SVGD's existing pmap pattern (`src/phasic/svgd.py` lines 3352-3375).

4. **Finally**: Multi-node via `jax.distributed.initialize()` + SLURM integration. Each node runs a subset of chains.

## Design Constraints

- The `model + observed_data + likelihood_correction` signature must not change
- SVGD and MCMC must interpret `observed_data` identically (see `memory/project_mcmc_svgd_consistency.md`)
- `theta_target_fn` provided by the user must be JAX-traceable (use `jnp.where` not Python `if`)
- Graph serialization happens once; all per-iteration computation goes through FFI with cached traces
