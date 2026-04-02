# Plan: Full JAX JIT Compatibility for BFFG Correction

## Problem

The BFFG `likelihood_correction` function in `src/phasic/bffg.py` cannot be JIT-compiled because:

1. **`sample_path_conditioned()`** returns variable-length arrays (JAX JIT requires static shapes)
2. **`sample_path_conditioned()`** uses C's `rand()` (not JAX's deterministic PRNG)
3. **Weight computation** uses Python loops over path steps with data-dependent control flow
4. **`theta_target_fn`** closures capture numpy arrays

Currently the correction runs outside JIT. The model evaluation (`compute_sojourn_times_ffi`) is already JIT-compatible. Making the correction JIT-compatible would enable:
- `jax.vmap` parallelism across chains (true multi-core)
- Potential GPU acceleration
- Cleaner integration with JAX ecosystem

## Approach: FFI with Fixed-Size Path Encoding

### Core Idea

Sample paths into fixed-size buffers (padded with sentinels), exposed via JAX FFI. Compute weights in pure JAX using masked operations. All operations become static-shape, JIT-traceable.

### Step 1: Fixed-Size C Sampling Function

**File: `api/c/phasic.h`**

```c
/**
 * Sample a conditioned path into fixed-size pre-allocated buffers.
 *
 * @param graph          The graph to sample from
 * @param backward_probs Pre-computed backward probabilities (from ptd_backward_probabilities)
 * @param max_length     Size of output buffers
 * @param seed           Random seed (deterministic given seed)
 * @param vertex_indices Output: visited vertex indices, -1 padded
 * @param entry_times    Output: cumulative entry times, 0.0 padded
 * @return               Actual path length (number of valid entries)
 */
size_t ptd_random_sample_path_conditioned_fixed(
    struct ptd_graph *graph,
    double *backward_probs,
    size_t max_length,
    unsigned int seed,
    int *vertex_indices,
    double *entry_times
);
```

**File: `src/c/phasic.c`**

Implementation mirrors `ptd_random_sample_path_conditioned` but:
- Uses `srandom(seed)` then `random()` instead of `rand()` for deterministic output
- Writes into caller-provided buffers instead of malloc
- Pads remaining slots with -1 (indices) and 0.0 (times)
- Returns actual path length

Max path length can be conservatively set to `2 * graph->vertices_length` — no path through an acyclic graph (excluding trash loops) visits more vertices than exist.

### Step 2: C++ FFI Handler

**File: `src/cpp/parameterized/ffi_handlers.cpp`**

New handler `sample_path_conditioned_handler`:

```
Inputs (buffers):
  - theta: float64[n_params]           — graph parameters
  - target_vertex: int32[1]            — which terminal to condition on
  - seed: uint32[1]                    — JAX PRNG seed

Outputs (buffers):
  - vertex_indices: int32[max_length]  — path vertex indices, -1 padded
  - entry_times: float64[max_length]   — cumulative times, 0.0 padded

Attributes (static):
  - structure_json: string             — serialized graph (cached in GraphBuilder)
  - max_length: int32                  — buffer size
```

Implementation:
1. Get/create cached `GraphBuilder` from `structure_json` (thread-local, same as existing handlers)
2. Call `graph.update_weight_parameterized(theta, n_params)` on the builder's graph
3. Compute backward probabilities for `target_vertex`
4. Call `ptd_random_sample_path_conditioned_fixed()` with seed and output buffers

Also add `backward_probabilities_handler`:
```
Inputs: target_vertices: int32[n_targets]
Outputs: backward_probs: float64[n_vertices]
Attributes: structure_json
```

This is simpler — deterministic, fixed output size.

### Step 3: Python FFI Wrappers

**File: `src/phasic/ffi_wrappers.py`**

```python
def sample_path_conditioned_ffi(
    structure_json: str | dict,
    theta: jax.Array,
    target_vertex: jax.Array,
    seed: jax.Array,
    max_length: int,
) -> tuple[jax.Array, jax.Array]:
    """Sample a conditioned path via FFI.

    Returns (vertex_indices, entry_times) with fixed size max_length.
    Unused entries are -1 (indices) and 0.0 (times).
    """
    _register_ffi_targets()
    structure_str = _ensure_json_string(structure_json)

    result_shapes = (
        jax.ShapeDtypeStruct((max_length,), jnp.int32),
        jax.ShapeDtypeStruct((max_length,), jnp.float64),
    )

    ffi_fn = jax.ffi.ffi_call(
        "ptd_sample_path_conditioned",
        result_shapes,
        vmap_method="expand_dims"
    )

    return ffi_fn(
        theta,
        jnp.atleast_1d(target_vertex).astype(jnp.int32),
        jnp.atleast_1d(seed).astype(jnp.uint32),
        structure_json=structure_str,
        max_length=np.int32(max_length),
    )


def backward_probabilities_ffi(
    structure_json: str | dict,
    theta: jax.Array,
    target_vertices: jax.Array,
) -> jax.Array:
    """Compute backward probabilities via FFI.

    Returns float64[n_vertices] with P(reach target | start at v) for each v.
    """
    ...
```

### Step 4: Pure JAX Weight Computation

**File: `src/phasic/bffg.py`**

Replace `_full_importance_log_weight` with a JAX-traceable version:

```python
def _importance_log_weight_jax(
    vertex_indices,          # (max_length,) int32, -1 padded
    entry_times,             # (max_length,) float64, 0.0 padded
    all_edge_coeffs,         # (n_verts, max_edges, n_params) — precomputed dense
    all_edge_targets,        # (n_verts, max_edges) int32 — precomputed
    all_n_edges,             # (n_verts,) int32 — number of valid edges per vertex
    theta_proposal,          # (n_params,)
    theta_target_at_times,   # (max_length, n_params) — theta_target_fn evaluated at each time
):
    """Pure JAX importance weight with masking."""
    max_len = vertex_indices.shape[0]
    sojourns = jnp.diff(entry_times)
    # Padding: sojourns after the path ends are 0, won't contribute

    log_w = 0.0
    def body_fn(carry, step):
        log_w = carry
        vi = vertex_indices[step + 1]  # +1 because step 0 is starting vertex
        valid = vi >= 0

        # Look up precomputed edge coefficients for this vertex
        coeffs = all_edge_coeffs[vi]       # (max_edges, n_params)
        n_edges = all_n_edges[vi]

        # Compute rates
        theta_t = theta_target_at_times[step + 1]
        edge_rates_prop = coeffs @ theta_proposal   # (max_edges,)
        edge_rates_tgt = coeffs @ theta_t           # (max_edges,)
        r_prop = jnp.sum(edge_rates_prop)
        r_tgt = jnp.sum(edge_rates_tgt)

        # Find taken edge: next vertex in path
        next_vi = vertex_indices[step + 2] if step + 2 < max_len else -1
        edge_match = (all_edge_targets[vi] == next_vi)
        # Use first matching edge (argmax on boolean gives first True)
        taken_edge = jnp.argmax(edge_match)

        # Exit rate ratio
        s_k = sojourns[step + 1]
        dw_rate = jnp.log(r_tgt) - jnp.log(r_prop) - (r_tgt - r_prop) * s_k

        # Transition probability ratio
        p_prop = edge_rates_prop[taken_edge] / r_prop
        p_tgt = edge_rates_tgt[taken_edge] / r_tgt
        dw_trans = jnp.log(p_tgt) - jnp.log(p_prop)

        # Only add if this step is valid
        dw = jnp.where(valid & (r_prop > 0) & (r_tgt > 0), dw_rate + dw_trans, 0.0)
        return log_w + dw, None

    log_w, _ = jax.lax.scan(body_fn, 0.0, jnp.arange(max_len - 2))
    return log_w
```

Key JAX patterns:
- `jax.lax.scan` instead of Python for-loop (JIT-compatible)
- `jnp.where` for conditional masking instead of Python if/break
- Dense precomputed arrays instead of dynamic list lookups
- `jnp.argmax` on boolean mask for edge matching

### Step 5: Dense Edge Coefficient Precomputation

At `bffg_log_prob` construction time, convert the sparse per-vertex edge data into dense JAX arrays:

```python
max_edges = max(len(v.parameterized_edges()) for v in vertices if v.edges_length() > 0)

all_edge_coeffs = np.zeros((n_verts, max_edges, n_params))
all_edge_targets = np.full((n_verts, max_edges), -1, dtype=np.int32)
all_n_edges = np.zeros(n_verts, dtype=np.int32)

for vi in range(n_verts):
    v = vertices[vi]
    edges = v.parameterized_edges()
    if not edges:
        continue
    for ei, e in enumerate(edges):
        coeffs = list(e.edge_state(n_params))
        all_edge_coeffs[vi, ei, :len(coeffs)] = coeffs
        all_edge_targets[vi, ei] = e.to().index()
    all_n_edges[vi] = len(edges)

# Convert to JAX arrays (static, not traced)
all_edge_coeffs = jnp.array(all_edge_coeffs)
all_edge_targets = jnp.array(all_edge_targets)
all_n_edges = jnp.array(all_n_edges)
```

### Step 6: JIT-Compatible Correction Function

```python
def likelihood_correction(theta_mcmc):
    """Fully JIT-compatible BFFG correction."""
    # Split JAX PRNG for all samples
    keys = jax.random.split(jax.random.PRNGKey(0), n_loci * n_paths)

    # theta_target at all times: precompute for masking
    # (This requires theta_target_fn to be JAX-traceable)

    total = 0.0
    for locus in range(n_loci):
        target_v = observed_data[locus]
        log_weights = jnp.empty(n_paths)

        for m in range(n_paths):
            key = keys[locus * n_paths + m]
            seed = jax.random.randint(key, (), 0, 2**31).astype(jnp.uint32)

            # FFI: sample conditioned path (fixed-size output)
            v_indices, e_times = sample_path_conditioned_ffi(
                structure_dict, jnp.array(theta_proposal),
                jnp.array([target_v], dtype=jnp.int32),
                seed, max_path_length
            )

            # Precompute theta_target at each time in the path
            theta_at_times = jax.vmap(
                lambda t: theta_target_fn(theta_mcmc, t)
            )(e_times)

            # Pure JAX weight computation
            log_weights = log_weights.at[m].set(
                _importance_log_weight_jax(
                    v_indices, e_times,
                    all_edge_coeffs, all_edge_targets, all_n_edges,
                    jnp.array(theta_proposal), theta_at_times
                )
            )

        total += logsumexp(log_weights) - jnp.log(n_paths)

    return total
```

Note: The outer loops over loci and paths could also be vmapped if the path sampling FFI supports batching.

## Prerequisite: `theta_target_fn` Must Be JAX-Traceable

For the epoch model: `lambda theta, t: [1/N(t), 1/N(t), 1.0, mut]` where `N(t) = theta[0] if t < boundary else theta[1]`.

This uses `jnp.where` instead of Python if/else:
```python
def theta_target_fn(theta_mcmc, t):
    rate = jnp.where(t < epoch_boundary, 1.0 / theta_mcmc[0], 1.0 / theta_mcmc[1])
    return jnp.array([rate, rate, 1.0, mutation_rate])
```

## Files to Modify

| File | Change | Estimated LOC |
|------|--------|---------------|
| `api/c/phasic.h` | `ptd_random_sample_path_conditioned_fixed` declaration | 10 |
| `src/c/phasic.c` | Implementation with seed + fixed-size buffers | 50 |
| `src/cpp/parameterized/ffi_handlers.cpp` | Two new FFI handlers (sampling + backward probs) | 80 |
| `src/cpp/phasic_pybind.cpp` | Capsule registration for new FFI targets | 15 |
| `src/phasic/ffi_wrappers.py` | `sample_path_conditioned_ffi`, `backward_probabilities_ffi` | 60 |
| `src/phasic/bffg.py` | Dense precomputation, JAX weight function, JIT correction | 120 |
| **Total** | | **~335 LOC** |

## Verification

1. **JIT compilation**: `jax.jit(likelihood_correction)(theta)` completes without error
2. **vmap**: `jax.vmap(likelihood_correction)(theta_batch)` parallelizes across chains
3. **Numerical**: Results match non-JIT version within stochastic tolerance (same seed → same output)
4. **Performance**: Benchmark JIT vs non-JIT correction on epoch model with M=50, 30 loci
5. **Existing tests**: All `test_bffg.py`, `test_mcmc.py`, `test_conditioned_sampling.py` pass
6. **Notebook**: Part 3 runs with `jit=True` (default) without `jit=False` override

## Dependencies

- JAX FFI API (already used for `compute_sojourn_times_ffi`)
- XLA headers for FFI compilation (already required for existing FFI)
- `theta_target_fn` provided by user must be JAX-traceable

## Risk Assessment

- **Medium risk**: FFI handler for stochastic sampling is novel (existing handlers are deterministic). Thread-safety of `srandom`/`random` needs care — use thread-local seed or per-call reseeding.
- **Low risk**: Weight computation in JAX is straightforward once edge data is dense.
- **Low risk**: Backward compatibility maintained via `return_model` parameter.

## Code to Remove After Implementation

The current thread-based parallelism workaround (documented in `docs/mcmc-thread-parallelism.md`) becomes obsolete. Remove:

### In `src/phasic/mcmc.py`:

1. **JIT auto-detection guard** (~line 765): Remove the `can_jit` logic that disables JIT when `likelihood_correction` is set. With JIT-compatible correction, JIT should always work.

2. **vmap auto-detection guard** (~line 410): Remove the `can_vmap` check that blocks vmap when `likelihood_correction` is set. The correction will be vmappable.

3. **Correction outside JIT in `_run_chain`** (~lines 500, 555): Remove the two blocks that apply `self.likelihood_correction(theta)` after `log_prob_fn()`. Instead, put the correction back inside `_log_prob` (which will be JIT'd):
   ```python
   # In _log_prob:
   if self.likelihood_correction is not None:
       log_lik = log_lik + self.likelihood_correction(theta)
   ```

4. **ThreadPoolExecutor in `_run_chains_vmap`** (~line 690): Remove the threaded correction block:
   ```python
   # Remove this entire block:
   if self.likelihood_correction is not None:
       from concurrent.futures import ThreadPoolExecutor
       def _eval_correction(c): ...
       with ThreadPoolExecutor(...) as pool: ...
       lp_proposed = lp_proposed + jnp.array(corrections)
   ```
   The correction will be evaluated inside `vmap_log_prob` automatically.

5. **Initial lp_all correction in `_run_chains_vmap`** (~line 625): Remove the per-chain correction loop after `lp_all = vmap_log_prob(phi_all)`.

### In `src/phasic/bffg.py`:

1. **`_full_importance_log_weight`** (~lines 336-376): Replace with `_importance_log_weight_jax` (pure JAX version with `jax.lax.scan`).

2. **Sparse edge coefficient caching** (~lines 314-334): Replace with dense JAX array precomputation (Step 5 in this plan).

3. **`likelihood_correction` function** (~lines 430-455): Rewrite to use FFI path sampling + JAX weight computation.

4. **`log_prob_fn` backward-compatible path** (~lines 458-490): Update to use FFI for model evaluation (already done) + JIT-compatible correction.

### Files to delete:

- `docs/mcmc-thread-parallelism.md` — obsoleted by this implementation.
