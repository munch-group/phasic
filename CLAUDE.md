# phasic - Quick Reference

**Version:** 0.22.0
**Paper:** [Røikjer, Hobolth & Munch (2022)](https://doi.org/10.1007/s11222-022-10155-6) - Statistics and Computing
**Repository:** https://github.com/munch-group/phasic
**Contact:** Kasper Munch (kaspermunch@birc.au.dk)


## Overview

**phasic** is a high-performance library for computing with **phase-type distributions** using graph-based algorithms. Phase-type distributions model the time until absorption in continuous or discrete-time Markov chains on finite state spaces.

### Key Innovation

Traditional matrix-based methods become computationally infeasible for systems with thousands of states. This library uses **graph-based algorithms** that:
- Execute **10-100x faster** than matrix methods for sparse systems
- Use dramatically less memory (O(n+m) vs O(n²))
- Scale to large state spaces (500,000+ states)
- Support iterative construction of complex models

### Primary Applications

- **Population genetics**: Coalescent models, site frequency spectra
- **Queuing theory**: Service time modeling, system reliability
- **Survival analysis**: Time-to-event modeling
- **Bayesian inference**: Efficient likelihood computation for MCMC/SVGD

## Development

- Always use pixi environment.
- Use "pixi run install-dev" for development install.
- Do not ever implement silent fallbacks. Code should work as specified or fail.
- In any response making claims about the code (signatures, behavior, file
  contents, line numbers, whether an API exists), explicitly state for each claim
  whether it is **grounded** — verified by reading the actual source this session
  (cite `file:line`) — or a **guess/recollection/inference**. Prefer reading the
  source over guessing; when you have not verified, say so plainly rather than
  presenting speculation as fact.


## Key Concepts

### Phase-Type Distributions

A **continuous phase-type (PH) distribution** represents the time until absorption in a continuous-time Markov chain:

- **PH(α, S)** where α = initial probability vector, S = sub-intensity matrix
- PDF: f(t) = α · exp(S·t) · s* (forward algorithm, Algorithm 4)
- Moments: E[Tᵏ] computed via reward transformation (Algorithm 2)

**Discrete phase-type (DPH)** distributions model number of jumps until absorption.

### Graph Representation

**Vertices** = states, **Edges** = transitions with rates/probabilities

- **Parameterized edges**: weight depends on `weight_mode`:
  - `'linear'` (default): weight = c₁θ₁ + c₂θ₂ + ... + cₙθₙ
  - `'log'`: weight = (c₁θ₁)(c₂θ₂)···(cₙθₙ) (multiplicative, computed in log-space)
  - `'callback'`: weight = callback(θ, coefficients) (arbitrary Python function; slow, and rejected on the daisy-chain path)
  - `'formula'`: weight = a formula string compiled to a bytecode tape and evaluated **per edge in C** (fast like linear/log; the only way to use non-inner-product weights on the daisy-chain/SVGD fast path). Set via `graph.weight_formula`.
- **Graph elimination** (Algorithm 3): Converts cyclic → acyclic via Gaussian elimination on graph
- **Sparse graphs**: Only store actual transitions, not full n×n matrix

### Trace-Based Elimination (Phases 1-4)

**Phase 1-2**: Record elimination operations as linear trace
**Phase 3**: SVGD integration with JAX (jit/grad/vmap)
**Phase 4**: ✅ Exact phase-type likelihood using C forward algorithm
**Phase 5 Week 3**: ✅ Forward algorithm PDF gradients in C
**Phase 5 (continuation)**: (In progress) JAX FFI gradients for full autodiff support

**Advantage**: Record once (O(n³)), evaluate many times (O(n)) → 5-10x faster than symbolic DAG for SVGD workloads

### JAX Integration

**Pattern**: Serialize graph → C++ builder → `jax.pure_callback` → JAX compatible

Supports:
- `jax.jit`: JIT compilation
- `jax.grad`: Automatic differentiation (via custom VJP)
- `jax.vmap`: Batching over parameters
- `jax.pmap`: Multi-device parallelization

## Python API Patterns

### Building Graphs

```python
from phasic import Graph
import numpy as np

# Callback-based construction
def coalescent_callback(state):
    n = state[0]
    if n <= 1:
        return []
    rate = n * (n - 1) / 2
    # For parameterized: return (next_state, [coeff_vector])
    return [(np.array([n - 1]), [rate])]

g = Graph(
    state_length=1,
    callback=coalescent_callback,
    parameterized=True,  # Enable parameterized edges
    nr_samples=5
)

# Or manual construction
g = Graph(1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v0.add_edge(v1, [2.0, 0.5])
# Edge weight = 2.0*θ[0] + 0.5*θ[1]
```

### Weight Modes

Parameterized edges support four weight computation modes that flow through the full JAX/FFI/SVGD pipeline:

```python
# Linear (default): weight = Σ c_k θ_k
g.weight_mode = 'linear'

# Log/multiplicative: weight = Π(c_k θ_k)
# Useful when coefficients and parameters represent factors
g.weight_mode = 'log'

# Callback: weight = callback(theta, coefficients)
# For arbitrary non-linear weight functions
def resistance_weight(theta, coeffs):
    return np.exp(-np.dot(coeffs, theta))

g.weight_callback = resistance_weight  # Automatically sets weight_mode='callback'

# Formula: weight = an expression evaluated PER EDGE IN C (fast, no Python in
# the hot loop). t0,t1,... = theta; c0,c1,... = the edge's coefficients.
g.weight_formula = "exp(c0*t0 + c1*t1) + c2"  # Automatically sets weight_mode='formula'
```

`weight_formula` is also available as a **one-shot kwarg** on `graph.update_weights`
and `graph.svgd` (sugar over the property, mirroring `callback=`; mutually
exclusive with it — rule R22). The `update_weights` form is handy for verifying a
formula via `graph.pdf(...)` before running SVGD; it does NOT change the graph's
persistent `weight_mode`:

```python
g.update_weights(theta, weight_formula="exp(c0*t0)")  # one-shot; then g.pdf(...) to check
g.svgd(obs, weight_formula="exp(c0*t0)", ...)          # one-call inference (works with epoch_starts)
```

The `SVGD` *class* takes an already-built model, so it has **no** `weight_formula`
kwarg (it would silently no-op). For direct `SVGD(model, ...)`, set the formula on
the graph BEFORE building the model:
`g.weight_formula = "…"; model = Graph.pmf_from_graph(g); SVGD(model, obs, ...)`.

Linear/log/formula compute weights in C and stay on the OpenMP-parallel FFI path
(`pmf_from_graph`, `pmf_and_moments_from_graph`, `graph.svgd()`, incl. the
daisy-chain `epoch_starts=` path); gradients are finite-difference via the FFI's
`custom_vjp`, so a formula needs no analytic Jacobian. Callback mode uses
`jax.pure_callback` (sequential, Python overhead per evaluation) and is **rejected
on the daisy-chain path** (R21) — use `weight_formula` there instead.

**`weight_formula` language** (`src/phasic/weight_formula.py`): `+ - * / **`,
`exp log sqrt logistic pow`, comparisons `== != < > <= >=`, `delta(a,b)`
(Kronecker), `and(x,y) or(x,y) not(x)`, and `select(cond, a, b)` (branchless
ternary). Comparison/`delta`/boolean conditions and `select`'s condition must be
**theta-independent** (may use `c<j>`/constants but not `t<i>`); a theta-dependent
condition raises at assignment (use `logistic(...)` for smooth theta-gating). The
formula is compiled once to a JSON-native bytecode tape that rides `serialize()`
into the C++ `GraphBuilder`; the C VM is in `ptd_weight_tape_eval` (`src/c/phasic.c`,
consulted by `ptd_graph_update_weights`). `record_elimination_trace` rejects
formula mode (it would silently compute the linear inner product) — use the FFI
path. Trace path mirroring of formula weights is a deliberate non-goal for now.

**Per-edge constant folding (performance).** `ptd_graph_update_weights` does not
run the full tape per edge per theta. On the first call it *specializes* the tape
for each edge (`ptd_weight_tape_specialize`): every theta-INDEPENDENT
subexpression is folded to a constant using that edge's coefficients, and dead
`select()` arms are pruned (the conditions are theta-independent by rule), giving a
small **theta-only residual tape** cached per edge (`graph->wf_residuals`,
invalidated when the tape changes). Subsequent `update_weights` run only the
residuals. Effect: a complex formula (e.g. a sum of `select()` dispatches) runs
~as fast as the linear inner product in the SVGD forward — only the taken arm
runs, with its coefficient arithmetic precomputed once (measured: a 5-`select`
coalescent-with-selection `update_weights` 167 µs → 35 µs; full forward 363 µs →
224 µs ≈ the 214 µs inner-product forward). The residual is bit-identical to the
full tape for every theta; on any allocation failure the code falls back to the
full tape (same result).

**`weight_formula` works on all SVGD paths**: the direct/reward path, the
`joint_index` path, and the daisy-chain `epoch_starts` path. On the joint paths,
set the formula on the **joint-prob graph** (`jpg = g.joint_prob_graph(...);
jpg.weight_formula = "..."`), using *that graph's* coefficient layout —
`joint_prob_graph` carries the base edges' FULL coefficients and appends a mutation
slot (so jpg coefficient length = base coefficient length + 1, theta dim = base
param_length + 1), and does **not** propagate a formula set on the base graph.

**`theta_dim` < coefficient length** (the rate depends on per-edge data beyond the
optimized parameters): build with `Graph(callback, ..., theta_dim=k)` (or
`g.set_param_length(k)` before adding edges) — `param_length` stays `k` while edges
carry longer coefficient vectors; the formula references `t0..t(k-1)` (theta) and
`c0..c(m-1)` (the full per-edge coefficients). Works on the direct, FFI/SVGD
(non-daisy), `joint_index`, and daisy (`epoch_starts`) paths, and survives the
`graph_cache` save/load roundtrip (`from_serialized` pins `param_length`). Through
`joint_prob_graph` the mutation slot is appended at index = base coefficient length
and the mutation rate parameter at theta index = base param_length, so author the
joint formula accordingly, e.g.
`select(c<base_coeff_len>==0, <base rate>, c<base_coeff_len> * t<base_param_len>)`
(the `==0` arm = dynamics edges; the other = mutation/trash/absorbing edges).

**`theta_dim` resolution in `svgd`** (when not passed explicitly) is by weight
mode, so it is always either specified or reliably inferred — never silently
taken to be the coefficient length when that is wrong:
- **linear/log**: `param_length()` (= coefficient length; the two must match).
- **formula**: the formula's own `n_theta` (highest `t`-index + 1) — so a
  formula model needs NO `theta_dim` at construction. An explicit `theta_dim`
  may exceed `n_theta` (reserve extra params); a smaller one raises.
- **callback** (kwarg `callback=` OR the `weight_callback` property): cannot be
  inferred (a callback is a black box) — `svgd` raises `SvgdConfigError` unless
  `theta_dim=`/`theta_init=` is given. `update_weights(theta, callback=...)`
  itself imposes no length check (the callback maps θ→coefficients).
(Resolution lives in `Graph.svgd`; `epoch_starts`/joint-prob graphs carry
`param_length == n_theta`, so the per-epoch dimension is unaffected.)

**Fixed-parameter SVGD is now cheap**: `svgd(..., fixed=[(i, v), ...])` skips the
finite-difference perturbation of fixed dims on every path (the gradient is 0 and
was discarded anyway), so cost scales with the number of FREE parameters, not the
total. Value-preserving for the learnable dimensions.

### Computing PDF/PMF

```python
# Direct C++ call (fast, not JAX-differentiable)
pdf_value = graph.pdf(time, granularity=0)  # granularity=0 → auto
pmf_value = graph.dph_pmf(jumps)

# Vectorized
times = np.array([0.5, 1.0, 1.5])
pdf_values = graph.pdf(times, granularity=100)

# JAX-compatible (for gradients)
import jax.numpy as jnp
from phasic.ffi_wrappers import compute_pmf_ffi

structure_json = graph.serialize()
theta = jnp.array([1.0, 0.5])
times = jnp.linspace(0.1, 5.0, 100)
pdf = compute_pmf_ffi(structure_json, theta, times, discrete=False, granularity=100)

# Works with JAX transformations
jitted = jax.jit(compute_pmf_ffi, static_argnums=(0, 3, 4))
grad_fn = jax.grad(lambda t: jnp.sum(compute_pmf_ffi(structure_json, t, times, False, 100)))
```

### Trace Elimination Workflow

```python
from phasic.trace_elimination import (
    record_elimination_trace,
    evaluate_trace_jax,
    instantiate_from_trace,
    trace_to_log_likelihood
)

# 1. Record trace (once, ~ms for 67 vertices)
trace = record_elimination_trace(graph, param_length=2)

# 2. Evaluate with concrete parameters (fast, O(n))
result = evaluate_trace_jax(trace, jnp.array([1.0, 2.0]))
# Returns: {'vertex_rates': ..., 'edge_probs': ..., 'vertex_targets': ...}

# 3. Instantiate concrete graph from trace
concrete_graph = instantiate_from_trace(trace, np.array([1.0, 2.0]))
pdf = concrete_graph.pdf(times, granularity=0)

# 4. For SVGD: exact phase-type likelihood
observed_times = np.array([1.5, 2.3, 0.8])
log_lik = trace_to_log_likelihood(trace, observed_times, reward_vector=None, granularity=0)

# Use with SVGD
from phasic import SVGD
svgd = SVGD(log_lik, theta_dim=2, n_particles=100, n_iterations=1000)
results = svgd.fit()
```

### SVGD Inference

```python
# High-level SVGD API
from phasic import Graph

# Build parameterized model
model = Graph.pmf_from_graph(graph, discrete=False)

# Run SVGD
results = Graph.svgd(
    model=model,
    observed_data=observations,
    theta_dim=2,
    n_particles=100,
    n_iterations=1000,
    learning_rate=0.01
)

print(f"Posterior mean: {results['theta_mean']}")
print(f"Posterior std: {results['theta_std']}")
```

### Profiling a model & choosing settings

Before committing to an expensive first evaluation of a large parameterized
graph, `phasic.profile_graph(graph)` (or `graph.profile()`) analyses it and
recommends — **with reasons** — the three settings that otherwise have to be
guessed:

```python
import phasic
prof = phasic.profile_graph(graph)      # or graph.profile()
print(prof)
# phasic graph profile — 1044 vertices, 8099 edges, param_length=2
#   SCC structure : 123 SCCs, largest 34 (3%), 13 levels, widest level 23
#   parallel_elim : RECOMMEND ON (23 independent SCCs at the widest level, 10 cores; ceiling ~6.1x; ...)
#   dyn_ordering  : leave OFF (largest SCC only 34 vertices — elimination <10 ms, ordering is immaterial)
#   eval path     : forward-PDF OK (max_rate 66 -> granularity 132)
#   -> phasic.configure(parallel_elimination=True)

prof.recommendations            # {'parallel_elimination': (bool, why), 'dyn_ordering': (bool|False, why), 'path': (str, why)}
prof.apply_snippet              # ready-to-paste configure(...) line
```

How each recommendation is derived:
- **`parallel_elimination`** — *structural* (cheap, O(V+E)): from the SCC
  condensation (`scc_decomposition` + level sets). ON only when there are
  independent SCCs to run concurrently and the work is not Amdahl-dominated by one
  giant SCC. A linear chain (e.g. single-locus coalescent) → OFF; the SCC-rich
  two-locus ARG → ON.
- **`dyn_ordering`** — *measured*: the largest SCC's synthetic graph is eliminated
  twice (default vs min-out-degree) and wall-times compared. dyn's fill benefit is
  model-dependent and not predictable statically, so phasic probes rather than
  guesses. Sub-10 ms eliminations are treated as "ordering immaterial".
- **eval path** — from `max_rate` → forward-PDF auto-granularity (`2·max_rate`).
  A high `max_rate` (stiff model) makes the forward-PDF likelihood slow per
  evaluation; **if your likelihood is over a joint/discrete index rather than
  continuous observation times**, use the `max_rate`-independent joint/sojourn
  path (`Graph.pmf_from_graph_joint_index` → `expected_sojourn_time`) instead.
  The forward-PDF path is only mandatory when you need the continuous density
  f(t) at observed times. `pass probe_dyn=True/False` to force/skip the dyn probe.

The dyn probe costs two eliminations of the *largest* SCC only (skipped by default
when that SCC is very large). The structural and path tiers are effectively free.

> **Import phasic before importing jax / creating jax arrays.** phasic enables
> `jax_enable_x64` on import; the C FFI requires float64 buffers. jax arrays
> created *before* `import phasic` are float32 and trip the FFI check "Wrong
> buffer dtype: expected F64 but got F32". If you hit this, restart the kernel
> and import phasic first (or recreate the arrays).

### Per-Observation Exposure (`exposure` / `exposure_param_index`)

Many phase-type inference problems have an observation-specific known
quantity that multiplies a rate parameter — what GLM literature calls
**exposure** (or "offset" in log-domain). For each observation $i$ the
model is evaluated at $\boldsymbol{\theta}^{(i)}$ where
$\theta^{(i)}_j = \theta_j$ for $j \neq k$ and
$\theta^{(i)}_k = \theta_k \cdot \alpha_i$, with
$k$ = `exposure_param_index` and $\alpha_i$ = `exposure[i]`. Use this
construct whenever the model's likelihood depends on a known
per-observation scaling of one rate-typed component of $\boldsymbol{\theta}$:

- **Coalescent-with-mutation**: $\alpha_i$ = segment length $L_i$ in
  bases; $\theta_k$ is the per-base mutation rate.
- **Survival / failure-time**: $\alpha_i$ = time-at-risk for unit
  $i$; $\theta_k$ is the hazard rate.
- **Spatial Poisson**: $\alpha_i$ = area or volume of region $i$;
  $\theta_k$ is the intensity per unit area.

```python
result = graph.svgd(
    observed_data=count_vectors,         # (n_observations,) or (n_observations, n_features)
    theta_dim=2,
    exposure=alphas,                     # scalar or 1D of length n_observations
    exposure_param_index=0,              # theta[0] is the rate parameter
    n_particles=40, n_iterations=400,
)
```

`exposure=None` (the default) reproduces existing behaviour exactly.
Sparse observations (`SparseObservations`) are not supported — they
carry no per-observation identity; convert to dense first.

#### Coverage matrix

| Combination | Status | Notes |
|---|---|---|
| Standard PMF (`rewards=None`) | ✅ | `_wrap_model_with_exposure` wraps the model; one model call per obs via `lax.map`. |
| 1D rewards | ✅ | Same wrapper path. |
| 2D rewards (multivariate) | ⚠️ | Wrapper applied; not benchmarked. R11 emits a warning. |
| `SparseObservations` | ❌ | Rejected by R6 — sparse format has no per-observation identity. |
| Vanilla joint-prob (`epoch_starts=None`) | ❌ | Rejected by R9 — wrapper would fan out to one full graph elimination per observation. Use a single-epoch daisy chain instead: `epoch_starts=[0.0]`. |
| Daisy-chain joint-prob (`epoch_starts=[…]`) | ✅ | Per-obs scaling pushed inside the daisy-chain FFI as a single batched call (`(n_obs, n_epochs * param_length)` theta). `parallel_mode` is forced to `'none'` to avoid stacking vmap on top of the FFI's own batch dimension. |

#### `exposure_param_index` semantics under daisy-chain

Under daisy chain (`epoch_starts=[…]`), the SVGD theta is flat with
shape `(n_epochs * param_length,)`. **`exposure_param_index` remains
the *local* per-epoch index** in `[0, param_length)`; the validator
broadcasts it across all epochs internally as
`[exposure_param_index, param_length + exposure_param_index, …, (n_epochs-1)*param_length + exposure_param_index]`.
This keeps the user-facing API uniform whether or not `epoch_starts`
is set. R8 enforces the local-index range; R14 enforces the same range
for `fixed=[(local_idx, value), …]` entries.

All combinations are checked once at construction time by
`phasic.svgd_config.validate(...)` (rules R1..R15). On a violation the
call raises `phasic.exceptions.SvgdConfigError` with a message that
names the offending combination, why it is invalid, and what to do
instead.

#### Performance: forward cost scales with *unique* exposure values

Under daisy chain, each observation's effective theta is the SVGD
particle with `exposure_param_index` slot scaled by `exposure[i]`.
Internally phasic deduplicates identical theta rows before calling
the C++ FFI, so **the forward computation cost scales with the number
of unique exposure values, not with `n_obs`**.

If your exposures are continuous (e.g. genomic segment lengths, time-
at-risk durations), rounding them to a coarser scale before passing
to `svgd` can dramatically reduce wall time without losing accuracy
of the SVGD posterior:

```python
# 312 continuous exposure values → 312 unique rows → 312 chains per FFI call
svgd_slow = graph.svgd(observations, exposure=tree_spans,
                       exposure_param_index=1, epoch_starts=[0, ...])

# Round to nearest 1000 → ~30 unique rows → ~30 chains per FFI call (~10× faster)
svgd_fast = graph.svgd(observations,
                       exposure=np.round(tree_spans, -3),
                       exposure_param_index=1, epoch_starts=[0, ...])
```

This is **not an approximation introduced by phasic** — identical
theta rows produce bit-identical FFI output, and the per-obs results
are scattered back via `inverse_idx`. The approximation comes from
your rounding choice, which you control. The daisy-chain joint-prob
distribution is smooth in `α`, so coarse rounding (e.g. nearest 1000
on segment lengths in the 1e3–1e6 range) typically gives <1% error
in the posterior.

### Multivariate Phase-Type Models (2D Observations & Rewards)

**New in v0.21.4**: Support for multivariate phase-type distributions where each feature dimension has its own reward vector.

**Updated in v0.22.22**: Reward matrix shape changed to `(n_features, n_vertices)` for more intuitive indexing.

```python
from phasic import Graph
import jax.numpy as jnp

# Create parameterized graph
graph = Graph(model_callback)

# Create multivariate model
model = Graph.pmf_and_moments_from_graph_multivariate(
    graph,
    nr_moments=2,
    discrete=False
)

# Setup multivariate data
n_times = 100
n_features = 3  # e.g., 3 marginal distributions
n_vertices = graph.vertices_length()

# 2D observations: (n_times, n_features)
observed_data = jnp.array([
    [obs_feature_0, obs_feature_1, obs_feature_2],
    ...
])  # Shape: (100, 3)

# 2D rewards: (n_features, n_vertices) - UPDATED v0.22.22
# Each row defines the complete reward vector for one marginal
rewards_2d = jnp.array([
    [r0_feat0, r1_feat0, r2_feat0, ...],  # Feature 0 reward vector
    [r0_feat1, r1_feat1, r2_feat1, ...],  # Feature 1 reward vector
    [r0_feat2, r1_feat2, r2_feat2, ...],  # Feature 2 reward vector
])  # Shape: (3, n_vertices)

# Run SVGD with multivariate model
svgd_result = graph.svgd(
    observed_data=observed_data,
    theta_dim=2,
    n_particles=100,
    n_iterations=1000,
    rewards=rewards_2d  # Pass 2D rewards
)

# Or use SVGD directly
from phasic import SVGD
svgd = SVGD(
    model=model,
    observed_data=observed_data,
    theta_dim=2,
    n_particles=100,
    rewards=rewards_2d
)
svgd.optimize()
```

**Key Features:**
- **Independent computation**: Each feature dimension computed separately with its reward vector
- **Log-likelihood**: Sum over all observation elements: `Σᵢⱼ log(PMF[i,j])`
- **Moment regularization**: Moments aggregated across features (mean)
- **NaN handling**: Missing observations (NaN) skipped in likelihood; NaN PMF for valid obs raises error
- **Backward compatible**: 1D rewards work exactly as before

**Shape Convention (v0.22.22):**
- Rewards: `(n_features, n_vertices)` - each row is one feature's reward vector
- Observations: `(n_times, n_features)`
- PMF output: `(n_times, n_features)` for 2D rewards, `(n_times,)` for 1D
- Moments output: `(n_features, nr_moments)` for 2D rewards, `(nr_moments,)` for 1D

**Migration from v0.21.4:**
```python
# Old shape (v0.21.4)
rewards_old = jnp.ones((n_vertices, n_features))  # Column per feature

# New shape (v0.22.22)
rewards_new = rewards_old.T  # Transpose: row per feature
```

### Sparse Observation Format (NEW)

For multivariate models where different features have different numbers of observations, use `SparseObservations` instead of dense NaN-padded arrays. This avoids NaN propagation through JAX gradients.

```python
from phasic import SparseObservations, dense_to_sparse, SVGD

# Option 1: Convert from dense NaN-padded array
dense_obs = jnp.array([
    [1.0, np.nan, 3.0],   # Feature 0 has value, Feature 1 missing, Feature 2 has value
    [1.5, 2.0, np.nan],   # etc.
])
sparse_obs = dense_to_sparse(dense_obs)

# Option 2: Create directly with pre-computed slices
# (Required for JAX JIT compatibility)
sparse_obs = SparseObservations(
    values=jnp.array([1.0, 1.5, 2.0, 3.0]),  # All valid values, grouped by feature
    features=jnp.array([0, 0, 1, 2]),         # Feature index for each value
    n_features=3,
    slices=((0, 2), (2, 3), (3, 4))           # (start, end) indices per feature
)

# Use in SVGD
svgd = SVGD(
    model=model,
    observed_data=sparse_obs,  # Works just like dense observations
    theta_dim=2,
    rewards=rewards_2d
)
svgd.optimize()
```

**Key Benefits:**
- No NaN propagation through JAX callbacks
- Memory efficient for very sparse observation patterns
- Supports unequal observation counts per feature
- `dense_to_sparse()` automatically computes slices

### Reward Transformation

```python
# Transform for higher moments or multivariate distributions
rewards = np.array([1.0, 2.0, 0.5, ...])  # One per vertex
transformed_graph = graph.reward_transform(rewards)

# For k-variate phase-type (v0.22.22+ shape):
reward_matrix = np.array([
    [r1_1, r2_1, r3_1, ...],  # Marginal 1: reward vector across all vertices
    [r1_2, r2_2, r3_2, ...],  # Marginal 2: reward vector across all vertices
    ...
    [r1_k, r2_k, r3_k, ...],  # Marginal k: reward vector across all vertices
])
# Each row = one marginal's complete reward vector
```

### HexGrid Spatial Models

`HexGrid` creates hexagonal grids within geographic boundaries, producing `Property` objects that compose with `StateIndexer` for spatial phase-type models.

```python
from phasic import Graph, HexGrid, StateIndexer, Property

# Create grid from shapefile
grid = HexGrid.from_shapefile('africa.shp', hex_size=5)

# Grid properties compose with other state properties
indexer = StateIndexer(
    cell=grid.properties() + [Property('lineage', min_value=1, max_value=2)]
)

# Transition function receives grid as second argument
def migration(state, grid, indexer=None, rate=None):
    transitions = []
    # ... use grid.neighbors(row, col), indexer.cell.p2i(), etc.
    return transitions

# Build graph via standard Graph constructor
graph = Graph(migration, ipv=[(initial, 1)], grid=grid, indexer=indexer, rate=1.0)

# Or via HexGrid.build_graph for simpler models
graph = grid.build_graph(transition_fn, indexer.cell)

# Map results back to grid
grid_values = grid.map_to_grid(graph, indexer.cell, values)
```

Key methods: `grid.neighbors(r, c)`, `grid.coords_to_rowcol(x, y)`, `grid.valid_cells()`, `grid.properties()`.

## Important Implementation Details

### Graph Elimination (Algorithm 3)

**Purpose**: Convert cyclic graph → acyclic graph for moment computation

**Algorithm**: Gaussian elimination on graph structure
- Iterate through vertices in order
- For each vertex i being eliminated:
  - For each parent → child pair:
    - Add bypass edge (or update existing)
    - Renormalize probabilities
  - Remove edges to eliminated vertex

**Complexity**: O(n³) one-time, enables O(n²) moment computation

### Forward Algorithm (Algorithm 4)

**Purpose**: Compute exact PDF/PMF via uniformization

**Used by**: `graph.pdf(time, granularity)`, `graph.dph_pmf(jumps)`

**Algorithm** (continuous):
1. Discretize time using uniformization (granularity = 2 × max_rate by default)
2. Simulate discrete-time chain via dynamic programming
3. Track probability mass at each vertex over time
4. Sum mass reaching absorbing states

**Complexity**: O(t · n² · g) where t = time, n = vertices, g = granularity

**Note**: This is the **exact** phase-type PDF computation, not an approximation

### JAX FFI Integration Pattern

**Location**: `src/phasic/ffi_wrappers.py`

**Pattern**:
```python
def compute_pmf_fallback(structure_json, theta, times, discrete, granularity):
    # Wrap C++ call with jax.pure_callback
    result_shape = jax.ShapeDtypeStruct(times.shape, jnp.float64)

    return jax.pure_callback(
        lambda theta_jax, times_jax: _compute_pmf_impl(
            structure_json,
            np.asarray(theta_jax),
            np.asarray(times_jax),
            discrete,
            granularity
        ),
        result_shape,
        theta,
        times,
        vmap_method='sequential'  # Enable vmap
    )
```

**Key**: `pure_callback` allows JAX jit/vmap while calling C++ code

### Performance Characteristics

**Graph vs Matrix** (sparse systems):
- PDF computation: 10-100x faster
- Memory: 100-1000x less (O(n+m) vs O(n²))
- Scales to 500K+ states (vs 10K for matrices)

**Trace vs Symbolic DAG** (repeated evaluation):
- Setup: ~0.5x time (faster to record trace)
- Evaluation: 5-10x faster per parameter vector
- Break-even: ~6 evaluations
- Ideal for SVGD: 100-1000 evaluations

**Phase 3 Targets** (1000 SVGD evaluations):
- 37 vertices: <5 min ✓ (actual: ~2s)
- 67 vertices: <30 min ✓ (actual: ~5s)

## Phase 4: Exact Phase-Type Likelihood ✅ COMPLETE

**Status**: ✅ Implemented October 2025

Upgraded `trace_to_log_likelihood()` from exponential approximation to exact phase-type likelihood using forward algorithm (Algorithm 4).

**Key Changes**:
- Use `instantiate_from_trace()` + `graph.pdf()` for exact PDF computation
- Add `granularity` parameter for accuracy control (default=100)
- Reward vector support: falls back to exponential with warning (exact support planned)
- Performance: 4.7ms per evaluation for simple models, well under targets

**Accuracy Improvement**:
- Erlang(3) distribution: Difference of 1.33 in log-likelihood vs exponential
- Critical for multi-stage phase-type distributions
- Exact computation essential for correct Bayesian inference

**Usage**:
```python
trace = record_elimination_trace(graph, param_length=2)
observed_times = np.array([1.5, 2.3, 0.8, 1.2])

# Exact likelihood with granularity control
log_lik = trace_to_log_likelihood(trace, observed_times, granularity=100)

# Use with SVGD
from phasic import SVGD
svgd = SVGD(log_lik, theta_dim=2, n_particles=100, n_iterations=1000)
results = svgd.fit()
```

**Performance**:
- 5-10× slower than exponential approximation
- Still meets Phase 3 targets with margin
- 67-vertex model: ~50s for 1000 evaluations (target: <2 min)

## Phase 5 Week 3: Forward Algorithm PDF Gradients ✅ COMPLETE

**Status**: ✅ Implemented October 2025

Implemented machine-precision gradient computation for phase-type distribution PDFs using uniformization-based forward algorithm.

**Key Features**:
- Exact PDF and gradient computation with error ≤ 2.05e-16
- Two API workflows: direct parameter passing and integrated parameter update
- Zero API signature changes - full backward compatibility with C++/R/Python
- Minimal code changes for gradient support

**API Functions**:

1. **`ptd_graph_pdf_with_gradient()`** - Direct parameter passing:
```c
int ptd_graph_pdf_with_gradient(
    struct ptd_graph *graph,
    double time,
    size_t granularity,
    const double *params,
    size_t n_params,
    double *pdf_value,
    double *pdf_gradient
);
```

2. **`ptd_graph_pdf_parameterized()`** - Integrated workflow:
```c
// Set parameters first
ptd_graph_update_weight_parameterized(graph, theta, n_params);

// Compute PDF + gradients using stored parameters
int ptd_graph_pdf_parameterized(
    struct ptd_graph *graph,
    double time,
    size_t granularity,
    double *pdf_value,
    double *pdf_gradient  // NULL for PDF-only
);
```

**Architecture Solution**:
Preserves original edge coefficients for gradient computation while allowing `update_weight_parameterized()` to store concrete weights for fast PDF computation.

**Test Results**:
- `test_single_exp_grad.c`: error = 0.00e+00 ✓
- `test_c_pdf_parameterized.c`: error ≤ 2.05e-16 ✓

**Performance**: ~4-5ms per PDF+gradient evaluation for simple models

**Documentation**: See `PHASE5_WEEK3_SOLUTION.md` for complete implementation details

**Python API (via trace)**:
```python
from phasic.trace_elimination import (
    record_elimination_trace,
    instantiate_from_trace
)

# 1. Build parameterized graph
graph = Graph(1)
# ... add parameterized edges ...

# 2. Record trace
trace = record_elimination_trace(graph, param_length=1)

# 3. Instantiate with concrete parameters
theta = np.array([2.0])
concrete_graph = instantiate_from_trace(trace, theta)

# 4. Compute PDF (gradient via finite differences)
pdf = concrete_graph.pdf(time=1.0, granularity=100)
```

**Key Insights**:
- Uniformization relates discrete jumps to continuous time: `dt = 1/λ`
- PMF is instantaneous absorption probability (with Poisson weighting)
- PDF = PMF / dt = PMF * λ (NOT PMF * granularity)
- Zeroing absorbed probability is critical to avoid cumulation

**Performance**:
- Single PDF+gradient evaluation: ~4-5ms for small models
- Suitable for gradient-based optimization (SVGD, HMC, etc.)

**Next Steps (Phase 5 continuation)**:
- JAX FFI integration for full autodiff (Week 4)
- Extend to reward-transformed graphs (Week 5)
- Benchmark on larger models (100+ vertices)

## Quick Reference

### Logging

**phasic** provides a unified logging system that integrates Python and C/C++ code logging into a single consistent interface.

**Default Behavior**: Logging is configured at WARNING level by default, so only important messages are shown unless explicitly enabled.

**Environment Variables**:
```bash
# Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
export PHASIC_LOG_LEVEL=DEBUG

# Write logs to file (in addition to console)
export PHASIC_LOG_FILE=/path/to/logfile.log

# Force colored output on/off (auto-detected by default)
export PHASIC_LOG_COLOR=1  # or 0 to disable
```

**Python API**:
```python
from phasic.logging_config import set_log_level, get_logger

# Enable debug logging for entire package
set_log_level('DEBUG')

# Enable debug logging for specific module
set_log_level('DEBUG', module='trace_elimination')

# Get logger for your module
logger = get_logger(__name__)
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

**Logger Hierarchy**:
- `phasic` - Root logger for entire package
- `phasic.c` - All C/C++ code logs appear here
- `phasic.module_name` - Module-specific loggers (e.g., `phasic.trace_elimination`)

**Examples**:
```python
# Example 1: Debug cache operations
from phasic.logging_config import set_log_level
set_log_level('DEBUG')

# Now you'll see detailed logs about:
# - Cache hits/misses
# - Hash computation
# - Trace serialization/deserialization
# - Graph operations

# Example 2: Silence all logging
from phasic.logging_config import disable_logging
disable_logging()

# Example 3: View only errors
set_log_level('ERROR')
```

**C Logging** (for developers):
```c
#include "phasic_log.h"

PTD_LOG_DEBUG("Processing vertex %d with rate %f", v_idx, rate);
PTD_LOG_INFO("Cache hit for hash %s", hash_hex);
PTD_LOG_WARNING("Parameter out of range: %d", param_idx);
PTD_LOG_ERROR("Failed to allocate memory for %zu bytes", size);
```

**Key Features**:
- Thread-safe logging from C code
- Automatic integration of C logs into Python logging hierarchy
- Colored console output (when terminal supports it)
- Zero overhead when logging is disabled
- File and console output simultaneously

**Implementation Details**:
- Python: `src/phasic/logging_config.py` - Unified logging configuration
- C API: `src/c/phasic_log.h/c` - Thread-safe C logging with callback mechanism
- Bridge: `src/cpp/phasic_pybind.cpp` - pybind11 bridge connecting C to Python logging
- Strategic logging in: `phasic_hash.c` (hash computation), `trace_cache.c` (cache operations)

### Disk caches

**phasic** maintains two on-disk caches under `~/.phasic_cache/`:

| Path | Contents | Populated by | Stage |
|---|---|---|---|
| `~/.phasic_cache/parameterized_reward_compute/<hash>.bin` | C-level symbolic elimination output (`parameterized_reward_compute_graph`) | `ptd_precompute_reward_compute_graph` on first call per (machine × model) | A2 |
| `~/.phasic_cache/traces/<hash>.{json,pkl}` | Python `EliminationTrace` objects | `record_elimination_trace` and the trace pipeline when `cache_trace=True` | pre-existing |

**Both caches are theta-independent** — keyed by graph
content hash (`ptd_graph_content_hash`, deterministic SHA-256 over
topology + coefficients), so the same parameterised model always
hits the same key regardless of theta history. Re-running SVGD
with new theta values does not produce new cache entries.

**Stage A2 specifically** persists the symbolic elimination across
processes so a fresh process (notebook restart, SLURM worker, CLI
run) can skip the O(n³) Gaussian elimination. **The cache uses the
rev-3 zero-copy format (magic `PTDPRMC3`, 2026-05):** the trace is
stored as offset/index-based commands that the loader `mmap`s
directly (`MAP_PRIVATE` copy-on-write) and the offset executor runs
in place — **no per-command pointer fixup, no copy**. This roughly
**halved the load CPU** vs the old rev-2 fixup format (two-locus
nr=7: load 1.095 s → 0.525 s, `scratch/io_overlap_probe.py`),
bringing it to ≈ the recompute cost.

Practical guidance (post-rev-3) — the decision hinges on **SCC size**,
because under `parallel_elimination=True` the cache competes with
*parallel* recompute:
- The per-SCC load is **I/O-bound and does not parallelize** (one
  disk), while recompute parallelizes across cores. So for models that
  decompose into **small SCCs**, parallel recompute *beats* the cache —
  measured ~11× faster at the plain two-locus nr8 (8407 states,
  maxSCC ≤ 102): warm load 5.0 s vs recompute 0.43 s
  (`scratch/overlap_bench.py`, controlled). Leave the cache **off** for
  these (the common sparse case; `profile_graph` flags it via the SCC
  structure).
- The cache pays only when **per-SCC recompute is expensive** = **large
  SCCs** (elimination is ~O(maxSCC³)). Migration / multi-population
  models reach this: the two-locus *ghost-island* model's base maxSCC
  grows 56 → 224 → 620 at nr_samples 3 → 5 (`scratch/base_scc_sweep.py`;
  the joint graph's maxSCC equals the base's). For such models, re-run
  across processes, the per-process load can beat re-eliminating.
  *(The exact maxSCC crossover, and a recompute-some / load-rest
  **hybrid** that overlaps parallel recompute with near-zero-CPU mmap
  loads, are an open follow-up — see `recompute-load-hybrid-plan.md`.)*
- It only ever affects the **first** call per process — the in-memory
  Stage-A1 persistent graph amortises the elimination across all later
  θ updates (the SVGD inner loop), so there is no per-θ benefit.
- For `parallel_elimination=True` the per-SCC cache (`scc_<hash>.bin`)
  uses the rev-3 zero-copy format and is the *only* cache (the monolith
  `<hash>.bin` is skipped); the distributed SLURM path
  (`precompute_distributed` → `scc_worker`) populates it. *(The
  EXTERNAL/`_ex` WP-3 path is dead/unused — superseded by WP-5
  edge-weight overrides — not a pending migration.)*
- Fallback (no silent fallback): if `mmap` is unavailable
  (`PHASIC_PCG_DISABLE_MMAP=1`, Windows, or a map failure) the loader
  logs and falls back to an explicit read+copy with identical results.
*(The pre-2026-05 "~6 ms → ~1 ms, 5× speedup" figure predates the
determinism fix; the rev-2 "net loss" assessment predates rev-3.)*

**Behaviour & opt-out**:
- Both caches honour `PHASIC_DISABLE_CACHE=1` to skip reads and writes.
- Format-version mismatches (header magic / `version` /
  `format_revision`) are treated as cache misses and trigger a
  rebuild that overwrites the bad file.
- Cache writes use atomic write-then-rename; multiple processes
  racing produce identical content.
- No automatic eviction. Users own the directory.

**Python API** (`src/phasic/cache.py`):
```python
import phasic.cache as cache

# Inspect
cache.param_compute_cache_info()
# {'cache_dir': '/home/.../parameterized_reward_compute', 'n_files': 12,
#  'total_size': 540288, 'disabled': False}

cache.trace_cache_info()
# {'cache_dir': '...', 'n_traces_json': 0, 'n_traces_pickle': 3, ...}

# Clear (returns number of files removed)
cache.clear_param_compute_cache()    # 12
cache.clear_trace_cache()            # 3
cache.clear_all_caches()             # {'param_compute': 0, 'traces': 0}

# Check if caching is disabled via env var
cache.is_cache_disabled()            # False
```

**C API**:
- `int ptd_save_parameterized_reward_compute_graph(const char *path, const struct ptd_desc_reward_compute_parameterized *compute, const struct ptd_graph *graph)` — declared in `api/c/phasic.h`. Atomic write of the symbolic elimination to disk. Returns 0 on success.
- `struct ptd_desc_reward_compute_parameterized *ptd_load_parameterized_reward_compute_graph(const char *path, const struct ptd_graph *graph)` — counterpart loader. Returns NULL on cache miss / corrupt file / version mismatch (caller falls back to rebuild).

**Implementation notes** (`src/c/phasic.c`):
- On-disk format starts with a 64-byte header (`PTDPRMC1` magic + `version` + `format_revision` + truncated graph hash + lengths). Bump `PTD_PCG_FORMAT_REVISION` whenever the layout changes; old caches will be detected as mismatched and silently overwritten.
- The compute graph contains pointers (`fromT`, `toT`, `multiplierptr`) into either the `mem` linked-list scratch buffer or live edge weights (`&edge->weight`). The save path encodes each pointer as `(kind, doubles_offset, vertex_idx, edge_idx, byte_offset_from_edge_weight)`; the load path resolves them against the loaded `mem` and the supplied graph's edges. A per-command-type liveness table avoids encoding fields the recorder doesn't initialise (e.g. `ZERO` only uses `fromT`).
- The cache hook is in `ptd_precompute_reward_compute_graph` (parameterised branch). It tries `ptd_load_...` first; on miss runs the elimination as before and writes back via `ptd_save_...`. Both directions gated by `PHASIC_DISABLE_CACHE`.

### Full API Documentation

- **C API**: See `api/c/phasic.h` (all C functions with comments)
- **C++ API**: See `api/cpp/phasiccpp.h` (object-oriented wrapper)
- **Python API**: Use `help(Graph)` or docstrings in code

### Key Files

**Core Implementation:**
- `src/c/phasic.c` - Core C algorithms
- `src/c/phasic_symbolic.c` - Symbolic expression system

**Python Modules:**
- `src/phasic/__init__.py` - Graph class, SVGD, main API
- `src/phasic/hex_grid.py` - HexGrid spatial grid utilities
- `src/phasic/trace_elimination.py` - Trace recording and evaluation (Phases 1-4)
- `src/phasic/ffi_wrappers.py` - JAX FFI integration (Phase 5 in progress)
- `src/phasic/svgd.py` - SVGD implementation
- `src/phasic/cache.py` - On-disk cache management (`~/.phasic_cache/`); see "Disk caches" section above

**Tests:**
- `tests/test_trace_recording.py` - Phase 1 tests
- `tests/test_trace_jax.py` - Phase 2 JAX integration tests
- `tests/test_trace_svgd_benchmark.py` - Phase 3 performance benchmarks
- `tests/test_trace_exact_likelihood.py` - Phase 4 exact likelihood tests

### Build and Install

```bash
# Development install
pip install -e .

# With JAX support
pip install -e .[jax]

# Using pixi (recommended)
pixi install
pixi run test
```

## Zero-inflation and AIC/BIC

`SVGD.log_likelihood()` now returns the **full** log-likelihood used
by the optimiser, including the zero-inflation point-mass term
`Σ_j n_zero_j · log(1 − p_j(θ))` when `Graph.svgd` attaches
`_zero_inflated_p_fn` to the model (partial-coverage rewards).
Previously this term was silently dropped from the public
`log_likelihood()` path, so any saved AIC/BIC numbers from
partial-coverage fits were off by `Σ_j n_zero_j · log(1 − p_j(θ̂))`
and must be re-computed.

Implementation: `SVGD._log_lik_from_pmf` (`src/phasic/svgd.py`) is the
single source of truth for the full LL summation; both
`_log_prob_unified` (SVGD optimisation) and `log_likelihood()`
(AIC/BIC/LRT) delegate to it.

*Last updated: 2026-05-16*
- When creating a markdown file summarizing changes made, please prompt to add changed and new files to git, commit them with a message from the markdown file, but do not add the markdown file itself. Prompt only once and then do git add, commit, and push without prompting further.
