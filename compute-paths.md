# phasic compute paths: PDF / PMF / moments

A working reference for which entry point computes what, the layered
call path each one takes, and which paths are live, fallback,
deprecated, or dead. Reflects the source tree as of this commit.

## Decision tree

Use this to pick the right entry point. Each leaf names the public
Python surface; subsequent sections describe the call stack.

```
Need a quantity from a phase-type / phase-type-with-rewards
distribution?
│
├─ Eager (no JAX, no gradients) — just compute the number?
│  │
│  ├─ Moments (E[T^k], variance, covariance, expectation)
│  │  └─ Graph.expectation / Graph.moments / Graph.variance / Graph.covariance
│  │     → C++ ptd_expected_waiting_time + reward chaining
│  │
│  ├─ PDF / PMF (continuous f(t) or discrete p(n))
│  │  └─ Graph.pdf  (auto-dispatches to discrete based on is_discrete)
│  │     → C++ uniformization (forward algorithm, Algorithm 4)
│  │
│  ├─ CDF
│  │  └─ Graph.cdf  → C++ pdf-via-uniformization, complement
│  │
│  ├─ Stop probabilities at time t (mass at each vertex)
│  │  └─ Graph.stop_probability  → C++ uniformization step loop
│  │
│  ├─ Sojourn times (E[time at vertex i] before absorption)
│  │  └─ Graph.expected_sojourn_time
│  │     → C++ ptd_expected_sojourn_time / _subset
│  │
│  └─ Joint probabilities (when graph is a joint_prob_graph)
│     └─ Graph.joint_prob_table  → C++ sojourn-time normalization
│
├─ JAX-traceable (jit/grad/vmap; for SVGD, MCMC, optimisation)?
│  │
│  ├─ Joint-prob graph (population genetics use case)
│  │  │
│  │  ├─ Single time-homogeneous epoch
│  │  │  └─ Graph.pmf_from_graph_joint_index
│  │  │     → ffi_wrappers.compute_sojourn_times_ffi
│  │  │     → C++ ComputeSojournTimesFfiImpl (XLA FFI, no callbacks)
│  │  │
│  │  └─ Daisy-chained (piecewise-constant time-inhomogeneous)
│  │     └─ Graph.daisy_chain_joint_probs
│  │     → ffi_wrappers.compute_daisy_chain_joint_probs_ffi
│  │     → C++ DaisyChainJointProbsFfiImpl (XLA FFI, no callbacks)
│  │
│  ├─ Standard parameterised graph, PDF/PMF only
│  │  └─ Graph.pmf_from_graph
│  │     → ffi_wrappers.compute_pmf_ffi
│  │     → C++ ComputePmfFfiImpl
│  │
│  ├─ Standard parameterised graph, moments only
│  │  └─ Graph.moments_from_graph
│  │     → ffi_wrappers.compute_moments_ffi (when use_ffi=True)
│  │
│  ├─ Standard parameterised graph, PDF/PMF + moments together
│  │  └─ Graph.pmf_and_moments_from_graph
│  │     → ffi_wrappers.compute_pmf_and_moments_ffi (when use_ffi=True)
│  │     → fallback path (use_ffi=False, the default) uses
│  │       jax.pure_callback + pybind11 GraphBuilder; live primary
│  │       path for SVGD-with-rewards
│  │
│  ├─ Multivariate (2D rewards: per-feature reward vectors)
│  │  └─ Graph.pmf_and_moments_from_graph_multivariate
│  │     → ffi_wrappers.compute_pmf_multivariate_ffi (when use_ffi=True)
│  │
│  ├─ User-supplied Python graph builder (graph structure depends on theta)
│  │  └─ Graph.pmf_from_graph_parameterized
│  │     → compiled C++ wrapper + ctypes + jax.pure_callback
│  │
│  └─ User-supplied .cpp file
│     └─ Graph.pmf_from_cpp
│        → compile user .cpp + ctypes + jax.pure_callback
│
└─ SVGD inference?
   └─ Graph.svgd(observed_data, ...)
      → dispatches to one of the JAX-traceable models above
        based on (joint_prob_graph?, rewards?, epoch_starts?, …)
```

## Layered architecture

The library is four layers thick. Top to bottom:

1. **Public Python methods on `Graph`** (`src/phasic/__init__.py`).
   Eager methods (`expectation`, `pdf`, `stop_probability`, …)
   call directly into the pybind11 C++ wrapper. JAX-traceable
   classmethod *factories* (`pmf_from_graph`, …) build a closure
   that calls layer 2 and wraps it in a `custom_vjp` for gradients.
2. **JAX wrappers** (`src/phasic/ffi_wrappers.py`). Each
   `compute_*_ffi` function calls `jax.ffi.ffi_call("ptd_…")`,
   passing graph topology as a static `structure_json` attribute
   and parameter buffers (theta, times, etc.) as XLA buffers.
3. **C++ FFI handlers** (`src/cpp/parameterized/graph_builder_ffi.cpp`).
   Each `*FfiImpl` is registered as an XLA custom call. Inside, an
   OpenMP-parallel vmap loop builds a `phasic::Graph` from the JSON
   (cached in a thread-local `builder_cache`) and calls into…
4. **C functions** (`src/c/phasic.c`). The actual numerics:
   `ptd_expected_waiting_time`, `ptd_expected_sojourn_time_subset`,
   `ptd_probability_distribution_step` (uniformization),
   `ptd_graph_pdf_with_gradient`, etc.

The pybind11 layer (`src/cpp/phasiccpp.cpp`) is a thin
class-style wrapper that connects `phasic::Graph` (C++) to its
underlying `ptd_graph` (C). Eager methods on the Python `Graph`
class call into pybind11 methods, which call C functions directly
— no XLA FFI involved. Layer 2 (XLA FFI) only matters for
JAX-traceable paths.

## Path-by-path summary

### Live primary paths

#### `Graph.expectation`, `Graph.moments`, `Graph.variance`, `Graph.covariance`
Eager (no JAX). Calls `super().moments(power, rewards)` which
delegates to the C++ `_moments` helper in `phasic_pybind.cpp:212`,
which iterates `expected_waiting_time` calls. For parameterised
graphs the underlying C uses the cached
`parameterized_reward_compute_graph` (Stage A0): O(n³) symbolic
elimination once, O(n) replay per `theta`. Used by user code that
doesn't need gradients (interactive analysis, sampling reference,
non-SVGD calculations).

#### `Graph.pdf` / `Graph.dph_pmf` / `Graph.cdf`
Eager. C++ uniformization (Algorithm 4): construct a
`ptd_probability_distribution_context`, step it forward to the
requested time, read absorbed mass. Auto-dispatches to discrete
flavour when `self.is_discrete`. The context is cached per-Graph
and invalidated by `weight_version` bumps from
`update_weights` / `update_ipv`. Used directly for visualisation
and spot-checks; also the inner kernel of every JAX wrapper that
needs PDF/PMF values.

#### `Graph.stop_probability`
Eager. Same uniformization context as `Graph.pdf`, but returns the
full per-vertex probability vector at time t rather than absorbed
mass. The daisy-chain FFI handler calls this internally between
epochs (and at the final `t_eval`). Cycle-safe.

#### `Graph.expected_sojourn_time` (and `_subset`)
Eager. C++ `ptd_expected_sojourn_time` / `_subset`. Computes
E[time at vertex i] for all vertices or a subset. The subset
variant is what `compute_sojourn_times_ffi` calls under the hood
(it's how `pmf_from_graph_joint_index` produces per-observation
joint probabilities for SVGD).

#### `Graph.pmf_from_graph`
JAX-traceable classmethod for parameterised graphs. Returns
`(theta, times) → pdf`. Inside: serialises the graph to JSON,
calls `compute_pmf_ffi(structure_json, theta, times)`, wraps in
`custom_vjp` with finite-difference backward (`eps=1e-7`). FFI is
default-on (`config.ffi=True`); the legacy non-FFI fallback at
`__init__.py:4404+` is **disabled** (raises `PTDBackendError` —
see "Disabled fallback" below).

#### `Graph.pmf_from_graph_joint_index`
JAX-traceable classmethod, used by SVGD on joint-prob graphs.
Returns `(theta, vertex_indices, rewards=None) →
(per_obs_probs, dummy_moments)`. Inside: calls
`compute_sojourn_times_ffi` on the observed vertex subset, then
on all terminal vertices for normalisation, returns
`sojourn_times[obs] / sum(sojourn_times[all_terminals])`. The
sojourn-times-as-joint-probs identity holds because of the
joint_prob_graph's "trash + terminal absorbing" topology. Custom
VJP with FD backward; honours `fixed_mask` to skip FD on pinned
parameters.

#### `Graph.pmf_and_moments_from_graph`
JAX-traceable classmethod. Returns
`(theta, times, rewards=None) → (pmf, moments)`. Two real branches:
- **Callback weight_mode** (when graph has a Python `weight_callback`):
  uses `jax.pure_callback` to call back into Python for weight
  evaluation. Primary path in this mode (no FFI alternative).
- **Standard mode**, `use_ffi=True`: routes through
  `compute_pmf_and_moments_ffi` — OpenMP-parallel XLA FFI, fast.
- **Standard mode**, `use_ffi=False` (the **method-signature default**):
  pybind11 GraphBuilder + `jax.pure_callback`. Slower (single-core,
  Python boundary per call) but *primary path* in default usage.
  Specifically: SVGD-with-rewards calls this without setting
  `use_ffi`, so it gets the fallback. **Not currently disabled.**
  See "Live `pure_callback` paths" below.

#### `Graph.pmf_and_moments_from_graph_multivariate`
Same shape as above but for 2D rewards (per-feature reward
vectors). FFI version is `compute_pmf_multivariate_ffi`. Same
`use_ffi=False` default → same `pure_callback` fallback as
`pmf_and_moments_from_graph` in default usage.

#### `Graph.daisy_chain_joint_probs`
JAX-traceable. Returns
`(epoch_thetas, epoch_dts, initial_ipv) → joint_probs[t-vertex]`.
Internally: builds `_daisy_chain` metadata (epoch_dts, t_eval,
ipv_target_indices, t_aux_keys, t_aux_values, t_vertex_indices)
and embeds it into the augmented `structure_json`, then calls
`compute_daisy_chain_joint_probs_ffi`. Custom VJP with FD
backward. Used by `Graph.svgd(epoch_starts=...)` for piecewise-
constant time-inhomogeneous joint-prob inference.

The C++ handler `DaisyChainJointProbsFfiImpl` does the entire
chain in C: `update_ipv → update_weights → stop_probability(dt)`
per transition, collapse t-aux pairs, project to IPV-coords,
final `stop_probability(t_eval)` and slice to t-vertex
positions. One Python↔C boundary crossing per forward.

### Live `pure_callback` paths (no FFI alternative)

These three classmethods unavoidably use `jax.pure_callback` because
the work isn't expressible as a fixed FFI signature.

#### `Graph.pmf_from_graph` callback weight_mode (`__init__.py:4308+`)
When the user sets `graph.weight_callback`, edge weights are
computed by a Python callable that XLA can't see. The model wraps
the callable in a `pure_callback`. Primary path for `weight_mode='callback'`.

#### `Graph.pmf_from_graph_parameterized` (`__init__.py:4529+`)
When the graph topology itself depends on `theta` — i.e. user
provides a Python `theta → Graph` builder — the model compiles a
C++ wrapper at runtime, loads it via ctypes, and wraps the call in
`pure_callback`. Used by power users who need parameter-dependent
graph structure (rare).

#### `Graph.pmf_from_cpp` (`__init__.py:4689+`)
User writes a `.cpp` file implementing `build_model(theta, n)`,
phasic compiles + loads it via ctypes + `pure_callback`. Used for
performance-critical models where the user wants raw C++ but still
JAX-compatible. Same machinery as `pmf_from_graph_parameterized`,
different entry point.

#### `Graph.pmf_and_moments_from_graph` non-FFI fallback (`__init__.py:6325+`)
The "fallback" branch when `use_ffi=False`. Listed under primary
paths above because the **default `use_ffi=False`** signature
makes this the *de facto* primary path for SVGD-with-rewards. Not
disabled. Annotated with a `NOTE` in source.

### Disabled fallback (commented out, raises if reached)

#### `Graph.pmf_from_graph` non-FFI fallback (`__init__.py:4404+`)
Was the `pure_callback` + pybind11 fallback for environments where
FFI is unavailable. Replaced by `raise PTDBackendError(...)`; the
legacy body kept in source as commented reference. Only reachable
if user sets `phasic.configure(ffi=False)`. Default is `ffi=True`.

### Deprecated paths

#### `Graph.compute_trace`, `Graph._ensure_trace`, `Graph._moments_from_trace`, `Graph._expectation_from_trace`, `Graph._variance_from_trace`
The Python `EliminationTrace` machinery
(`src/phasic/trace_elimination.py` +
`src/phasic/hierarchical_trace_cache.py`). Never JAX-traceable;
records and replays the elimination as a list of operations in
Python. Used to gate `moments()`/`expectation()`/`variance()` on
parameterised graphs through `_ensure_trace` → `instantiate_from_trace`
→ `concrete_graph.moments()`.

**Now disabled at the entry-point level.** `Graph.moments`,
`Graph.expectation`, and `Graph.variance` route directly to
C++ `super().moments()` etc., skipping the Python trace
recording entirely. The trace branches are commented out;
`_moments_from_trace` etc. are still in source for direct callers.
`Graph.compute_trace()` still works but emits a `DeprecationWarning`.
`Graph(cache_trace=True)` and `Graph(cache_trace=hierarchical=...)`
emit a `DeprecationWarning` and force the flag to `False`.

#### `phasic.cache.clear_trace_cache`, `trace_cache_info`
On-disk trace cache (`~/.phasic_cache/traces/`). Nothing in the
public API populates it anymore; helpers retained to clean up
leftover files from prior phasic versions. Both emit
`DeprecationWarning`. `phasic.cache.clear_all_caches` calls
`clear_trace_cache` internally, so it inherits the warning.

`phasic.cache.clear_param_compute_cache` (the C-level Stage A2
cache at `~/.phasic_cache/parameterized_reward_compute/`) is
**not** deprecated — that cache is populated by every C-path
forward call.

#### `Graph.daisy_chain_log_likelihood_joint_snapshot`
Removed entirely (was a thin wrapper that summed log-likelihoods
externally; SVGD's model contract already does its own log-sum,
no callers anywhere).

### Removed (no source remains)

These were removed during the daisy-chain FFI consolidation:

- `Graph.epoch_transition_fn` — per-epoch JAX transition; replaced
  by the single-FFI-call architecture.
- `Graph._collapse_t_aux` — collapsing now done in C++ inside the
  daisy-chain handler.
- `Graph.joint_stop_probabilities`, `Graph.joint_probs_with_time` —
  eager numpy helpers; nothing called them.
- `Graph.joint_snapshot_indexer` — outcome→index lookup;
  `_daisy_chain_svgd_model` builds its own equivalent dict.
- `Graph._converged_joint_probs`, `Graph._collapsed_position` —
  convergence loop superseded by single fixed `t_eval`.
- `_create_jax_callback_wrapper` (`__init__.py:542+`) — was a
  factory; never had any callers.
- `Graph._ipv_collapsed_positions` attribute on JSP graph — was
  used only by `epoch_transition_fn`.

## SVGD model dispatch

`Graph.svgd(observed_data, ...)` chooses one of the JAX-traceable
factories above based on the call shape. The dispatch is at
`__init__.py:5450+`:

| Condition | Model used |
|---|---|
| Joint-prob graph + `epoch_starts is None` | `pmf_from_graph_joint_index` (sojourn-times) |
| Joint-prob graph + `epoch_starts is not None` | `_daisy_chain_svgd_model` → `daisy_chain_joint_probs` |
| Non-joint-prob graph + 2D rewards | `pmf_and_moments_from_graph_multivariate` |
| Non-joint-prob graph + 1D rewards (or no rewards) | `pmf_and_moments_from_graph` |

The daisy-chain branch builds a JSP graph from `self`, fits a
time-homogeneous reference prior via `probability_matching` on
`self`, broadcasts that prior across the `n_epochs` epoch slots,
and constructs a model closure that calls `daisy_chain_joint_probs`.

## C-layer reference

The numerics live in `src/c/phasic.c`. Most relevant entry points:

| Function | What | Used by |
|---|---|---|
| `ptd_graph_update_weights` | Set non-IPV edge weights from `theta` | All paths after each parameter update |
| `ptd_graph_update_ipv` | Set IPV edge weights from `ipv` vector | Daisy-chain transitions |
| `ptd_expected_waiting_time` | E[T] via reward-compute graph | `_moments` (eager `Graph.moments`) |
| `ptd_expected_sojourn_time_subset` | E[time at subset of vertices] | `compute_sojourn_times_ffi` |
| `ptd_probability_distribution_context_create` + `_step` | Uniformization forward iteration | `Graph.pdf`, `Graph.stop_probability`, daisy-chain handler |
| `ptd_graph_pdf_with_gradient` | PDF + analytic gradient (Phase 5 Week 3) | C-level only; not yet wired to public API |
| `ptd_precompute_reward_compute_graph` | O(n³) symbolic elimination → cached | All parameterised eager moment / waiting-time calls |
| `ptd_graph_clone` | Clone graph (with weight copy) | `reward_transform`, daisy-chain `_daisy_chain_svgd_model` |

The `parameterized_reward_compute_graph` (Stage A0) — the symbolic
output of `ptd_precompute_reward_compute_graph` — survives
`update_weights`/`update_ipv` calls because the recorded commands
hold pointers (`multiplierptr`) into live edge-weight slots, so
weight changes propagate without re-elimination. Stage A1
(in-memory persistent `phasic::Graph` per `GraphBuilder`) and
Stage A2 (on-disk persistent symbolic compute graph at
`~/.phasic_cache/parameterized_reward_compute/`) extend the same
cache to cross thread/process boundaries.

## Custom-VJP gradient pattern

Every JAX-traceable factory wraps its forward in a `jax.custom_vjp`
with a finite-difference backward at `eps=1e-7`. The pattern:

```
@jax.custom_vjp
def model(theta, ...): return forward(theta, ...)
def model_fwd(theta, ...): return forward(theta, ...), saved_inputs
def model_bwd(saved, g):
    grads = []
    for i in range(theta.shape[0]):
        if i in fixed_indices: grads.append(0.0); continue
        f_plus  = forward(theta + eps·e_i, ...)
        f_minus = forward(theta - eps·e_i, ...)
        grads.append(jnp.sum(g * (f_plus - f_minus) / (2·eps)))
    return jnp.stack(grads)
model.defvjp(model_fwd, model_bwd)
```

This means each `jax.grad` call costs `(n_params - n_fixed) × 2`
forward FFI calls. Currently no factory uses analytic gradients,
even though `ptd_graph_pdf_with_gradient` exists at the C level.
Wiring that up is on the long-term roadmap (Phase 5 Week 4).

## On-disk caches summary

| Path | Layer | Lifetime | Cleared by |
|---|---|---|---|
| `~/.phasic_cache/parameterized_reward_compute/` | Stage A2 (live) | persistent, content-hashed | `phasic.cache.clear_param_compute_cache` |
| `~/.phasic_cache/traces/` | Python EliminationTrace (deprecated) | persistent, content-hashed | `phasic.cache.clear_trace_cache` (warns) |
| `~/.phasic_cache/graphs/` | GraphCache (live) | persistent, content-hashed | `phasic.graph_cache.GraphCache.clear_graph_cache` |
| `~/.jax_cache/` | JAX compilation cache (live) | persistent | `phasic.clear_jax_cache` |

Stage A0 (in-process symbolic compute graph) and Stage A1
(thread-local persistent `phasic::Graph`) are in-memory only.
