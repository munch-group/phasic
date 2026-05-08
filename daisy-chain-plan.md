# Plan: Daisy-chained likelihood for time-inhomogeneous joint-prob models — C path

## Implementation status

Phases 1, 2, and 3 (joint-snapshot variant) **landed** in
src/phasic/__init__.py with regression tests in
tests/pytest/test_daisy_chain_c_path.py (20 cases passing, 1 skipped
for jax.grad — see deferred work below).

Deferred to a follow-up:

- **`daisy_chain_log_likelihood_per_event`** — the plan's per-event
  variant. The notebook `time_inhom_joint_prob.ipynb` does not have a
  reference per-event computation, so the semantic ("which graph
  evaluates the PMF — JSP or joint-prob?") is ambiguous. Pick this up
  when there is a concrete reference to bit-compare against. The
  joint-snapshot variant covers the user's actual workflow.
- **`jax.grad` through the daisy chain** — the underlying
  `pure_callback`s have no JVP rule, so `jax.grad` over `epoch_thetas`
  raises `Pure callbacks do not support JVP`. The fix is a custom
  `jax.custom_vjp` that computes gradients via finite differences on
  the boundary, mirroring the pattern at `__init__.py:4322-4332`. The
  test `test_grad` skips with this rationale; once the VJP lands, the
  skip flips to a finite-difference check.
- **Phase 4 SVGD smoke test** — depends on `jax.grad`. Deferred with
  the gradient work.

## Status & currency

**Last reconciled with the codebase: 2026-05-05** (commit `69a5857`,
post-Stages 1/2/3/A0/A1/A2 + `phasic.cache` module + tutorial
notebooks).

**Architecture pivot (2026-05-05)**: previous revisions of this plan
extended the Python `EliminationTrace` path via codegen
(`_generate_cpp_stop_prob_from_trace`,
`_wrap_trace_stop_prob_for_jax`). That path inherits the Python
recorder's cyclic-graph rejection, and joint-prob graphs are always
cyclic. This rewrite drops the codegen route entirely and builds the
daisy chain on the C path — `Graph.update_ipv` (from the
prerequisite plan) plus `Graph.stop_probability` (already C-side,
already cycle-safe via uniformization at `phasic.c:9536-9650`).

Greenfield additions verified by grep: zero hits for
`joint_stop_prob_graph`, `daisy_chain_log_likelihood`,
`epoch_transition_fn` in `src/`, `tests/`.

## Hard prerequisites — and what they block

This plan has **one** hard prerequisite:

1. **`update_ipv-plan.md` (all five batches)** must land — it
   provides `Graph.update_ipv(ipv)` and the C-side
   `ptd_graph_update_ipv` that this plan calls between epochs.

The cyclic-graph blocker that gated the previous version of this
plan is **gone**: the C `parameterized_reward_compute_graph`
machinery handles cycles correctly via the self-multiply trick at
`phasic.c:5305+`, and the `ptd_probability_distribution_context`
(uniformization) at `phasic.c:9536+` is fundamentally cycle-safe.
`Graph.joint_prob_graph(...).expectation()` already works on cyclic
joint-prob graphs today through this same C path.

## Goal

Provide a JAX-compatible `Graph.daisy_chain_log_likelihood_*` API
that computes the data likelihood under a piecewise-constant
time-inhomogeneous joint-probability model. The user's eager loop
today (`docs/pages/tutorial/time_inhom_joint_prob.ipynb`):

```python
for i in range(n_epochs - 1):
    cloned = clone_with_ipv(joint_kernel, epoch_ipv[i])
    jsp_graph, t_map = joint_stop_prob_graph(cloned)
    jsp_graph.update_weights(epoch_theta[i])
    epoch_ipv[i+1] = joint_stop_probabilities(jsp_graph, epoch_starts[i+1], t_map)

log_lik = ... function of joint_probs_with_time(jsp_graph_final, ...) ...
```

becomes:

```python
jsp_graph = joint_kernel.joint_stop_prob_graph()  # built once
INITIAL_IPV = ...  # set once before SVGD; not optimised

def daisy_chain(epoch_thetas):
    ipv = INITIAL_IPV
    for i in range(n_epochs - 1):
        jsp_graph.update_ipv(ipv)
        jsp_graph.update_weights(epoch_thetas[i])
        ipv = jsp_graph.joint_stop_probabilities(epoch_dts[i])
    jsp_graph.update_ipv(ipv)
    jsp_graph.update_weights(epoch_thetas[-1])
    return log_lik_from_final_epoch(jsp_graph, observed_times)
```

`epoch_thetas` is the SVGD parameter (one rate vector per epoch).
`INITIAL_IPV` is fixed by the user before SVGD starts. The
intermediate `ipv` values that flow between epochs are computed by
the daisy-chain loop itself (each is the survival distribution from
the previous epoch's `stop_probability`); they are
library-internal, not optimised.

JAX traceability comes from threading `update_ipv` and
`update_weights` between epochs — both will be wrapped as
`jax.pure_callback` boundaries with `vmap_method='sequential'` (same
pattern as the existing FFI wrappers in
`src/phasic/ffi_wrappers.py`).

### User-confirmed constraints (carried over from previous plan)

- Epoch boundaries `epoch_starts` are **fixed** (not inferred). Each
  epoch's `dt` is a static Python float.
- Initial epoch IPV is **fixed** at the model's defined IPV. Not an
  SVGD parameter.
- Number of epochs typically **<30**. Unrolling the daisy chain is
  fine; `lax.scan` is overkill.
- `t_aux_map` may be inferred from graph structure (preferred) — see
  `joint_stop_prob_graph` design below.

### Why not `add_epoch`?

`Graph.add_epoch(time, ...)` (`__init__.py:3086+`) is the existing
"build one big multi-epoch graph" pattern. It wires sister-vertex
transitions weighted by `stop_probability/accumulated_occupancy`,
correct for standard phase-type epoch composition. **It does not
preserve the joint-distribution-over-rewards** needed for joint-prob
inference (the reward-extended state and the t-vertex
absorption-mass interpretation). The user genuinely needs the daisy
chain pattern for the joint-prob case. `add_epoch` continues to be
the right answer for non-joint cases.

## What's already in the codebase (reusable)

- **`Graph.update_ipv(ipv)`** — provided by `update_ipv-plan.md`.
  Updates starting-vertex edge weights at runtime; symbolic compute
  graph cache survives.
- **`Graph.update_weights(theta)`** at `__init__.py:2132-2173`.
  Updates non-IPV parameterised edge weights at runtime; symbolic
  compute graph cache survives (Stage A0 invariant at
  `phasic.c:4280-4304`).
- **`Graph.stop_probability(time, granularity=0)`** at
  `__init__.py:2768-2794`, C++ wrapper at `phasiccpp.h:764-796`,
  pybind binding at `phasic_pybind.cpp:2704`. Backed by the C
  uniformization context (`ptd_probability_distribution_context_*`
  at `phasic.c:9536-9650`). The C++ wrapper already caches the
  context across calls and invalidates correctly when
  `weight_version` changes (so each `update_ipv` /
  `update_weights` call cleanly rebuilds the per-call context but
  reuses the symbolic compute graph). Cycle-safe.
- **`Graph.joint_prob_graph(...)`** at `__init__.py:7281-7558`.
  Builds the reward-extended joint kernel with trash pair
  (`__init__.py:7484-7487`) and t-vertices. Sets
  `_joint_prob_base_graph_indexer` and `_rewarded_props` on the
  result. **Caveat**: `joint_prob_graph` does NOT propagate
  `_cache_trace`; the new `joint_stop_prob_graph` must set
  `new._cache_trace = self._cache_trace` explicitly.
- **The notebook helper `joint_stop_prob_graph`** in
  `docs/pages/tutorial/time_inhom_joint_prob.ipynb`. Structural
  transformation that needs to be promoted into the library
  (Phase 1 below).
- **`phasic.cache.param_compute_cache_info` /
  `clear_param_compute_cache`** — for users who want to verify the
  symbolic cache survived a full daisy-chain sweep.

## Design decision: no silent fallbacks

Per the user's explicit preference, the API exposes **two distinct
entry points** for the two distinct likelihood patterns rather than
one method that switches behaviour on an optional argument. Each
has a fully required, fully explicit signature.

- `daisy_chain_log_likelihood_per_event` — observed times are
  per-event (one event = one absorption); likelihood is `Σ log
  p(t_event | epoch)` summed over events grouped by epoch.
- `daisy_chain_log_likelihood_joint_snapshot` — observed data are
  joint snapshots at known times; likelihood is read from joint
  probability mass at each observation time.

---

# Phase 1 — `joint_stop_prob_graph` and helpers

Insertion point: `src/phasic/__init__.py` immediately after
`Graph.joint_prob_graph` (currently ends near line 7558; the next
`def` is `_get_joint_probs` at line 7561).

### `Graph.joint_stop_prob_graph(self) -> Graph`

Library version of the notebook helper. Differences from notebook:

1. Does **not** call `clone_with_ipv` internally. With
   `update_ipv` available, the JSP graph is built **once** off the
   base joint-prob graph and the IPV is set per epoch via
   `update_ipv`.
2. Detects t-vertices structurally: vertex `v` is a t-vertex iff any
   outgoing edge leads to an absorbing vertex
   (`len(edge.to().edges()) == 0`).
3. Detects the trash pair via the notebook's `is_trash` predicate
   (state-zero self-loop pair).
4. For each t-vertex, replaces its outgoing edges with a "trapping
   aux loop": creates an aux vertex with state-zero, adds
   bidirectional unit-weight parameterised edges. Mass that reaches
   a t-vertex shuttles between it and its aux indefinitely, so
   `stop_probability(t)[t_vertex] + stop_probability(t)[aux]` equals
   the cumulative joint absorption mass at that t-state by time `t`.
5. Sets attributes on the returned graph:

   ```python
   new._joint_prob_base_graph_indexer = self._joint_prob_base_graph_indexer
   new._rewarded_props = self._rewarded_props
   new._joint_stop_prob_graph = True
   new._t_vertex_indices = sorted(t_vertex_indices)
   new._t_aux_map = t_aux_vertex_indices
   new.is_discrete = self.is_discrete
   new._cache_trace = self._cache_trace  # explicit propagation
   ```

6. Validation: raises `ValueError` if
   `not self._joint_prob_base_graph_indexer` or
   `self.param_length() == 0`.

`_t_aux_map` and `_t_vertex_indices` are kept in **new-graph index
space**.

### `Graph.joint_stop_probabilities(self, t) -> np.ndarray`

Eager numpy version. Calls `self.stop_probability(t)` and applies
`_collapse_t_aux` using `self._t_aux_map`:

```python
def joint_stop_probabilities(self, t):
    if not getattr(self, '_joint_stop_prob_graph', False):
        raise ValueError(
            ".joint_stop_probabilities requires graph from "
            ".joint_stop_prob_graph()"
        )
    raw = np.asarray(self.stop_probability(t))
    return self._collapse_t_aux(raw)

def _collapse_t_aux(self, raw_vec):
    n = self.vertices_length()
    aux_set = set(self._t_aux_map.values())
    out = []
    for i in range(n):
        if i in aux_set:
            continue
        p = raw_vec[i]
        if i in self._t_aux_map:
            p = p + raw_vec[self._t_aux_map[i]]
        out.append(p)
    return np.asarray(out)
```

Output length: `vertices_length() - n_aux`, in non-aux vertex order.

### `Graph.joint_probs_with_time(self, t) -> pd.DataFrame`

Display helper. Uses `self._t_vertex_indices` to slice the collapsed
vector and builds a DataFrame of (state, joint_prob) rows.

**Tests** in `tests/pytest/test_daisy_chain_c_path.py`:

- `test_joint_stop_prob_graph_structure`: build a small joint-prob
  graph, transform, assert each t-vertex has exactly one aux
  partner, t-aux edges are unit-weight, trash-pair preserved.
- `test_joint_stop_probabilities_matches_notebook`: build the same
  3-epoch model used in `time_inhom_joint_prob.ipynb` and
  bit-compare `joint_stop_probabilities(t)` against the notebook's
  eager loop output.
- `test_joint_stop_prob_graph_cyclic`: build a joint-prob graph that
  has cycles in its reward-extended state space, transform, call
  `stop_probability(1.0)` — must succeed (cycle-safe via C
  uniformization).

**Test gate**: all three pass; existing
`tests/pytest/test_modeling_compose.py` still passes.

---

# Phase 2 — `Graph.epoch_transition_fn` and the daisy-chain loop

### `Graph.epoch_transition_fn(self, dt: float) -> Callable`

Returns a callable `(theta, ipv) -> ipv_next` that:

1. `self.update_ipv(ipv)` — sets starting-vertex edge weights.
2. `self.update_weights(theta)` — sets non-IPV edge weights.
3. `raw = self.stop_probability(dt)` — C uniformization, cycle-safe.
4. `ipv_next = self._collapse_t_aux(raw)` — returns surviving mass
   collapsed over aux vertices.

The returned callable is wrapped via `jax.pure_callback` with
`vmap_method='sequential'` so it composes inside `jax.jit` and
tolerates `jax.vmap` over particles. The wrapper signature:

```python
def transition(theta, ipv):
    result_shape = jax.ShapeDtypeStruct((n_non_aux,), jnp.float64)
    return jax.pure_callback(
        lambda t_np, i_np: _transition_impl(self, dt, t_np, i_np),
        result_shape, theta, ipv,
        vmap_method='sequential',
    )
```

**Tests**:

- `test_epoch_transition_fn_eager`: build JSP graph, call
  `transition_fn(theta, ipv)` directly (no JAX), compare to manually
  doing `update_ipv → update_weights → stop_probability → collapse`.
- `test_epoch_transition_fn_jit`: wrap in `jax.jit`, run the same
  inputs, expect bit-identical output.
- `test_epoch_transition_fn_vmap`: vmap over a (10,) batch of `theta`
  particles with shared `ipv`; expect (10, n_non_aux) output that
  matches a Python-loop reference.

**Test gate**: all three pass.

---

# Phase 3 — Two daisy-chain log-likelihood methods

Both methods iterate epochs in pure Python (no `lax.scan`). Each
iteration calls the per-epoch transition fn from Phase 2, threading
IPV forward.

### `Graph.daisy_chain_log_likelihood_per_event(...)` 

Signature:

```python
def daisy_chain_log_likelihood_per_event(
    self,
    *,
    epoch_dts: list[float],          # length n_epochs, static
    epoch_thetas: jnp.ndarray,       # shape (n_epochs, theta_dim) — SVGD parameter
    epoch_events: list[jnp.ndarray], # length n_epochs; epoch_events[i] is times in epoch i
    initial_ipv: np.ndarray,         # shape (n_non_aux,) — fixed before SVGD; not traced
    granularity: int = 100,
) -> jnp.ndarray:                    # scalar log-lik
    """Per-event likelihood: Σ_i Σ_e log p(t_e | epoch i, ipv_i, theta_i).

    `initial_ipv` is the IPV for epoch 0. Subsequent epoch IPVs are
    computed internally by the daisy chain (each is the survival
    distribution of the previous epoch's stop_probability) and are
    not user-visible.

    `initial_ipv` is **not** an SVGD-optimised parameter; the user
    pins it before calling SVGD. Inside the daisy chain it enters
    the `pure_callback` boundary as a constant.
    """
```

Body (sketch):

```python
ipv = initial_ipv
log_lik = 0.0
n_epochs = len(epoch_dts)
transition = self.epoch_transition_fn  # cached factory

for i in range(n_epochs - 1):
    # PMF for events in this epoch given current IPV+theta.
    pmf_at_events = self._epoch_event_pmf(
        epoch_thetas[i], ipv, epoch_events[i], granularity
    )
    log_lik = log_lik + jnp.sum(jnp.log(pmf_at_events))

    # Advance to next epoch.
    ipv = transition(epoch_dts[i])(epoch_thetas[i], ipv)

# Last epoch: same PMF treatment, no transition out.
pmf_last = self._epoch_event_pmf(
    epoch_thetas[-1], ipv, epoch_events[-1], granularity
)
log_lik = log_lik + jnp.sum(jnp.log(pmf_last))
return log_lik
```

`_epoch_event_pmf` is a thin wrapper that calls
`self.update_ipv(ipv); self.update_weights(theta);
self.compute_pmf(events, granularity)` inside a `pure_callback`.
`compute_pmf` already exists in `ffi_wrappers.py` and is cycle-safe
via the same uniformization path.

### `Graph.daisy_chain_log_likelihood_joint_snapshot(...)`

Signature:

```python
def daisy_chain_log_likelihood_joint_snapshot(
    self,
    *,
    epoch_dts: list[float],
    epoch_thetas: jnp.ndarray,        # SVGD parameter
    initial_ipv: np.ndarray,          # fixed before SVGD; not traced
    snapshot_times: jnp.ndarray,      # times at which joint-prob is observed
    snapshot_indices: jnp.ndarray,    # which t-vertex each snapshot picks
    snapshot_counts: jnp.ndarray,     # observation counts at each snapshot
    granularity: int = 100,
) -> jnp.ndarray:
    """Joint-snapshot likelihood: Σ counts[k] · log joint_prob[k].

    `initial_ipv` is the epoch-0 IPV; subsequent epoch IPVs are
    computed internally. `initial_ipv` is **not** SVGD-optimised.
    """
```

Body daisies through epochs the same way, but instead of summing
PMF-of-events per epoch, the final epoch's `joint_stop_probabilities`
at the snapshot times is the source of likelihood:

```python
# After daisying through n_epochs-1 transitions:
joint_probs = self._epoch_joint_stop_probs(
    epoch_thetas[-1], ipv, snapshot_times, granularity
)  # shape (n_snapshot_times, n_t_vertices)
selected = joint_probs[jnp.arange(n_snap), snapshot_indices]
log_lik = jnp.sum(snapshot_counts * jnp.log(selected))
return log_lik
```

`_epoch_joint_stop_probs` wraps the
`update_ipv → update_weights → stop_probability(t) → collapse_t_aux`
sequence in a `pure_callback` that returns the collapsed vector at
each `t`.

**Tests** in `tests/pytest/test_daisy_chain_c_path.py`:

- `test_daisy_chain_per_event_matches_eager`: compare against the
  eager Python loop in `time_inhom_joint_prob.ipynb`. Tolerance:
  `rtol=1e-9` (uniformization is deterministic at fixed
  granularity).
- `test_daisy_chain_joint_snapshot_matches_eager`: same, for the
  snapshot variant.
- `test_daisy_chain_jit`: both methods inside `jax.jit` produce
  the same output as eager.
- `test_daisy_chain_grad`: `jax.grad` over `epoch_thetas` (with
  `initial_ipv` held fixed as a closed-over numpy array) gives
  finite gradients matching finite differences at `rtol=1e-3` on a
  small model (3 epochs, 7 vertices). IPV is not differentiated
  through.
- `test_daisy_chain_vmap`: 10-particle vmap completes and produces
  shape-correct output.
- `test_daisy_chain_symbolic_cache_survives`: across a 100-iteration
  daisy-chain sweep, `param_compute_cache_info()['n_files']`
  increases by exactly 1 (the JSP graph's symbolic compute graph,
  computed once and reused).

**Test gate**: all six pass.

---

# Phase 4 — SVGD integration smoke test

End-to-end SVGD over `epoch_thetas` on a 3-epoch joint-prob model.
IPV is **not** an SVGD parameter — it is captured by closure and
held fixed for the duration of the SVGD loop:

```python
from phasic import SVGD

jsp = joint_kernel.joint_stop_prob_graph()

# User sets the initial IPV once before SVGD. The daisy chain then
# propagates surviving mass between epochs internally.
INITIAL_IPV = np.asarray([...])  # fixed; not optimised

def model(theta):
    return jsp.daisy_chain_log_likelihood_per_event(
        epoch_dts=[0.5, 0.5, 0.5],
        epoch_thetas=theta.reshape(3, 2),
        epoch_events=observed_events_grouped,
        initial_ipv=INITIAL_IPV,   # closed over; constant per evaluation
    )

# theta_dim = 3 epochs × 2 rate parameters = 6.
svgd = SVGD(model, theta_dim=6, n_particles=50, n_iterations=200)
result = svgd.fit(...)
```

`initial_ipv` enters `daisy_chain_log_likelihood_per_event` as a
constant numpy array (or `jnp.asarray(...)` if the user prefers,
but its value is pinned before SVGD starts). Per-epoch IPV
propagation is library-internal: the daisy-chain loop computes
`ipv_{i+1} = transition_fn(epoch_dts[i])(epoch_thetas[i], ipv_i)`,
which threads through `update_ipv` between epochs but is not part
of the SVGD particle vector.

Performance target: 50 particles × 200 iterations on a 67-vertex
joint-prob model, **<2 minutes** (matches Phase 3 budget).

**Test gate**: `tests/pytest/test_daisy_chain_svgd_smoke.py` runs
green within budget.

## Critical files

Python:
- `src/phasic/__init__.py` — add `joint_stop_prob_graph`,
  `joint_stop_probabilities`, `joint_probs_with_time`,
  `_collapse_t_aux`, `epoch_transition_fn`,
  `daisy_chain_log_likelihood_per_event`,
  `daisy_chain_log_likelihood_joint_snapshot`,
  `_epoch_event_pmf`, `_epoch_joint_stop_probs`. Insertion points:
  - `joint_stop_prob_graph` family after `joint_prob_graph` (line 7558).
  - `epoch_transition_fn` and the daisy-chain methods after that.
- `src/phasic/ffi_wrappers.py` — possibly add
  `compute_stop_prob_ffi` if the existing `compute_pmf_ffi` doesn't
  cover the stop-prob signature.

C / C++ / pybind: **no changes**. Stage A0 + the existing
`stop_probability` and `update_weights` bindings, plus the
`update_ipv` binding from the prerequisite plan, are sufficient.

Tests (new):
- `tests/pytest/test_daisy_chain_c_path.py`
- `tests/pytest/test_daisy_chain_svgd_smoke.py`

## Verification (end-to-end)

Success criterion (the user's actual workflow):

```python
# Build joint kernel once.
joint_kernel = Graph(coal_callback, indexer=indexer).joint_prob_graph(
    indexer, mutation_rate=0.1, reward_limit=5,
)
jsp = joint_kernel.joint_stop_prob_graph()
INITIAL_IPV = np.asarray([...])  # set once before SVGD; not optimised

# JAX-traceable daisy-chain — only epoch_thetas is differentiated.
def log_lik(epoch_thetas):
    return jsp.daisy_chain_log_likelihood_per_event(
        epoch_dts=[0.5, 0.5, 0.5],
        epoch_thetas=epoch_thetas.reshape(3, 2),
        epoch_events=observed_events_grouped,
        initial_ipv=INITIAL_IPV,   # closed over; constant per evaluation
    )

# Must JIT and grad over epoch_thetas cleanly.
log_lik_jit = jax.jit(log_lik)
grad_fn = jax.grad(log_lik)
v = log_lik_jit(jnp.array([1.0, 0.5, 1.5, 0.8, 2.0, 1.0]))
g = grad_fn(jnp.array([1.0, 0.5, 1.5, 0.8, 2.0, 1.0]))

# Cache survived the full sweep.
import phasic.cache as cache
assert cache.param_compute_cache_info()['n_files'] == 1
```

Equivalence: the JIT output must bit-match the eager-Python-loop
output at `rtol=1e-9, atol=1e-12` (uniformization is fully
deterministic).

## Risks and open issues

1. **`pure_callback` overhead per epoch**. Each epoch transition
   crosses the JAX→Python→C boundary three times (`update_ipv`,
   `update_weights`, `stop_probability`). For 30 epochs this is 90
   crossings per evaluation. Each crossing is ~10-50 µs on M1, so
   the overhead floor per evaluation is ~1-5 ms before any actual
   compute. For 1000 SVGD evaluations that is 1-5 seconds of pure
   boundary cost — acceptable but worth measuring.
   *Mitigation*: a future optimisation could fuse the three calls
   into a single `epoch_transition_ffi` C function; not in scope
   for this plan.
2. **`stop_probability` per-call context rebuild**. The
   `weight_version` cache check at `phasiccpp.h:770` invalidates
   the `ph_context_markov` on every `update_weights` /
   `update_ipv` call — so each epoch transition rebuilds the
   uniformization context from scratch. This is `O(vertices²)` per
   epoch; for 67 vertices and 30 epochs that is ~135k operations
   per evaluation. Negligible compared to the actual uniformization
   step count, but worth verifying with a benchmark.
   *Mitigation*: none needed at this stage; the symbolic compute
   graph cache (Stage A0) already handles the expensive part.
3. **Cycle robustness in t-aux loops**. The trapping aux loop
   construction (Phase 1, item 4) creates a 2-vertex cycle
   (t-vertex ↔ aux). The C uniformization path handles cycles, but
   the `granularity = 0` (auto) heuristic computes `2 × max_rate`
   from edge weights — large IPV or theta values could spike
   `max_rate` and inflate the per-call step count. Document the
   trade-off in the docstring; recommend explicit `granularity`
   when sweeping wide-range thetas.
4. **Snapshot-variant indexing**. The `snapshot_indices` argument
   in `daisy_chain_log_likelihood_joint_snapshot` requires the user
   to know the t-vertex order in the JSP graph. Provide a helper
   `Graph.joint_snapshot_indexer(state)` that converts a base-graph
   state to its t-vertex index in the JSP graph; document the
   helper as the recommended way to construct `snapshot_indices`.
5. **`add_edge_parameterized` deprecation**. The notebook helper
   uses `add_edge_parameterized` directly. The library version
   should use the modern list-based `add_edge(to, [coefficients])`
   form to avoid emitting deprecation warnings to users running
   `daisy_chain_log_likelihood_*`.

## Future work (deferred)

- **Fused C kernel** for `epoch_transition`: a single C function
  that takes `(graph, theta, ipv, dt)` and returns `ipv_next` in
  one FFI call instead of three. Would amortise the
  `pure_callback` boundary cost. Worth doing if benchmarks show
  the boundary cost dominates.
- **`lax.scan`-based daisy-chain**: if the number of epochs ever
  grows above ~50, unrolling becomes a JIT-compilation pain point
  and `lax.scan` becomes attractive. Out of scope here per the
  user's <30-epoch constraint.
- **Joint inference of epoch boundaries**: the user has confirmed
  epoch boundaries are fixed. If that ever changes, the daisy
  chain needs `dt`s to become JAX-traceable; the per-epoch loop
  would need to use `lax.cond` instead of Python `for`. Out of
  scope.

## Out of scope

- Any modifications to `src/phasic/trace_elimination.py` or to the
  Python codegen path (`_generate_cpp_from_trace`,
  `_wrap_trace_log_likelihood_for_jax`). The Python trace path is
  not on the critical path for this plan.
- Any changes to `src/c/phasic.c`, `src/cpp/phasiccpp.cpp`, or
  `src/cpp/phasic_pybind.cpp` — the prerequisite `update_ipv-plan.md`
  has already covered the C-side surface this plan needs.
- A `joint_stop_prob_graph` alternative that does not use t-aux
  loops. The aux-loop construction is the established notebook
  pattern; reproducing it in the library preserves bit-equivalence
  with the eager reference.
- Restructuring `joint_prob_graph` itself. Treat it as a read-only
  building block.
