# Plan: JAX-traceable daisy-chained likelihood for time-inhomogeneous joint-prob models

This plan is a follow-on to `update_ipv-plan.md`. It assumes Phase 1 of that plan has landed (i.e. `Graph.update_ipv(weights)` exists and `record_elimination_trace(parameterized_ipv=True)` records IPV as runtime PARAM ops in the trace's extended parameter vector `[theta, ipv, rewards]`).

This plan is self-contained and can be picked up later independently. It does not modify or supersede anything in `update_ipv-plan.md`.

## Context

The user computes the likelihood of a piecewise-constant time-inhomogeneous coalescent model by **daisy-chaining** continuous joint-probability kernels — see `docs/pages/tutorial/time_inhom_joint_prob.ipynb`. The eager Python loop is:

```python
for i in range(n_epochs - 1):
    cloned = clone_with_ipv(joint_kernel, epoch_ipv[i])
    jsp_graph, t_map = joint_stop_prob_graph(cloned)
    jsp_graph.update_weights(epoch_theta[i])
    epoch_ipv[i+1] = joint_stop_probabilities(jsp_graph, epoch_starts[i+1], t_map)

# final epoch: read joint absorption probabilities at observed times
log_lik = ... function of joint_probs_with_time(jsp_graph_final, ...) ...
```

`epoch_ipv[0]` is the kernel's natural IPV. Per epoch: clone kernel → set IPV → transform graph (`joint_stop_prob_graph`) → set rates → evaluate `stop_probability` at the epoch's end time → collapse t-aux pairs → that becomes the next epoch's IPV. The final epoch is special: it reads off joint probabilities used as data-likelihood factors.

For SVGD inference of `epoch_thetas`, this loop must be JAX-traceable (`jit`/`grad`/`vmap` compatible). Phase 1 of `update_ipv-plan.md` is necessary but insufficient — it makes IPV a runtime parameter on the trace, but the per-epoch `stop_probability` call and the `t_aux_map` collapse must also become JAX-compatible. This plan provides those pieces and assembles them into `log_lik(epoch_thetas) -> scalar` for SVGD.

### User-confirmed constraints

- Epoch boundaries `epoch_starts` are **fixed** (not inferred). Therefore each epoch's `dt` is a static Python float.
- Initial epoch IPV is **fixed** at the model's defined IPV. Not an SVGD parameter.
- Number of epochs is typically **<30**. Unrolling the daisy chain is fine; `lax.scan` is overkill.
- `t_aux_map` may be inferred from graph structure, which is preferred — see Phase 2.1.

### Why not `add_epoch`?

`Graph.add_epoch(time, ...)` (`src/phasic/__init__.py:3052-3278`) is the existing "build one big multi-epoch graph" pattern. It wires sister-vertex transitions weighted by `stop_probability/accumulated_occupancy`, which is correct for standard phase-type epoch composition. **It does not preserve the joint-distribution-over-rewards** needed for joint-prob inference (the reward-extended state and the t-vertex absorption-mass interpretation). The user genuinely needs the daisy chain pattern for the joint-prob case. `add_epoch` continues to be the right answer for non-joint cases.

## What's already in the codebase (reusable)

- `phasic::Graph::stop_probability(time, granularity)` — C++ method, pybind-exposed at `src/cpp/phasic_pybind.cpp:2633`, Python wrapper at `src/phasic/__init__.py:2734-2760`. Same algorithmic complexity as `pdf` (uniformization), returns probability mass vector at each vertex.
- `Graph.joint_prob_graph(...)` at `src/phasic/__init__.py:7242-7519` — produces continuous joint-probability kernels with reward-extended state, "trash" pair (state-zero self-loop), and t-vertices. Sets `_joint_prob_base_graph_indexer` and `_rewarded_props` on the result.
- `_generate_cpp_from_trace` at `src/phasic/__init__.py:801` — emits standalone C++ for log-likelihood with embedded trace data. The trace evaluation block (`evaluate_embedded_trace`) and trace-to-graph instantiation (lines 1059-1097) are reusable for the stop-prob variant.
- `_wrap_trace_log_likelihood_for_jax` (`__init__.py:~1231`) — wraps a compiled `.so` for JAX via `pure_callback` with `vmap_method='sequential'`. Template for the stop-prob wrapper.
- `_compile_trace_library` — disk-cached compilation of generated C++.
- The notebook helper `joint_stop_prob_graph` (in `docs/pages/tutorial/time_inhom_joint_prob.ipynb`) is the structural transformation that needs to be promoted into the library.

## Design decision: no silent fallbacks

Per the user's explicit preference, the API exposes **two distinct entry points** for the two distinct likelihood patterns rather than one method whose behavior switches on an optional argument. Each method has a fully required, fully explicit signature.

---

# Phase 2 — JAX-traceable per-epoch IPV transition

## Phase 2.1 — Promote `joint_stop_prob_graph` from notebook to library

Insertion point: `src/phasic/__init__.py` immediately after `Graph.joint_prob_graph` (currently ends at line 7519). Add three new methods on `Graph`:

### `Graph.joint_stop_prob_graph(self) -> Graph`

Library version of the notebook function. Differences from notebook:

1. Does **not** call `clone_with_ipv` internally. Phase 1 makes IPV a runtime parameter on the trace; the JSP graph is built **once** off the base joint-prob graph and the IPV is set at evaluation time via `update_ipv`.
2. Detects t-vertices structurally: vertex `v` is a t-vertex iff any outgoing edge leads to an absorbing vertex (`len(edge.to().edges()) == 0`).
3. Detects the trash pair via the notebook's `is_trash` predicate (state-zero self-loop pair).
4. For each t-vertex, replaces its outgoing edges with a "trapping aux loop": creates an aux vertex with state-zero, adds bidirectional unit-weight parameterized edges. Mass that reaches a t-vertex shuttles between it and its aux indefinitely, so `stop_probability(t)[t_vertex] + stop_probability(t)[aux]` integrates to the joint cumulative absorption mass for that t-state.
5. Sets attributes on the returned graph:
   ```python
   new._joint_prob_base_graph_indexer = self._joint_prob_base_graph_indexer
   new._rewarded_props = self._rewarded_props
   new._joint_stop_prob_graph = True              # marker
   new._t_vertex_indices = sorted(t_vertex_indices)   # new-graph indices
   new._t_aux_map = t_aux_vertex_indices              # new-graph indices
   new.is_discrete = self.is_discrete
   new._cache_trace = self._cache_trace
   ```
6. Validation: raises `ValueError` if `not self._joint_prob_base_graph_indexer` or `self.param_length() == 0`.

The `_joint_stop_prob_graph = True` marker lets downstream methods verify the graph type without duck-typing. `_t_aux_map` and `_t_vertex_indices` are kept in **new-graph index space** (matching what the trace will see).

### `Graph.joint_stop_probabilities(self, t) -> np.ndarray`

Eager numpy version. Calls `self.stop_probability(t)` and applies `_collapse_t_aux` using `self._t_aux_map`:

```python
def joint_stop_probabilities(self, t):
    if not getattr(self, '_joint_stop_prob_graph', False):
        raise ValueError(".joint_stop_probabilities requires graph from .joint_stop_prob_graph()")
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

Display helper. Uses `self._t_vertex_indices` to slice the collapsed vector and builds a DataFrame of (state, joint_prob) rows.

## Phase 2.2 — `_generate_cpp_stop_prob_from_trace`

Insertion point: `src/phasic/__init__.py` next to `_generate_cpp_from_trace` (around line 801). Reuses ~95% of the existing codegen — same trace metadata block, same `evaluate_embedded_trace` block (already extended in Phase 1 to take `(theta, ipv)`), same trace-to-graph instantiation. Only the final compute step changes.

Generated C++ entry point:

```cpp
extern "C" void compute_stop_prob(
    const double* theta, int n_theta,
    const double* ipv,   int n_ipv,
    double t,
    double* out_stop_prob, int n_out_vertices)
{
    if (n_theta != PARAM_LENGTH || n_ipv != IPV_LENGTH ||
        (size_t)n_out_vertices != N_VERTICES) {
        for (int i = 0; i < n_out_vertices; i++) out_stop_prob[i] = NAN;
        return;
    }
    struct ptd_trace_result* result = evaluate_embedded_trace(theta, n_theta, ipv, n_ipv);
    struct ptd_graph* graph = ptd_instantiate_from_trace(result, &trace_struct);

    phasic::Graph g_wrap(graph, /*own_avl_tree=*/...);
    std::vector<double> sp = g_wrap.stop_probability(t, GRANULARITY);
    for (size_t i = 0; i < N_VERTICES; i++) out_stop_prob[i] = sp[i];

    ptd_trace_result_destroy(result);
    // g_wrap destructor frees graph + avl_tree
}
```

Key details:

- **`t` is a runtime arg**, not embedded as static const. One binary serves all `t` values.
- **`granularity` is embedded** at codegen time as `GRANULARITY` (matches existing `_generate_cpp_from_trace`).
- **`#include "phasiccpp.h"`** is already in the existing codegen template.
- **Constructing `phasic::Graph` from `ptd_graph*`**: use the C++ wrapper to leverage `stop_probability`'s implementation. The wrapper destructor handles cleanup. Alternative: inline the C-level uniformization (`ptd_probability_distribution_context_create`, `ptd_probability_distribution_step` loop). Recommend the wrapper for simplicity.

Cache key for the compiled `.so` must include a function-kind discriminator (e.g. `"stop_prob"` vs `"log_lik"`) so the two binaries don't collide. `t` is **not** in the cache key.

## Phase 2.3 — `_wrap_trace_stop_prob_for_jax`

Insertion point: `src/phasic/__init__.py` after `_wrap_trace_log_likelihood_for_jax` (~line 1310).

```python
def _wrap_trace_stop_prob_for_jax(lib_path, param_length, ipv_length, n_vertices):
    """Wrap C++ stop_probability function for JAX.

    Returns a JAX-compatible function:
        stop_prob(theta, ipv, t) -> jnp.ndarray of shape (n_vertices,)

    `t` may be a Python float (closed over) or a JAX scalar (vmap'd over).
    """
    import ctypes
    lib = ctypes.CDLL(lib_path)
    lib.compute_stop_prob.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ]
    lib.compute_stop_prob.restype = None

    def stop_prob_cpp(theta_np, ipv_np, t_float):
        out = np.empty(n_vertices, dtype=np.float64)
        lib.compute_stop_prob(
            theta_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), param_length,
            ipv_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ipv_length,
            float(t_float),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n_vertices,
        )
        return out

    def stop_prob_jax(theta, ipv, t):
        result_shape = jax.ShapeDtypeStruct((n_vertices,), jnp.float64)
        return jax.pure_callback(
            lambda th, iv, tt: stop_prob_cpp(np.asarray(th), np.asarray(iv), float(tt)),
            result_shape, theta, ipv, t,
            vmap_method='sequential',
        )
    return stop_prob_jax
```

`vmap_method='sequential'` matches the existing log-likelihood wrapper. Gradients route through finite differences (same as log-lik wrapper); a `custom_vjp` upgrade is future work.

## Phase 2 deliverables

- `Graph.joint_stop_prob_graph(self) -> Graph`
- `Graph.joint_stop_probabilities(self, t)` (instance method using `self._t_aux_map`)
- `Graph.joint_probs_with_time(self, t)` (display helper)
- `Graph._collapse_t_aux(self, raw_vec)` (helper)
- New attributes on JSP graphs: `_joint_stop_prob_graph`, `_t_aux_map`, `_t_vertex_indices`
- `_generate_cpp_stop_prob_from_trace(trace, granularity=0)` in `__init__.py`
- `_wrap_trace_stop_prob_for_jax(lib_path, param_length, ipv_length, n_vertices)` in `__init__.py`
- `_compile_trace_library` cache key extended with function-kind discriminator

---

# Phase 3 — Daisy-chain log-likelihood

## Phase 3.1 — `Graph.epoch_transition_fn(self, dt)`

Insertion point: `src/phasic/__init__.py`, near `joint_stop_prob_graph`.

```python
def epoch_transition_fn(self, dt):
    """Return JAX function (theta, ipv) -> next_ipv for one epoch of duration dt.

    Self must be the output of .joint_stop_prob_graph(). Records the elimination
    trace once (with parameterized_ipv=True), compiles the C++ stop-probability
    binary once (cached on disk), and on every subsequent call evaluates
    stop_probability(dt) under the supplied (theta, ipv) and applies the t_aux
    collapse to produce the next epoch's IPV.

    `dt` is closed over (static Python float). Different dt → different
    transition function, but trace and binary are reused via the on-disk cache.
    """
    if not getattr(self, '_joint_stop_prob_graph', False):
        raise ValueError("requires graph from .joint_stop_prob_graph()")

    trace = self.record_elimination_trace(parameterized_ipv=True)
    cpp_code = _generate_cpp_stop_prob_from_trace(trace, granularity=0)
    lib_path = _compile_trace_library(cpp_code, _stop_prob_cache_key(trace))
    stop_prob_fn = _wrap_trace_stop_prob_for_jax(
        lib_path, trace.param_length, trace.ipv_length, trace.n_vertices,
    )

    # Precompute static collapse indices
    n = self.vertices_length()
    aux_set = set(self._t_aux_map.values())
    keep_indices = [i for i in range(n) if i not in aux_set]
    aux_for_keep = [self._t_aux_map.get(k, -1) for k in keep_indices]
    keep_arr = jnp.asarray(keep_indices)
    has_aux = jnp.asarray([a >= 0 for a in aux_for_keep])
    aux_arr = jnp.asarray([a if a >= 0 else 0 for a in aux_for_keep])

    def collapse(raw_vec):
        kept = raw_vec[keep_arr]
        aux_contrib = jnp.where(has_aux, raw_vec[aux_arr], 0.0)
        return kept + aux_contrib

    def epoch_transition(theta, ipv):
        # ipv is in collapsed (non-aux) layout; pad with zeros at aux slots
        # before passing to the C++ binary (which expects trace's full layout).
        padded_ipv = _pad_ipv_aux_zeros(self, ipv)
        raw = stop_prob_fn(theta, padded_ipv, dt)   # length n_vertices
        return collapse(raw)                         # length n_vertices - n_aux

    return epoch_transition
```

Two important properties:

1. **`dt` closed over**, not JAX-traced — appropriate because epoch boundaries are fixed.
2. **Static collapse and pad indices** — baked at construction time. JAX trace through `epoch_transition` has fully static control flow.

### IPV layout invariant (critical)

The trace's `ipv_length` (Phase 1 Choice B) counts non-absorbing-non-starting vertices, which **includes aux vertices** in the JSP graph. The collapsed output IPV (length `n_vertices - n_aux`) is shorter. `epoch_transition_fn` must zero-pad incoming `ipv` at aux slots before passing to the C++ binary — those slots must always be zero (no probability mass should start in an aux vertex).

Implementation: store `_aux_ipv_slot_indices` and `_collapsed_to_padded_indices` on the JSP graph during `joint_stop_prob_graph()`. Use them in `_pad_ipv_aux_zeros` to expand a length-`(n_vertices - n_aux)` collapsed IPV into the length-`ipv_length` padded layout.

Add an assertion at construction time that the trace's IPV slot order matches the collapsed (non-aux) order; if they differ, materialize a permutation index inside the pad/collapse helpers.

## Phase 3.2 — Two distinct daisy-chain methods

The two likelihood patterns are exposed as **two distinct methods** with fully required, fully explicit signatures. No optional arguments that switch behavior.

### Pattern A: per-event likelihood

Used when each observation is a distinct absorption event with a known time and a known target t-state.

```python
def daisy_chain_log_likelihood_per_event(
    self,
    initial_ipv,           # length self.vertices_length() - n_aux
    epoch_dts,             # length n_epochs (Python list / numpy — static)
    event_times,           # length n_events; each in the FINAL epoch's local time
    event_targets,         # length n_events; each is an index into self._t_vertex_indices
):
    """Return log_lik(epoch_thetas) -> scalar.

    Likelihood model: each observation k is an absorption event observed at
    time event_times[k] (within the final epoch) at the t-state
    self._t_vertex_indices[event_targets[k]]. The log-likelihood is

        Σ_k log p_{event_targets[k]}(event_times[k])

    where p_v(t) is the joint absorption probability for t-state v at time t.

    Self must be the output of .joint_stop_prob_graph().

    Parameters
    ----------
    initial_ipv : array-like, length self.vertices_length() - n_aux
        IPV at the start of the first epoch. Caller is responsible for
        building this from the base joint-prob graph's natural IPV.
    epoch_dts : array-like, length n_epochs
        Duration of each epoch. Static — closed over.
    event_times : array-like, length n_events
        Observation times within the final epoch (measured from final epoch start).
    event_targets : array-like of ints, length n_events
        For each observation, the index k into self._t_vertex_indices of the
        absorbing t-state. Required.

    Returns
    -------
    log_lik(epoch_thetas) -> scalar.
        epoch_thetas is shape (n_epochs, param_length). JIT/vmap/grad-compatible.
    """
    n_epochs = len(epoch_dts)
    if len(event_times) != len(event_targets):
        raise ValueError("event_times and event_targets must have same length")

    transitions = [self.epoch_transition_fn(float(dt)) for dt in epoch_dts]

    expected_len = self.vertices_length() - len(self._t_aux_map)
    if len(initial_ipv) != expected_len:
        raise ValueError(f"initial_ipv length {len(initial_ipv)} != expected {expected_len}")

    initial_ipv = jnp.asarray(initial_ipv, dtype=jnp.float64)
    event_times = jnp.asarray(event_times, dtype=jnp.float64)
    event_targets = jnp.asarray(event_targets, dtype=jnp.int32)

    # The final epoch reuses the same compiled binary but with t as a JAX-traced arg.
    # Rebuild the wrapper without closing over dt:
    final_stop_prob = transitions[-1].__closure__[...]  # extract stop_prob_fn
    # (Or factor out: have epoch_transition_fn return both `epoch_transition` AND
    # the underlying `stop_prob_fn` so daisy_chain can vmap it over event times.)
    final_collapse = transitions[-1].__closure__[...]   # extract collapse fn

    def log_lik(epoch_thetas):
        ipv = initial_ipv
        for i in range(n_epochs - 1):
            ipv = transitions[i](epoch_thetas[i], ipv)

        # Final epoch: vmap over events
        final_theta = epoch_thetas[-1]
        padded_ipv = _pad_ipv_aux_zeros(self, ipv)
        t_vertex_arr = jnp.asarray(self._t_vertex_indices)

        def per_event(t_obs, target_k):
            raw = final_stop_prob(final_theta, padded_ipv, t_obs)
            collapsed = final_collapse(raw)
            # event_targets[k] indexes into self._t_vertex_indices
            t_vertex_idx = t_vertex_arr[target_k]
            joint_prob = collapsed[t_vertex_idx]
            return jnp.log(jnp.maximum(joint_prob, 1e-300))

        log_factors = jax.vmap(per_event)(event_times, event_targets)
        return jnp.sum(log_factors)

    return log_lik
```

### Pattern B: joint-snapshot likelihood

Used when the data is a vector of counts over t-states, observed at a single fixed time within the final epoch.

```python
def daisy_chain_log_likelihood_joint_snapshot(
    self,
    initial_ipv,           # length self.vertices_length() - n_aux
    epoch_dts,             # length n_epochs (static)
    snapshot_time,         # single float (within the final epoch)
    counts,                # length len(self._t_vertex_indices)
):
    """Return log_lik(epoch_thetas) -> scalar.

    Likelihood model: at the final epoch's snapshot_time, the joint
    absorption probabilities at all t-states are observed as count data.
    The log-likelihood is

        Σ_v counts[v] · log p_v(snapshot_time)

    (a multinomial-style log-likelihood; for each t-state v, counts[v] is
    the number of independent observations absorbed at v.)

    To match the notebook's joint_probs_with_time pattern (one joint_prob
    per t-state, treated as a single observation per state), pass
    counts=jnp.ones(len(self._t_vertex_indices)).

    Self must be the output of .joint_stop_prob_graph().

    Parameters
    ----------
    initial_ipv : array-like, length self.vertices_length() - n_aux
    epoch_dts : array-like, length n_epochs (static)
    snapshot_time : float
        Single time within the final epoch at which joint probs are evaluated.
    counts : array-like, length len(self._t_vertex_indices)
        Observation count for each t-state. Required — caller must specify
        explicitly (no default of "all ones").

    Returns
    -------
    log_lik(epoch_thetas) -> scalar.
    """
    n_epochs = len(epoch_dts)
    n_t_vertices = len(self._t_vertex_indices)
    if len(counts) != n_t_vertices:
        raise ValueError(
            f"counts length {len(counts)} != number of t-states {n_t_vertices}"
        )

    transitions = [self.epoch_transition_fn(float(dt)) for dt in epoch_dts]

    expected_len = self.vertices_length() - len(self._t_aux_map)
    if len(initial_ipv) != expected_len:
        raise ValueError(f"initial_ipv length {len(initial_ipv)} != expected {expected_len}")

    initial_ipv = jnp.asarray(initial_ipv, dtype=jnp.float64)
    counts = jnp.asarray(counts, dtype=jnp.float64)
    t_vertex_arr = jnp.asarray(self._t_vertex_indices)

    final_stop_prob = transitions[-1].__closure__[...]
    final_collapse = transitions[-1].__closure__[...]

    def log_lik(epoch_thetas):
        ipv = initial_ipv
        for i in range(n_epochs - 1):
            ipv = transitions[i](epoch_thetas[i], ipv)

        # Final epoch: single stop_prob call at snapshot_time
        final_theta = epoch_thetas[-1]
        padded_ipv = _pad_ipv_aux_zeros(self, ipv)
        raw = final_stop_prob(final_theta, padded_ipv, snapshot_time)
        collapsed = final_collapse(raw)
        joint_probs = collapsed[t_vertex_arr]    # length n_t_vertices
        return jnp.sum(counts * jnp.log(jnp.maximum(joint_probs, 1e-300)))

    return log_lik
```

### Why two methods, not one with a switch?

Per the user's "no silent fallbacks" preference (`feedback_no_silent_fallbacks.md`): an API whose behavior changes implicitly based on which optional argument is passed is a silent fallback. The two patterns produce *fundamentally different* likelihood expressions (one is `Σ_k log p_{tgt[k]}(t[k])`, the other is `Σ_v counts[v] log p_v(t)`). They warrant distinct names so the caller's intent is explicit at the call site.

Both methods refuse to default `counts` or `event_targets` to "all ones" or "all t-states" — those would be silent fallbacks too. Caller must specify them explicitly.

### Refactoring note

The `__closure__[...]` access patterns above are pseudocode — in real implementation, refactor `epoch_transition_fn` to return a small dataclass:

```python
@dataclass
class EpochKernel:
    transition: Callable        # (theta, ipv) -> next_ipv (dt closed over)
    stop_prob_fn: Callable      # (theta, ipv, t) -> raw vec   (general)
    collapse: Callable          # (raw_vec) -> collapsed_vec
    pad_ipv: Callable           # (collapsed_ipv) -> padded_ipv
```

so `daisy_chain_log_likelihood_*` can directly reference `kernels[-1].stop_prob_fn` and friends instead of digging into closures.

### Why unrolled instead of `lax.scan`

`lax.scan` requires each scanned function to share signature. Each `transitions[i]` closes over a different `dt`. Two ways to use scan: (a) make `dt` a JAX-traced argument to a single transition function, or (b) unroll. **Recommend unrolling** at <30 epochs — simpler, equally efficient, JIT produces a single fused XLA program. Document escape hatch: switch to `lax.scan` with `dt` as a scan input if `n_epochs` ever grows to thousands.

## Phase 3 deliverables

- `Graph.epoch_transition_fn(self, dt)` returning an `EpochKernel` (dataclass with `transition`, `stop_prob_fn`, `collapse`, `pad_ipv`)
- `Graph.daisy_chain_log_likelihood_per_event(...)` (Pattern A)
- `Graph.daisy_chain_log_likelihood_joint_snapshot(...)` (Pattern B)
- `Graph._pad_ipv_aux_zeros(self, collapsed_ipv)` (helper)
- New attribute on JSP graphs: `_aux_ipv_slot_indices`, `_collapsed_to_padded_indices`

## Phase 3 verification plan

New test file `tests/pytest/test_daisy_chain_inference.py` with these tiers:

**Tier 1 — bit-equivalence with notebook eager Python loop**

1. Build small parameterized coalescent (3 samples, mutation rate 0.5, reward limit 5) via `Graph.joint_prob_graph`.
2. Build `jsp = base.joint_stop_prob_graph()`.
3. Set up 5 epochs with known thetas and dts.
4. Reference: run the notebook's eager Python loop literally (`clone_with_ipv` → `joint_stop_prob_graph` → `update_weights` → `joint_stop_probabilities`).
5. JAX path A: `log_lik = jsp.daisy_chain_log_likelihood_per_event(...)`; evaluate.
6. JAX path B: `log_lik = jsp.daisy_chain_log_likelihood_joint_snapshot(...)`; evaluate.
7. Assert both match the eager reference to `atol=1e-9, rtol=1e-9`.

**Tier 2 — JIT, grad, vmap**

8. `jax.jit(log_lik)` matches eager.
9. `jax.grad(log_lik)` returns finite gradients of shape `(n_epochs, param_length)`.
10. `jax.vmap(log_lik)(particle_thetas)` matches per-particle Python loop.

**Tier 3 — single-epoch sanity**

11. With `n_epochs == 1`, daisy chain reduces to a single `epoch_transition` followed by final-epoch likelihood. Compare to direct `jsp.joint_stop_probabilities(t)` + manual log on t-vertex indices.

**Tier 4 — SVGD round-trip (smoke test)**

12. Generate synthetic observations from a known `epoch_thetas_true` via the eager Python path (Pattern A: per-event observations).
13. Run `phasic.SVGD(log_lik, theta_dim=n_epochs * param_length, n_particles=50, n_iterations=500)`.
14. Assert posterior mean within 20% of true on each epoch.
15. Instrument `record_elimination_trace` call count — must be exactly 1 across all SVGD iterations.

**Tier 5 — t_aux_map structural detection**

16. `_t_aux_map` and `_t_vertex_indices` from library `joint_stop_prob_graph` match notebook eager construction on the same input.

**Tier 6 — IPV layout invariant**

17. Build a JSP graph; confirm `_aux_ipv_slot_indices` exactly matches the trace's IPV slots that correspond to aux vertices.
18. Confirm `_pad_ipv_aux_zeros` round-trips: pad then read back the non-aux slots → original input.
19. Build a graph where collapsed-layout order differs from trace IPV-slot order; confirm permutation is applied correctly.

**Tier 7 — validation errors**

20. Each input to both methods has at least one validation rule (length, type, etc.); test each rule fires with a clear `ValueError`.

Smoke broader: `pixi run -- pytest tests/pytest/ -k "daisy or epoch or joint_stop or svgd" -x`.

## Implementation order recommendation

1. **Prerequisite**: Phase 1 of `update_ipv-plan.md` must be landed (parameterized IPV in trace).
2. **Phase 2.1** — promote `joint_stop_prob_graph`, `joint_stop_probabilities`, `joint_probs_with_time` into the library. Notebook updates to import library functions instead of pasting helpers. Tier 5 of verification.
3. **Phase 2.2** — extend `_generate_cpp_from_trace` to emit stop-prob variant. Test via direct `ctypes` call vs `phasic::Graph::stop_probability`.
4. **Phase 2.3** — `_wrap_trace_stop_prob_for_jax`. Test under `jax.jit` and `jax.vmap`.
5. **Phase 3.1** — `Graph.epoch_transition_fn`. Test single-epoch propagation against notebook eager path. Tier 6 (IPV layout invariant).
6. **Phase 3.2** — both `daisy_chain_log_likelihood_*` methods. Run Tiers 1, 3, 7.
7. **Phase 3.2 cont.** — Run Tiers 2, 4 (JIT/grad/vmap, SVGD round-trip).

Steps 2–4 can be developed in parallel with Phase 1 (a JSP graph can be hand-constructed for unit tests).

## Critical files

- **`src/phasic/__init__.py`**:
  - After `joint_prob_graph` (line 7519): `joint_stop_prob_graph`, `joint_stop_probabilities`, `joint_probs_with_time`, `_collapse_t_aux`, `_pad_ipv_aux_zeros`, `epoch_transition_fn`, `daisy_chain_log_likelihood_per_event`, `daisy_chain_log_likelihood_joint_snapshot`.
  - Near `_generate_cpp_from_trace` (line 801): `_generate_cpp_stop_prob_from_trace`.
  - Near `_wrap_trace_log_likelihood_for_jax` (~line 1231): `_wrap_trace_stop_prob_for_jax`.
  - `_compile_trace_library` cache key: add function-kind discriminator.
- **`docs/pages/tutorial/time_inhom_joint_prob.ipynb`**: update to import library functions instead of pasting helpers.
- **`api/cpp/phasiccpp.h`** / **`src/cpp/phasiccpp.cpp`**: no edits needed. `stop_probability` and `Graph(graph_ptr, avl_tree)` constructor are already exposed.
- **`src/c/phasic.c`**: no edits needed.
- **`tests/pytest/test_daisy_chain_inference.py`**: new file with seven-tier verification.

## Risks and open issues

1. **`pure_callback` gradients via finite differences**. For SVGD with 50 particles × 30 epochs × 2 params × 2 finite-diff evals, that's ~6000 forward calls per gradient step. At ~1ms each on small graphs, ~6s per SVGD step. Acceptable for v1; future upgrade is `custom_vjp` wrapping a hand-derived stop-probability gradient.
2. **Aux-vertex IPV slot zeroing**. Aux vertices are non-absorbing → eligible by Phase 1's IPV slot definition. Must always receive zero IPV (no mass starts there). `epoch_transition_fn` zero-pads incoming collapsed-layout `ipv` to the trace's aux-inclusive layout. Mechanical but easy to get wrong — Tier 6 verification is essential.
3. **`is_trash` heuristic fragility**. The notebook's `is_trash` predicate depends on the exact structure produced by `joint_prob_graph`'s trash-pair construction (`__init__.py:7444-7448`). Add a comment in `joint_prob_graph` referencing the consumer; if trash construction ever changes, `is_trash` must be updated in lockstep.
4. **Stop-prob caching across `t` values**. The C++ wrapper caches a `ph_context_markov` keyed on granularity and reuses it for monotonically increasing `t` — but the codegen constructs a fresh `phasic::Graph` per call, so we don't get this incremental optimization. For per-epoch use (single `dt` per call) this is fine. For Pattern A's vmap-over-event-times, there's a potential 10–100× speedup by computing one cumulative trajectory and reading off multiple times. Defer as future optimization.
5. **`stop_probability` granularity choice**. Defaults to `granularity=0` (auto-select based on max rate). Document that very large `dt` × large `theta` may need a granularity override.
6. **Initial IPV construction**. The user's notebook has `epoch_ipv = get_ipv(joint_graph)` extracting IPV from the base joint-prob graph. The library should expose `Graph.get_ipv(self) -> np.ndarray` as a public method (the notebook helper version). Length contract: returns a length-`vertices_length()` array (full layout). The caller passes it to `daisy_chain_log_likelihood_*` after dropping aux slots (or the method internally drops them and validates the resulting length matches the JSP graph's collapsed layout). Recommend: add `Graph.get_ipv()` as a separate small method and have `daisy_chain_log_likelihood_*` accept input in the **JSP graph's collapsed layout** (length `vertices_length() - n_aux`), with a clear error if the user passes the wrong-length array.

## Future work (deferred)

- `custom_vjp` for stop-prob with hand-derived analytic gradients (~10–100× SVGD speedup).
- Final-epoch optimization for Pattern A: single cumulative trajectory across multiple event times.
- `lax.scan` variant of `daisy_chain_log_likelihood_*` for `n_epochs > 100`.
- Generalize `add_epoch` to handle joint-prob graphs (would subsume the daisy chain entirely — much larger project).
