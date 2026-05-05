# Plan: `Graph.update_ipv(weights)` with parameterized IPV in the elimination trace

## Status & currency

**Last reconciled with the codebase: 2026-05-05** (commit `69a5857`,
post-Stages 1/2/3/A0/A1/A2 + `phasic.cache` module).

This plan has not been started. Greenfield additions; verified by
grepping master for `update_ipv`, `parameterized_ipv`, `ipv_length`,
`ipv_targets` (zero hits in `src/`, `tests/`).

### Hard prerequisite — and what it blocks

**Hard prerequisite**: this plan extends `record_elimination_trace`
(`src/phasic/trace_elimination.py`). That function currently raises on
cyclic graphs at `trace_elimination.py:834`:

```
RuntimeError: Trace-based elimination cannot handle the cycle
(parent={i} → i={j} → parent={i}): self-loop correction 1/(1 − q)
is not implemented.
```

This is the v1 trace-plan's central deferred work and is **not** part
of any of the C-side stages 1/2/3/A0/A1/A2 that have landed. Until
that fix lands, the IPV machinery this plan adds will only work on
acyclic parameterised graphs. Most real models (ARGs,
joint-probability kernels, anything with back-edges) have cycles, so
this is a sharp limit on the plan's reach.

**Recommendation**: pick up the v1 cyclic-elimination work *before*
implementing this plan, or commit to landing this plan in two phases:
(1) acyclic-only IPV traces with explicit RuntimeError on cyclic
graphs (small, ships); (2) cyclic-graph support after the v1 work.

### Forward references to other plans

- `trace-plan-v2.md` and commits `23119fe` (Stages 1/2/3/A0/A1) plus
  `153f603` (Stage A2) plus `5e0c15a` (`phasic.cache` module): all
  on the C side and complete. Stage A0 in particular makes the
  cache-invalidation table in this plan's "Cache impact" section
  *load-bearing on real behaviour* rather than aspirational.
- `disk-trace-cache.md` describes Stage A2 (now complete). The
  on-disk symbolic compute graph cache is keyed by graph content
  hash including IPV edge weights, so any `update_ipv` call that
  appends a new IPV edge or changes an existing IPV-edge weight
  invalidates that disk-cache key. The Python `_trace`/`_trace_dirty`
  on the `Graph` wrapper is preserved across IPV updates (when the
  trace was recorded with `parameterized_ipv=True`); only the
  C-side disk cache misses.
- `phasic.cache` module (`src/phasic/cache.py`): provides the
  user-facing `clear_param_compute_cache`, `param_compute_cache_info`,
  and `clear_all_caches` helpers. This plan does not call into it
  directly (the cache machinery is internal to the C runtime), but
  document it in the user-facing `update_ipv` docstring as the
  inspection/clearing surface if users want to verify cache state
  after IPV changes.

### Stage A1 interaction (important)

SVGD models built via `Graph.pmf_and_moments_from_graph` snapshot
the graph's JSON structure at model-construction time
(`src/phasic/__init__.py:5919`). The persistent `phasic::Graph`
inside `GraphBuilder` is built from that frozen JSON. So a user
who calls `graph.update_ipv(weights)` *after* constructing an
SVGD model will mutate the user's wrapper but **not** propagate
IPV changes into the SVGD model's `GraphBuilder`. This is
consistent with how `update_weights(theta)` works (theta is passed
through at call time, not snapshotted), but IPV is structural in
the FFI/pybind path — not a runtime parameter visible to
`GraphBuilder::build`. Users who want runtime IPV in SVGD should
call `update_ipv` *before* `pmf_and_moments_from_graph`.

The trace-based path
(`record_elimination_trace(parameterized_ipv=True)` +
`evaluate_trace_jax(trace, theta, ipv=...)`) is unaffected — the
trace is the abstraction that lets IPV flow as a runtime
parameter, and is what the user gets by going through
`cache_trace=True` rather than `pmf_and_moments_from_graph`.

## Summary

Add a new method `Graph.update_ipv(weights)` that takes a length-`graph.vertices_length()` array of edge weights from the starting vertex to every other vertex (with zeros at the starting-vertex index and at absorbing-vertex indices, and at least one entry > 0). The method:

1. Updates the in-graph IPV edges to match the supplied weights, so direct calls like `graph.pdf(...)` reflect the new IPV.
2. Stashes the IPV vector for use by trace-based evaluations.
3. **Does not invalidate any `EliminationTrace` recorded with the new flag `parameterized_ipv=True`** — the trace stores IPV as runtime PARAM ops, and `update_ipv` only changes the values consumed at evaluation time (mirroring how `update_weights(theta)` does not invalidate traces).
4. Issues a clear warning (and marks the in-process trace dirty) if a trace was recorded with `parameterized_ipv=False`.

The plan also adds a `parameterized_ipv` flag to `record_elimination_trace`, threads an `ipv` runtime parameter through `evaluate_trace_jax`, `instantiate_from_trace`, `trace_to_log_likelihood`, and the C++ codegen path, and preserves all existing JAX/SLURM compatibility.

## Context: why this is desirable

phasic's elimination trace lets parameterized graphs evaluate PDFs/likelihoods quickly across many `theta` values: record once (O(n³)), evaluate many times (O(n)) with `update_weights(theta)`. This is the engine of SVGD inference.

Today, the trace bakes IPV edge weights as `OpType.CONST` operations — so changing the IPV requires rebuilding the graph and rerecording the trace. SVGD pipelines that need to sweep over different starting distributions (or jointly infer IPV with rate parameters) pay an O(n³) re-record per IPV change.

(Stages A0/A1 amortise the O(n³) elimination across SVGD theta
calls within a single process, but the elimination still runs once
per fresh `record_elimination_trace` call, so rerecording on every
IPV change remains expensive when the user wants to sweep IPVs.
This plan eliminates that re-record by making IPV a runtime trace
parameter on equal footing with `theta`.)

The new design treats IPV as a second runtime parameter vector on equal footing with `theta`. The trace records IPV edges as `OpType.PARAM` references into an extended parameter vector `[theta, ipv, rewards]`. Trace evaluation reads IPV at runtime, so `update_ipv(weights)` is O(n_vertices), not O(re-record).

## Experimental verification of design assumptions

Three assumptions had to hold for this design to be sound. All were verified before writing this plan.

### Assumption 1: zero-weight starting-vertex edges are inert in the C forward algorithm

The C forward algorithm (Algorithm 4) is the runtime that `graph.pdf()` and `instantiate_from_trace`-built graphs eventually reach. If it crashed or produced wrong results for zero-weight IPV edges, the design would be untenable.

**Experiment**: Built two equivalent graphs — one with a single IPV edge `start → v1 (weight=1.0)`, one with that edge plus a `start → v_dead (weight=0.0)` edge (where `v_dead` has its own outgoing edge to keep it non-absorbing). Compared `pdf(t)` at six time points.

**Result**: bit-identical PDFs at every time point, no exceptions. Conclusion: zero-weight IPV edges are completely inert at the C runtime layer.

```
t=0.1: ref=0.653285180068  zero=0.653285180068  diff=0
t=0.5: ref=0.493560023230  zero=0.493560023230  diff=0
...
```

This means we do not need any C/C++ modifications to support zero IPV weights. The optional `if (w > 1e-12)` guard in the generated C++ is purely a cleanliness/performance optimization, not a correctness requirement.

### Assumption 2: in-place mutation of starting-vertex edges produces graphs equivalent to fresh builds

`update_ipv` is implemented in terms of two existing primitives that are exposed to Python:
- `edge.update_weight(w)` — updates an existing edge's weight in place; for constant edges (`coefficients_length == 1`) it also maintains the invariant `coefficients[0] == weight`.
- `start.add_edge(target, w)` — appends a new constant edge to the starting vertex.

There is no edge-removal API at any layer, so "removed" edges are left in the array with `weight = 0`. We need to confirm the resulting graph is indistinguishable from a freshly built one.

**Experiment**: Built reference graphs A1 with IPV `[0.4, 0.3, 0.0, 0.3]` and A2 with IPV `[0.0, 0.5, 0.5, 0.0]`. Built graph B with initial IPV `[1.0, 0, 0, 0]` and mutated it via `update_weight`/`add_edge` first to A1's IPV, then to A2's. Compared PDFs at every step.

**Result**: bit-identical PDFs to the reference graphs at every mutation. After the second mutation, `start.edges()` contains four edges (two with weight 0) — proving zero-weight edges remain in the array but are observationally inert.

```
After mutation IPV:
  -> [1]: weight=0.0   (zero — was nonzero in IPV1)
  -> [2]: weight=0.5
  -> [4]: weight=0.0   (zero — was nonzero in IPV1)
  -> [3]: weight=0.5   (newly appended)
```

### Assumption 3: parameterized-IPV trace evaluation works under JAX

The trace evaluator (`evaluate_trace_jax`) needs to handle `PARAM` ops referencing IPV slots in the extended parameter vector. The mathematical structure is `prob_j = ipv_j / Σ ipv_k`, evaluated via `INV` of `SUM` of `PARAM`s, multiplied by each `PARAM`. This must be JAX-traceable, vmap-compatible, jit-compatible, and produce finite gradients including when individual IPV slots are zero (only the sum must stay > 0).

**Experiment**: Hand-constructed a trace with three IPV slots, evaluated with both `[0.4, 0.3, 0.3]` and `[0.5, 0.0, 0.5]`, then took JAX gradients of `edge_probs[0][0]` w.r.t. IPV at each, and ran `jax.vmap` and `jax.jit` over a batch of three IPV vectors.

**Result**:
- All-nonzero IPV → `edge_probs[0] = [0.4, 0.3, 0.3]`, sum = 1.0. Correct.
- Zero IPV slot → `edge_probs[0] = [0.5, 0.0, 0.5]`, sum = 1.0. Correct.
- Gradient at `[0.4, 0.3, 0.3]` of `edge_probs[0][0]` → `[0.6, -0.4, -0.4]` — matches analytic `[d/d(ipv_j) of ipv_0/Σipv]`.
- Gradient at `[0.5, 0.0, 0.5]` → `[0.5, -0.5, -0.5]` — finite, no NaN.
- `jax.vmap` over batch of 3 IPVs → correct per-IPV results in one pass.
- `jax.jit` → works.

Conclusion: the design is JAX-clean and supports differentiation w.r.t. IPV (enabling joint SVGD inference of `theta` and IPV).

## Design

### IPV as a second runtime parameter vector

The extended parameter vector layout becomes:

```
extended_params = [theta_0, ..., theta_{P-1}, ipv_0, ..., ipv_{I-1}, reward_0, ..., reward_{R-1}]
                  └─── theta ───┘ └────── ipv ──────┘ └────── rewards ──────┘
```

with offsets `theta_offset = 0`, `ipv_offset = P`, `reward_offset = P + I`. The `ipv` segment has one slot per non-absorbing, non-starting vertex (call this the "eligible vertex set"). Zero values are honored: an IPV slot at zero contributes `0.0` to the sum and `0.0` to the corresponding edge probability, which `instantiate_from_trace` then drops via its existing `prob < 1e-12` filter (`trace_elimination.py:1352-1353`).

### Sizing the IPV vector — Choice B

Two reasonable choices for `ipv_length`:

- **Choice A**: only the IPV targets present at record time (`ipv_length = len(starting_vertex.edges())`). Mirrors `update_weights(theta)` exactly — you can change weights of recorded edges, not add new targets. Smallest extended vector.
- **Choice B**: every eligible (non-absorbing, non-starting) vertex (`ipv_length = number of eligible vertices`). Matches the user's stated API (length-`vertices_length()` weights with zeros at ineligible positions). Supports adding/removing IPV targets without rerecording.

**This plan adopts Choice B** because it matches the user-facing API (`update_ipv(weights)` with `len(weights) == vertices_length()`) and supports the more powerful "sweep over arbitrary IPVs" workflow. The cost is `O(n_vertices)` extra trace ops at record time — negligible against the `O(n³)` elimination work.

### `Graph.update_ipv(weights)` API

Insertion point: `src/phasic/__init__.py` after the existing `Graph.update_weights` method (which currently runs from line 2132 — insert immediately after; `update_weights` ends near line 2173). **Not** decorated with `@_invalidates_trace` (defined at `__init__.py:1574`) — the whole point is that the trace remains valid.

```python
def update_ipv(self, weights: ArrayLike) -> None:
    """Set IPV weights for trace evaluation and update in-graph IPV edges to match.

    Parameters
    ----------
    weights : array-like of length ``self.vertices_length()``
        Edge weights from the starting vertex to ``vertices()[i]``. Must be 0 at
        index 0 (the starting vertex itself) and at indices of absorbing
        vertices (vertices with no outgoing edges). Must be non-negative and
        finite. At least one entry must be > 0.

    Notes
    -----
    Unlike full graph mutation, this does NOT invalidate any
    ``EliminationTrace`` recorded with ``parameterized_ipv=True`` — the new
    IPV is read by trace evaluation as a runtime parameter alongside theta.
    Traces recorded with ``parameterized_ipv=False`` (the default) DO become
    stale; this method emits a warning and marks the in-process cached trace
    dirty in that case.

    The in-graph IPV edges are also updated (in place where possible, with
    new edges appended for previously-unused targets and "removed" edges
    left in the array with weight 0). This keeps direct calls to
    ``graph.pdf(...)`` consistent with trace-based evaluation. Repeated calls
    may grow ``starting_vertex().edges_length()`` by the number of distinct
    targets ever used; zero-weight edges are observationally inert.

    The disk-based trace cache is keyed on the C-level graph content hash,
    which includes IPV edge weights. After ``update_ipv``, a fresh
    ``record_elimination_trace`` call would miss the disk cache and re-record.
    The in-process cached trace, however, is preserved when recorded with
    ``parameterized_ipv=True``.
    """
```

#### Validation (raise `ValueError`/`TypeError` with specific messages — no silent fallbacks)

1. Coerce `weights` to a 1-D `np.ndarray[float64]`; reject non-array-likes with `TypeError`.
2. `len(weights) == self.vertices_length()` — else `ValueError("weights length L does not match vertices_length() = N")`.
3. `np.all(np.isfinite(weights))` — else `ValueError("weights contains non-finite values at indices [...]")`.
4. `np.all(weights >= 0)` — else `ValueError("weights must be non-negative; got negative values at indices [...]")`.
5. `weights[0] == 0` — else `ValueError("weights[0] must be 0 (the starting vertex cannot be its own IPV target)")`.
6. For each `i` where `weights[i] != 0`: check `self.vertex_at(i).edges_length() > 0` — else `ValueError("weights[{i}] is non-zero but vertices[{i}] is absorbing (no outgoing edges)")`.
7. `np.any(weights > 0)` — else `ValueError("at least one weight must be > 0 (IPV cannot be empty)")`.

#### Body

```python
# 1. Update in-graph IPV edges in place + append new ones
start = self.starting_vertex()
existing = {edge.to().index(): edge for edge in start.edges()}
vertices = list(self.vertices())  # cached for the call

for i, w in enumerate(weights):
    w = float(w)
    if i in existing:
        existing[i].update_weight(w)
    elif w > 0:
        start.add_edge(vertices[i], w)
    # else: no edge, w == 0 → nothing to do

# 2. Stash the IPV vector packed into the trace's eligible-slot order
self._ipv = self._pack_ipv_for_trace(weights)

# 3. Detect stale (non-parameterized) cached trace
if self._trace is not None and not getattr(self._trace, 'ipv_length', 0) > 0:
    import warnings
    warnings.warn(
        "Cached EliminationTrace was recorded with parameterized_ipv=False; "
        "it is now stale after update_ipv. Re-record with parameterized_ipv=True "
        "to enable IPV updates without re-recording.",
        UserWarning, stacklevel=2,
    )
    self._trace_dirty = True

# 4. Invalidate C++-side runtime contexts
self.notify_change()
```

`_pack_ipv_for_trace(weights)` selects the eligible (non-absorbing, non-starting) entries from `weights` in a fixed order matching the trace's `ipv_targets` field. The mask is structural — depends only on which vertices are absorbing — so it can be cached on the `Graph` wrapper.

#### Why not `@_invalidates_trace`?

Decorating with `@_invalidates_trace` would set `_trace_dirty = True` unconditionally, defeating the entire design. We invalidate the trace only when it was recorded with `parameterized_ipv=False` (handled explicitly above), and emit a warning so the user knows to re-record with the new flag.

### Trace recording changes

Add a `parameterized_ipv: bool = False` parameter to `record_elimination_trace` (default preserves existing behavior bit-for-bit).

When `parameterized_ipv=True`:

#### `EliminationTrace` dataclass additions (`trace_elimination.py:109-163`)

```python
ipv_length: int = 0           # Number of IPV slots in extended parameter vector
ipv_targets: list[int] = field(default_factory=list)  # vertex_index for each IPV slot
```

Both default to `0`/`[]`, preserving backward compat. The disk cache JSON deserializer uses dataclass defaults for missing fields, so existing on-disk traces still load.

#### Phase 1 — starting vertex rate (replaces `trace_elimination.py:596-619`)

When `parameterized_ipv=True`:

```python
if i == starting_vertex_idx:
    # Build IPV slot order: every non-absorbing non-starting vertex
    eligible_vertex_indices = [
        idx for idx, v in enumerate(vertices_list)
        if idx != starting_vertex_idx and v.edges_length() > 0
    ]
    ipv_targets = eligible_vertex_indices
    ipv_length = len(ipv_targets)

    # PARAM op for each IPV slot, indexing into extended_params at offset theta_dim + j
    ipv_param_ops = [builder.add_param(theta_dim + j) for j in range(ipv_length)]

    # Rate = 1 / sum(ipv_j)
    sum_op = builder.add_sum(ipv_param_ops)
    vertex_rates[i] = builder.add_inv(sum_op)
```

Stash `ipv_param_ops` and `ipv_targets` in a local for use in Phase 2.

#### Phase 2 — starting vertex edge probabilities (replaces `trace_elimination.py:638-664`)

When `parameterized_ipv=True`:

```python
if i == starting_vertex_idx:
    for j, target_idx in enumerate(ipv_targets):
        prob_op = builder.add_mul(ipv_param_ops[j], vertex_rates[i])

        # Reward transformation if enabled (offset shifts: theta_dim + ipv_length + target_idx)
        if reward_length > 0:
            reward_param_idx = theta_dim + ipv_length + target_idx
            reward_op = builder.add_param(reward_param_idx)
            prob_op = builder.add_mul(prob_op, reward_op)

        edge_probs[i].append(prob_op)
        vertex_targets[i].append(target_idx)
        edge_map[(i, target_idx)] = len(edge_probs[i]) - 1
    continue
```

#### Reward index offsets

Since IPV slots come between theta and rewards in the extended vector, every existing reward `add_param(theta_dim + i)` becomes `add_param(theta_dim + ipv_length + i)`. There are a few such sites in trace_elimination.py — search for `theta_dim + i` (or similar reward-offset patterns) and update.

#### `EliminationTrace` population at end of recording

```python
trace.ipv_length = ipv_length if parameterized_ipv else 0
trace.ipv_targets = ipv_targets if parameterized_ipv else []
```

### Trace evaluation changes

#### `evaluate_trace_jax` (`trace_elimination.py:1430`)

Current signature: `evaluate_trace_jax(trace, params, rewards=None, use_log=False) -> dict[str, Any]`.

Add an `ipv` parameter and concatenate it into the extended vector after `theta` and before `rewards`:

```python
def evaluate_trace_jax(trace, params, ipv=None, rewards=None, use_log=False):
    # Validation
    if trace.param_length > 0 and (params is None or len(params) != trace.param_length):
        raise ValueError(f"trace requires {trace.param_length} theta values")
    if trace.ipv_length > 0 and (ipv is None or len(ipv) != trace.ipv_length):
        raise ValueError(f"trace requires {trace.ipv_length} ipv values, got {None if ipv is None else len(ipv)}")
    # (existing reward validation...)

    # Build extended parameter vector: [theta, ipv, rewards]
    parts = []
    if trace.param_length > 0:
        parts.append(params)
    if trace.ipv_length > 0:
        parts.append(ipv)
    if trace.reward_length > 0:
        parts.append(rewards if rewards is not None else jnp.ones(trace.n_vertices))
    extended_params = jnp.concatenate(parts) if parts else jnp.array([])
    # (existing op-execution loop unchanged)
```

The op-execution loop (PARAM, DOT, MUL, etc.) is untouched — PARAM ops simply index into `extended_params` and don't care which segment a slot belongs to.

#### `evaluate_trace` (non-JAX Python path, `trace_elimination.py:956`)

Current signature: `evaluate_trace(trace, params=None, rewards=None, use_log=False) -> dict[str, Any]`.

Mirror the same change for the numpy path. Same signature `evaluate_trace(trace, params, ipv=None, rewards=None, use_log=False)`.

#### `instantiate_from_trace` (`trace_elimination.py:1324`)

Add `ipv` parameter, thread to `evaluate_trace`:

```python
def instantiate_from_trace(trace, params=None, rewards=None, ipv=None, use_log=False):
    result = evaluate_trace(trace, params, ipv=ipv, rewards=rewards, use_log=use_log)
    # (rest unchanged — the `if prob < 1e-12: continue` filter at
    # trace_elimination.py:1409 already handles zero-IPV-derived edges
    # correctly)
```

### `trace_to_log_likelihood` API (`trace_elimination.py:1618`)

Current signature: `trace_to_log_likelihood(trace, observed_data, reward_vector=None, granularity=0, use_cpp=True, use_log=False)`.

**Decision: pass IPV at function call time, not closure-baked.** Reasoning (from prior turn):

1. Symmetry with `theta`: IPV is a runtime parameter on equal footing with `theta`.
2. Joint SVGD inference of `(theta, ipv)` requires `ipv` to be a JAX-traceable input to `jax.grad`.
3. No statefulness in the LL function — its behavior is fully determined by inputs.
4. Single C++ binary per trace, reused across every IPV value (no per-IPV recompile).

The returned function's signature is determined by the trace's `ipv_length`:

- `trace.ipv_length == 0` → `log_lik(theta) -> scalar` (existing behavior).
- `trace.ipv_length > 0` → `log_lik(theta, ipv) -> scalar`.

If a user wants to fix the IPV, they write `partial(log_lik, ipv=my_ipv)` — explicit, no library magic.

### C++ codegen

Two layers in `__init__.py`:

#### `_generate_cpp_from_graph` (`__init__.py:694`)

(Plan originally referenced `_generate_cpp_graph_builder` at line 686-797; the function is named `_generate_cpp_from_graph` and starts at line 694 in current master.)

Add starting-vertex parameterized-IPV emissions analogous to the existing `start_param_edges` block. The generated C++ takes `(theta, ipv)` separately — or a single packed `params` vector with documented layout, whichever is cleaner. Concretely:

```cpp
// Parameterized IPV edges (weights are runtime ipv values)
for (size_t j = 0; j < n_ipv; j++) {
    double w = ipv[j];
    if (w > 0.0) {
        start->add_edge(*vertices[ipv_targets[j]], w);
    }
}
```

The `if (w > 0.0)` guard is **not** required for correctness (Experiment 1 proved zero-weight edges are inert), but is a clean optimization that avoids materializing useless edges in the rebuilt graph. It also matches the behavior of `instantiate_from_trace`'s `prob < 1e-12` filter.

#### `_generate_cpp_from_trace` (`__init__.py:814`)

**Signature change since this plan was written**: the function now
takes `observed_data` as a required parameter:

```python
def _generate_cpp_from_trace(trace, observed_data, granularity=0) -> str
```

The observations are *embedded* into the generated C++ as static
arrays (so the compiled binary contains both the trace and the data
it scores). When extending for IPV, do NOT remove `observed_data`;
instead add an IPV-handling branch alongside the existing data-
embedding logic. The generated `compute_log_likelihood` reads its
runtime input as `(theta[, ipv])` — the observed times are already
baked in.

Extend the embedded trace metadata:

```cpp
static const size_t IPV_LENGTH = {arrays['ipv_length']};
static const size_t ipv_targets[] = {format_array(arrays['ipv_targets'], 'size_t')};
```

Extend the generated `compute_log_likelihood` signature:

```cpp
double compute_log_likelihood(const double* theta, int n_theta,
                              const double* ipv, int n_ipv);
```

When `IPV_LENGTH == 0`, the generated wrapper accepts only `(theta, n_theta)` to preserve the existing API.

#### `_wrap_trace_log_likelihood_for_jax` (`__init__.py:1249`)

Currently uses `jax.pure_callback` with `vmap_method='sequential'` and a single `theta` argument. Extend to accept `(theta, ipv)`:

```python
def wrapped(theta, ipv):
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    return jax.pure_callback(
        lambda t, i: _call_compiled_lib(lib_path, np.asarray(t), np.asarray(i)),
        result_shape,
        theta, ipv,
        vmap_method='sequential',
    )
```

`pure_callback` handles multiple array arguments natively. `vmap_method='sequential'` continues to work for batched evaluation.

#### `trace_to_c_arrays` (`trace_elimination.py:1193`)

Emit `ipv_length` and `ipv_targets` so the codegen can read them.

## Cache impact

| Cache | `update_ipv` behavior | Why |
|---|---|---|
| Python `_trace` (cached on `Graph`) recorded with `parameterized_ipv=True` | **Preserved** | The whole point. IPV is consumed at evaluation time. |
| Python `_trace` recorded with `parameterized_ipv=False` | Marked dirty + warning | Trace baked old IPV as constants. |
| Python `_last_theta` | Untouched | IPV is orthogonal to theta. |
| C `reward_compute_graph` | Freed by `update_weight` and/or `add_edge` | Existing C primitive behavior. |
| C `parameterized_reward_compute_graph` | Freed only by `add_edge`; preserved on pure weight updates | Acceptable — its validity depends on graph structure. **Stage A0** (committed `23119fe`) made this load-bearing on real behaviour: `ptd_graph_update_weights` no longer destroys the symbolic compute graph, so SVGD theta sweeps amortise the O(n³) elimination across calls. `update_ipv` only invalidates the cache when the IPV mutation needs a `start.add_edge` (new target); pure weight updates on existing IPV edges preserve it. |
| C `reward_compute_graph_mpfr` | Freed on `add_edge` | Same. |
| C++ `ph_context`/`dph_context` | Cleared by explicit `notify_change()` call | Required for direct `graph.pdf()` consistency. |
| Disk-based trace cache (keyed on `ptd_graph_content_hash`) | **Self-invalidates** because hash includes IPV edge weights | The in-graph IPV mutation changes the hash. |
| Compiled `.so` cache (keyed on trace hash) | **Self-invalidates** for the new trace fields | New `ipv_length` enters the cache key — different binary per trace shape. |

The disk cache misses are wasteful: after `update_ipv`, calling `record_elimination_trace` again would recompute and recache from scratch. The in-process trace, however, is preserved — which is the win for SVGD pipelines that hold a `Graph` wrapper across many IPV updates.

A future enhancement (deferred — not in this plan) would modify `ptd_graph_content_hash` to exclude starting-vertex edge weights when a graph-level `parameterized_ipv` flag is set. Listed under "Future work" below.

## JAX / SLURM compatibility

The design preserves **all** existing JAX and distributed-computing capabilities:

- **`jax.jit`**: traces over `(theta, ipv)` the same way it traces `theta`. The trace operation list is static after recording; only array shapes/values flow.
- **`jax.grad`**: works on `ipv` automatically — SUM/MUL/INV ops are differentiable. Verified gradients to be analytically correct, including at zero-IPV slots.
- **`jax.vmap`**: vmapping over `ipv` works the same as over `theta`. Verified with a batch of three IPV vectors.
- **`jax.pmap` / sharding**: pmap operates at the JAX-array level. It doesn't care whether a function takes one runtime array or two. Users shard `(theta, ipv)` the same way they shard `theta` today.
- **Multi-node SLURM**: each worker constructs/loads the same `EliminationTrace` (the new `ipv_length`/`ipv_targets` fields are JSON-serializable plain int / list-of-int) and computes on its shard. The compiled `.so` cache is keyed on a hash that now includes `ipv_length`, so traces with and without parameterized IPV get distinct entries — no collision on shared filesystems.

### Two design constraints to honor during implementation

These must not drift during implementation:

1. **No data-dependent control flow inside `evaluate_trace_jax`.** Specifically: do not write `if ipv[j] == 0: skip` inside the trace evaluation loop. Zero IPV weights must flow through JAX as `0.0`, producing `0.0` probabilities. The "drop zero edges" logic stays in `instantiate_from_trace` (pre-JAX) and in the C++ codegen (also pre-JAX). Inside JAX, every operation is traced unconditionally on every IPV slot.
2. **Generated C++ must take `ipv` as a runtime pointer, not embed it as constants.** Otherwise we'd recompile per IPV value, blowing the `.so` cache.

## Backward compatibility

- Default `parameterized_ipv=False` for `record_elimination_trace`, preserving existing behavior bit-for-bit.
- `EliminationTrace.ipv_length` and `EliminationTrace.ipv_targets` default to `0`/`[]`. Existing serialized traces in `~/.phasic_cache/traces/` deserialize correctly (missing fields → defaults).
- All existing call sites of `evaluate_trace_jax(trace, theta)` continue to work; the new `ipv=None` parameter is optional and only required when `trace.ipv_length > 0`.
- `trace_to_log_likelihood(trace, observations)` with a default-recorded trace continues to return `log_lik(theta)`. Only traces with `parameterized_ipv=True` produce the `log_lik(theta, ipv)` signature.

## Implementation steps

A reasonable order for picking this up later:

1. **Add fields to `EliminationTrace` dataclass** (`trace_elimination.py:109-163`). Add `ipv_length: int = 0` and `ipv_targets: list[int] = field(default_factory=list)`. Verify existing tests pass.
2. **Add `parameterized_ipv` flag to `record_elimination_trace`**. Implement Phase-1 and Phase-2 changes for the starting vertex when the flag is True. Update reward index offsets to account for the IPV segment. Add a unit test that records a trace with `parameterized_ipv=True` and inspects `ipv_length`, `ipv_targets`, and the operation list.
3. **Extend `evaluate_trace_jax` and `evaluate_trace`** to accept `ipv` and concatenate it into the extended parameter vector. Add tests that record a trace with `parameterized_ipv=True`, evaluate at several IPV values, and compare to `instantiate_from_trace` + direct `pdf`.
4. **Extend `instantiate_from_trace`** to accept and thread `ipv`. Add a test that confirms zero-IPV slots are dropped (the `prob < 1e-12` filter).
5. **Implement `Graph.update_ipv(weights)`** in `__init__.py`. Add the `_pack_ipv_for_trace` helper and `_ipv_eligible_mask` cache. Add tests:
   - Validation errors fire correctly (parametrize over each rule).
   - `update_ipv` then `pdf()` matches a freshly built graph (uses Experiment 2 evidence).
   - `update_ipv` does not mark `_trace_dirty` when trace was recorded with `parameterized_ipv=True`.
   - `update_ipv` warns and marks `_trace_dirty` when trace was recorded without IPV parameterization.
   - Repeated `update_ipv` calls leave `start.edges()` correctly updated.
6. **Extend `trace_to_log_likelihood`** to inspect `trace.ipv_length` and return either `log_lik(theta)` or `log_lik(theta, ipv)`. Add tests for both signatures.
7. **Extend C++ codegen** (`_generate_cpp_from_graph`, `_generate_cpp_from_trace`, `_wrap_trace_log_likelihood_for_jax`, `trace_to_c_arrays`). Add tests that record with `parameterized_ipv=True`, generate C++, compile, and call from Python — comparing results to the Python evaluation path.
8. **End-to-end SVGD test**: parameterized-IPV trace + sweep over IPVs without re-recording. Confirm trace is reused (check `_trace_dirty` stays False; check on-disk cache hit count if accessible).
9. **JAX compatibility tests**: `jit`, `vmap`, `grad` over IPV all produce expected results.

## Critical files

(Line numbers below reflect master at commit `23119fe`. Re-verify
on next read with `grep -n` since unrelated edits will drift them.)

- **`src/phasic/trace_elimination.py`**
  - `EliminationTrace` dataclass at line 109 — add `ipv_length`, `ipv_targets`.
  - `record_elimination_trace` (Phase 1 at line 596–619, Phase 2 at line 638–664) — branch on `parameterized_ipv`. Update reward-offset arithmetic throughout.
  - `evaluate_trace_jax` (line 1430) and `evaluate_trace` (line 956) — accept `ipv`, build extended params from `[theta, ipv, rewards]`.
  - `instantiate_from_trace` (line 1324; zero-prob filter at line 1409) — thread `ipv` through.
  - `trace_to_log_likelihood` (line 1618) — branch on `trace.ipv_length` for return signature.
  - `trace_to_c_arrays` (line 1193) — emit `ipv_length`, `ipv_targets`.

- **`src/phasic/__init__.py`**
  - Add `Graph.update_ipv(weights)` after `Graph.update_weights` (currently ends near line 2173). **Do NOT** decorate with `@_invalidates_trace` (decorator at line 1574).
  - Add `Graph._pack_ipv_for_trace` and `Graph._ipv_eligible_mask` helpers.
  - `_generate_cpp_from_graph` (line 694) — emit parameterized-IPV `add_edge` calls, optionally guarded by `if (w > 0.0)`. *(Plan originally referred to this as `_generate_cpp_graph_builder`; that name does not exist in master.)*
  - `_generate_cpp_from_trace` (line 814) — extend embedded metadata (`IPV_LENGTH`, `ipv_targets`); extend generated `compute_log_likelihood` signature. **Note**: this function now takes `observed_data` as a required parameter; the IPV extension must work alongside the embedded-data path, not replace it.
  - `_wrap_trace_log_likelihood_for_jax` (line 1249) — pass `(theta, ipv)` through `jax.pure_callback`.

- **No edits to `src/c/phasic.c`, `src/cpp/phasiccpp.cpp`, or `src/cpp/phasic_pybind.cpp`** — all required primitives (`Edge.update_weight`, `Vertex.add_edge`, `Graph.notify_change`) are already exposed.

## Verification plan

Tests in `tests/pytest/test_update_ipv.py` (new file) and additions to `tests/pytest/test_trace_*.py`. Run:

```
pixi run -- pytest tests/pytest/test_update_ipv.py -x -v
pixi run -- pytest tests/pytest/ -k "trace or pdf or expectation or svgd" -x
```

Tests:

1. **Backwards compat: existing traces unaffected.** Record a parameterized graph with default `parameterized_ipv=False`; evaluate; PDF must match a stored fixture from before the change. Confirms the default code path is bit-stable.
2. **`update_ipv` matches a freshly built graph (continuous and discrete).** Build graph A with one IPV; record with `parameterized_ipv=True`. Build graph B with a different IPV; call `B.update_ipv(weights_for_A)`. Compute PDF via trace evaluation on B's trace + IPV. Compare to direct `graph.pdf` on a fresh A. Must match to machine precision. Repeat for `is_discrete=True`.
3. **Trace remains valid across many `update_ipv` calls.** Record once with `parameterized_ipv=True`; sweep over 10 different IPVs; for each, evaluate trace and compare to `instantiate_from_trace` + `pdf`. All must match. Confirm `graph._trace_dirty` stays False throughout.
4. **Zero IPV weights are dropped before reaching the C forward algorithm.** Set IPV with several zero entries; confirm the `instantiate_from_trace`-built graph has only the nonzero edges from the starting vertex. Inspect generated C++ and confirm no `add_edge` call materializes for zero entries (or, if the `if (w > 0.0)` guard is implemented, that it correctly skips).
5. **Validation errors fire correctly.** Parametrize over each rule (length mismatch, NaN, inf, negative, weights[0] != 0, weights[absorbing] != 0, all-zero, non-array-like input).
6. **Trace recorded without `parameterized_ipv` is detected as stale on `update_ipv`.** Build, record with default flag, `update_ipv(...)`. Confirm a `UserWarning` is emitted and `_trace_dirty == True`. Subsequent `graph.pdf()` (which uses the in-graph mutated edges, not the trace) must still be correct.
7. **SVGD round-trip with parameterized IPV.** Set up a simple two-parameter coalescent model with a parameterized IPV. Run SVGD jointly inferring `theta` and IPV. Confirm convergence within usual error bounds. Confirm the trace is reused across iterations (no re-record).
8. **Non-IPV edges untouched after `update_ipv`.** Build, snapshot every non-IPV edge `(target, weight, coefficients)`, `update_ipv(...)`, re-walk and assert bit-identical.
9. **In-graph IPV edges reflect `update_ipv`.** After `update_ipv(weights)`, walking `start.edges()` should produce edges whose weights match `weights[i]` for each `i` with an existing or newly-added edge. Edges to indices that went from non-zero to zero remain in the array with weight 0.
10. **JAX compatibility.** `jit`, `vmap` over `ipv`, `grad` w.r.t. `ipv` — all produce expected results. Use the analytical gradient check from Experiment 3 as a baseline.
11. **C++ codegen path matches Python path.** Record with `parameterized_ipv=True`; generate C++; compile; call from Python via `_wrap_trace_log_likelihood_for_jax` with `(theta, ipv)`; compare to Python-mode `evaluate_trace_jax`. Must match to machine precision.

## Open questions / risks

1. **Disk trace cache invalidation on every `update_ipv`.** For SVGD users this is fine — the trace is reused in-process. For users who serialize traces to disk and want to vary IPV across runs, they pay an O(n³) re-record per IPV change. Acceptable for v1; addressable later via a graph-level `parameterized_ipv` flag that excludes IPV weights from the content hash.
2. **`update_ipv` fails loudly on parameterized IPV edges.** If a user already constructed parameterized IPV via `start.add_edge_parameterized(target, 0, [c1, c2])`, the in-place `edge.update_weight()` call would silently break the parameterization invariant. Implementation should detect this case (any starting-vertex edge with `coefficients_length() > 1`) and raise `ValueError("update_ipv does not support graphs with parameterized IPV edges; use start.add_edge_parameterized() directly to update them")`. Add a test for this.
3. **Audit `cache_trace=True` consumers.** The `Graph._trace`/`_trace_dirty` machinery is consulted in several `Graph` wrapper methods. Audit those to confirm none of them assumes a stored trace's IPV is constant in a way that conflicts with the new design. Likely safe — most consumers go through `_ensure_trace` which already respects `_trace_dirty` — but worth a grep.

## Future work (deferred, not in this plan)

- **Graph-level `parameterized_ipv` flag** that excludes starting-vertex edge weights from `ptd_graph_content_hash`. Would prevent disk-cache misses on `update_ipv`. Requires a small C-side change in `phasic_hash.c`.
- **Differentiable `update_ipv` via SVGD on IPV alone.** With the design above, this is already supported through `trace_to_log_likelihood` returning `log_lik(theta, ipv)`. A higher-level `Graph.svgd(infer_ipv=True, ...)` convenience wrapper could be added.
- **Parameterized IPV with coefficient vectors** (analogous to non-IPV parameterized edges, where `weight = c · θ`). Would require a different design — out of scope for this plan, which targets the simpler "IPV weights as a runtime vector" case.
