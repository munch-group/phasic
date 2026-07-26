# Plan — consistent reward-length handling (and the discrete-graph inconsistency behind it)

Branch: `fix/is-discrete-propagation`. N1/N2/variance_discrete/moments_discrete already landed.
Decision taken: **keep the `vertices_length()` convention** (not transient-only). This plan was
**adversarially reviewed** (4 agents, code-grounded); their corrections are folded in and marked
`[adv]`. **Not yet implemented.**

## 1. The finding (grounded)

The reward-length convention is **not** ambiguous. The canonical length is **`vertices_length()`**
(unambiguous: `== len(graph.vertices()) == serialized n_vertices`; the 2D validator uses it for its
vertex axis; `discretize()` *raises* it by the number of aux vertices, so the reference is always the
**current** graph's `vertices_length()`). It is enforced by `reward_transform`,
`reward_transform_discrete`, `expectation`, `variance`, `moments`, `sample`, discrete `covariance`,
and `_validate_rewards` — all **reject** a wrong length. `"one per vertex"` is documented in 5
places; **`"excluding start"` appears only in one test comment** (`test_rewards_support.py:68`) — it
is a misconception, not a convention. The C `ptd_graph_reward_transform` / `ptd_expected_waiting_time`
read exactly `vertices_length` entries from a raw pointer.

**Unenforced sites (all read `vertices_length` entries with no length check → truncate-if-long /
OOB-if-short):**
- `pmf_and_moments_from_graph` and `..._multivariate` (the JAX factories).
- `[adv, Agent 1]` `Graph.expected_waiting_time(rewards)` — the shared primitive under
  moments/variance/covariance; unguarded when called directly.
- `[adv, Agent 1]` `Graph.covariance(rewards1, rewards2)` **continuous branch** (`_covariance`);
  the discrete branch accidentally validates via `_moments`.

**Corrections to the earlier draft `[adv, Agents 2 & 4]`:**
- `test_rewards_support.py::test_rewards_transformation` uses length **V−1** (OOB read). It is
  numerically *correct here only* because the single dropped vertex is the **absorbing** one
  (edgeless; its reward is force-set to 1 in `phasic.c:6196`). The OOB becomes non-deterministic
  garbage only when a **transient** vertex is dropped (reproduced: 9/10 runs → `[0,0]`, 1/10 →
  `[0.5,0.5]`).
- `test_multivariate.py` uses length **V+1** (`n_vertices=4` hard-coded on a V=3 graph) → **silent
  truncation**, a *different* failure mode. The earlier "both use V−1" statement was wrong.
- Caller-scope is otherwise **complete** `[adv, Agent 4]`: SVGD pre-validates rewards eagerly
  (`__init__.py:5538`), `method_of_moments`/`probability_matching` pass no rewards, `pmf_from_graph`/
  daisy/`joint_index` ignore or forbid them, and every notebook `states().T` idiom resolves to
  `vertices_length()`.

## 2. B1 — enforce `vertices_length()` on every unenforced reward path

**Mechanics (all `[adv, Agent 4]`):**
- **`.shape`-only check, NOT `_validate_rewards`.** `_validate_rewards` calls `np.asarray(rewards)`
  + a graph BFS → `TracerArrayConversionError` under `jit`/`vmap`. The JAX model is advertised
  jit/grad/vmap/pmap-safe, so the guard must compare `rewards.shape[-1]` (and `ndim`) to
  `vertices_length()` only — reuse `_validate_rewards`'s **count + message text**, not its body.
- **Placement across the `custom_vjp`.** `pmf_and_moments_from_graph` builds three `_compute_pure`
  closures (callback `:6608`, FFI `:6770`, pybind `:6830`) and a `custom_vjp` (`model`/`model_fwd`/
  `model_bwd`, `:6999+`). `jax.grad` (SVGD) runs **fwd/bwd, not the primal** — so put the check at
  the **top of each `_compute_pure`** (covers primal+fwd+bwd), and add an explicit up-front check in
  `_multivariate` (it slices `rewards_arr[j,:]` before calling `model_1d`).
- **1D vs 2D.** 1D factory validates 1D only; 2D `(n_features, n_vertices)` lives only in
  `_multivariate`. **Reconcile the orientation clash first** `[adv, G5]`: `_validate_rewards`
  enforces `(n_features, n_vertices)` but `compute_pmf_and_moments_ffi` (`ffi_wrappers.py:749`) and
  the pybind `_compute_pure` output logic (`__init__.py:6836`, `n_features=shape[1]`) assume
  `(n_vertices, n_features)`. Pick one orientation, assert it, and make the axis meaning consistent.
- **C++ backstop in BOTH engines** `[adv, G6]`: `graph_builder.cpp` (`compute_pmf_and_moments`,
  which derives `n_vertices` from the reward shape at `:730`) **and** `graph_builder_ffi.cpp`
  (`ComputePmfAndMoments`/`Multivariate`, which derive `n_rewards` from the buffer with no compare to
  the graph). Validate against the builder's `n_vertices_` (= `vertices_length()`), raise loudly.
- **The shared primitive.** Guard `Graph::expected_waiting_time(std::vector<double>)` and the
  continuous `_covariance` at the C++ wrapper (the vector size is known there) so a direct call
  can't OOB either. This is the single highest-leverage guard (it backs moments/variance/covariance).

**Tests/docs to fix (`[adv]`):** `test_rewards_support.py` V−1 → V; `test_multivariate.py` V+1 → V
(3, not 4); correct the `"# One per vertex"` len-4 docstring examples at `__init__.py:6537, 7531,
7537`; delete the `"excluding start"` comment. Verified safe: at length V the assertions hold with
identical numbers.

**New guard tests:** wrong length (short AND long, 1D AND 2D) → clean `ValueError`, on the eager,
`jax.jit(model, rewards-as-traced-arg)`, and `jax.grad` paths (not just eager); a direct
`expected_waiting_time`/continuous-`covariance` wrong-length → raises.

## 3. B2 — fix the `was_dph` / native-DPH normalisation inconsistency

Reproduced `[adv, Agent 3]`: a **native DPH** (is_discrete + was_dph set) has `update_weights`
auto-normalise each vertex's single out-edge to 1.0 → collapses to a deterministic walk
(`moments → [2,6,24]` vs true `[7,71,991]`). A **`discretize()`'d graph genuinely needs** was_dph
(its vertices have multiple out-edges/self-loops; without normalisation the outgoing rate stays >1
and `pdf_discrete` **throws**). `serialize()` carries `is_discrete` but not `was_dph`, and the
current `from_serialized` latches `was_dph=True` from `is_discrete` — which **corrupts a
round-tripped native DPH** (moments `[7,71,991]` → `[2,4,8]`; reproduced). was_dph is **not read by
the FFI/parameterized path** (inference numerics untouched); it only drives direct `update_weights`
normalisation and the graph hash.

**Fix, with the review's corrections:**
- **Serialize via `self.get_was_dph()`** — NOT `getattr(self,'was_dph',…)` `[adv, G3]`: there is no
  Python `was_dph` attribute (`getattr → MISSING`); the flag is C-level. `get_was_dph()` exists on
  the raw base `_Graph`, so it is SLURM-raw-safe.
- **`from_serialized` absent-key default = `is_discrete`, NOT `False`** `[adv, G2 / Agent 3]`:
  `data.get('was_dph', data.get('is_discrete', False))`. `graph_cache` keys on `callback_hash`, so
  pre-B2 on-disk / mixed-version SLURM payloads lack the `was_dph` key; a `False` default would flip
  cached **discretized** graphs to `was_dph=False` → invalid DPH. The `is_discrete` fallback
  replicates today's latch for old caches while new caches carry the explicit flag.
- **Drop the unconditional `set_was_dph(True)`** I added in Batch 2; restore faithfully instead.
- Benign: restoring `was_dph=False` for a native DPH changes its graph hash → pre-fix cached traces
  become cache **misses** (recompute, not incorrect).

**Gate test must be CROSS-VERSION** `[adv, Agent 3 & G2]`: a same-session round-trip won't catch the
regression (fresh serialize includes was_dph). Construct a pre-B2 payload (dict **without** the
`was_dph` key, `is_discrete=True`) for a discretized graph and assert `from_serialized` keeps it a
valid DPH; plus a native-DPH round-trip (is_discrete, was_dph=False) that stays un-normalised.

## 4. Design decision — RESOLVED

Keep `vertices_length()` (all vertices incl. start/absorbing; a couple of entries are "don't care").
Transient-only convention shelved as a separate future UX change.

## 5. Batches (test-gated) — ALL DONE

- **B1b** — DONE (`bd751e03`). Jit/grad/vmap-safe `.shape`-only guard (`_check_rewards_len`) at the
  top of each of the three `_compute_pure` closures; multivariate factory inherits it via
  `model_1d`. Reconciled the 2D orientation on the canonical `(n_features, n_vertices)`: SVGD's
  model-validation slice `self.rewards[:, :test_times.shape[1]]` corrupted the vertex axis (silent
  OOB; loose 2D SVGD tests passed on garbage) — now passes the reward whole; fixed the misleading
  doc. Fixed mis-lengthed tests (`test_rewards_support` V−1→V, `test_multivariate` V+1→V). Verified
  clean `ValueError` under eager/jit/grad.
- **B1a** — DONE (`66a5976d`). Guarded the shared C++ primitives `expected_waiting_time`,
  `reward_transform[_p]`, `dph_reward_transform[_p]` — higher leverage than per-handler checks:
  they transitively cover pybind `compute_pmf_and_moments`, every FFI handler, and continuous
  `_covariance` (its `expected_waiting_time` calls precede the `rewards[i]` loop). FFI multivariate
  extracts the reward buffer's own vertex dim (in-bounds); the length-vs-graph mismatch is caught by
  the guarded `reward_transform`.
- **B2** — DONE (`3cb3846f`). serialize `was_dph` via `get_was_dph()`; `from_serialized` faithful
  restore, absent-key default `is_discrete`; dropped the unconditional latch. Native-DPH round-trip
  stays un-normalised; discretize round-trip stays valid; cross-version pre-B2 payload validated.

## 6. Follow-ups discovered during implementation (separate, out of scope)

- **G5 residue (FFI multivariate orientation).** `ComputePmfMultivariateFfiImpl` reads a 2D reward
  as `(n_vertices, n_features)` (`graph_builder_ffi.cpp:613`), opposite to the canonical
  `(n_features, n_vertices)`. The `use_ffi=True` multivariate path is off the default; the primitive
  guard still prevents OOB. Reconcile if that path is promoted.
- **Parameterized-`discretize()` round-trip fidelity.** A *parameterized* discretize()'d graph does
  not fully round-trip through `serialize` + re-`update_weights` (E[N] off by ~1). Structural/param
  gap, unrelated to `was_dph`; affects caching of such graphs.
