# Plan — constant graphs come back parameterized after serialize/load

Status: v2, 2026-10-11 — two adversarial reviewers have been through it.
Both said SOUND-WITH-CORRECTIONS, and their corrections are folded in
below (record in §10). No code changes are authorized by this document.
The fix modifies existing code (`serialize()` and `graph_cache.py`), so it
needs Kasper's sign-off, including the two decisions in §11.

## 1. Problem

A graph whose `parameterized()` is False reports `parameterized() == True`
after a round trip through `Graph.serialize()` → `Graph.from_serialized()`.
The on-disk graph cache, which is on by default, makes exactly this round
trip:

```python
graph = Graph(coalescent)          # callback returns float rates
graph.parameterized()              # False on a cold cache, True once cached
```

This was found while writing tests from the tutorial notebooks: the answer
depends on whether `~/.phasic_cache/graphs/` already holds the graph.

**Affected graphs.** Every graph that has `parameterized() == False` and
`param_length() >= 1`. There are two kinds:
- **CONSTANT edge mode.** Built with scalar `add_edge(v, rate)`, including
  graphs from a callback that returns floats, `discretize()` of such a
  graph, and native DPH graphs.
- **UNLOCKED edge mode.** Graphs that C code builds without ever locking a
  mode, which leaves `parameterized()` False. Measured: `reward_transform()`
  of a 1-parameter or 2-parameter graph, and `laplace_transform()` of a
  constant graph. These are effectively constant already, since their own
  `update_weights` raises "empty graph".

## 2. Root cause (read from source; confirmed independently by reviewer A)

1. A constant edge is stored as a one-element coefficient array holding the
   rate (`src/cpp/phasiccpp.cpp:1490`, `ptd_graph_add_edge(..., &coeff, 1)`).
2. The C layer sets `param_length` from the coefficient length of the first
   edge that does not leave the starting vertex (`src/c/phasic.c:4876-4881`).
   So a constant graph has `param_length() == 1`. What tells it apart from a
   genuine 1-parameter graph is the `parameterized` flag, which `edge_mode`
   sets (`src/c/phasic.c:4886-4890`).
3. `serialize()` defaults `theta_dim = self.param_length()`
   (`src/phasic/_graph_serialize.py:64-65`). Whenever `theta_dim > 0` it
   exports every edge that carries a coefficient array as a *parameterized*
   edge (`_graph_serialize.py:75`). It never consults `parameterized()`.
4. `from_serialized()` re-adds `param_edges` with the array syntax
   `add_edge(to, [coeffs])` (`_graph_serialize.py:562`), which locks the
   rebuilt graph to PARAMETERIZED.
5. The graph cache stores `graph.serialize()` (`src/phasic/graph_cache.py:133`)
   and loads it via `Graph.from_serialized` (`graph_cache.py:212`).

## 3. Evidence

Measured on the installed build, with no source edits. Scripts are in the
session scratchpad (`coal_param.py`, `derisk_ser.py`, `derisk_disc.py`;
the reviewers' scripts are in `review_A/` and `review_B/`). Reviewer A
reproduced the table exactly.

| Graph | Today, default `serialize()` | With the fix (`theta_dim=0`) |
|---|---|---|
| Constant coalescent, 4 samples, from a callback | Reloads with `parameterized=True`; values agree to 4.4e-16. `update_weights([2.0])` is ACCEPTED and halves the mean (1.5 → 0.75), whereas the freshly built graph refuses it. | Reloads with `parameterized=False`; values equal |
| The same graph after `discretize(0.1)` (`was_dph=True`) | **`pdf()` RAISES** "outgoing rate <= 1 … Is 6.1" | Reloads constant; `is_discrete` and `was_dph` preserved; `pdf()` equal |
| Native DPH constant graph | Reloads with `parameterized=True`; pdf equal | Reloads constant; pdf equal |
| `reward_transform()` of a parameterized graph (UNLOCKED) | Reloads parameterized; `update_weights([2.0])` scales rates the transform had already fixed (difference 2.5) | Reloads constant (CONSTANT mode, so the `update_weights` error message changes from "empty graph" to "constant graph") |
| Genuine 1-parameter and 2-parameter graphs | — | Byte-identical serialization (by construction; measured) |

"Equal" means agreement to round-off: differences of 4.4e-16 occur, for
example clone versus reload. Tests compare with `rtol ≈ 1e-14`, not exactly.

The defect is therefore more than a wrong flag. A cached constant graph
silently becomes a model with one global rate multiplier θ, and a
discretized constant graph is broken outright. The existing test
(`tests/pytest/test_graph_cache.py::test_simple_graph_round_trip`) misses
it: it passes `theta_dim=0` explicitly and checks only vertex counts.

## 4. The fix

### 4a. Serialization (one line, root cause)

In `serialize()` (`src/phasic/_graph_serialize.py:64-65`):

```python
if theta_dim is None:
    theta_dim = self.param_length() if self.parameterized() else 0
```

- `from_serialized()` already rebuilds a `theta_dim=0` serialization
  correctly. Reviewer A verified edges, constant_edges, start_edges,
  is_discrete, was_dph, dyn_ordering, and the duplicated all-zero aux
  states created by discretize.
- Parameterized graphs are byte-identical, because the branch differs only
  when `parameterized()` is False.
- An explicit `theta_dim=` still wins.
- The SLURM shim (`distributed_scc.py:74`) is unaffected: synthetic SCC
  graphs are always marked parameterized (`scc_synthetic.c:989-990`), even
  when the parent graph is constant.
- No graph was found that has `parameterized()==False` and still needs
  `param_edges`. Both reviewers searched: start-vertex-only parameterized
  edges, `set_param_length(1)` / `(2)`, every weight mode, clone, copy,
  discretize, the joint_* graph builders, formula mode, and the callback
  builder with `theta_dim=`.

Two stale texts go in the same change:
- The comment at `src/phasic/__init__.py:3617-3621`, which claims
  `serialized['param_length']` is ≥ 1 even for constant graphs.
- The assertion at `tests/pytest/inference/_demo_unified_edge_correctness.py:194`,
  which expects `param_length==1` for a constant graph. Pytest does not
  collect that file, but it fails if run.

### 4b. Stale cache entries — REVISED in v2 (decision D1 in §11)

Entries written before the fix hold the bad layout and would keep loading
as parameterized. Only entries whose stored `param_length == 1` can be
affected: under the old code, a constant graph and a genuine 1-parameter
graph serialize identically, while entries with `param_length` 0 or ≥ 2 are
certainly correct.

**Recommended (v2): a format marker in `graph_cache.py`.**
- `save_graph` writes `'format': 2` into the cache entry.
- `load_graph` treats an entry that has no marker AND has
  `graph_data['param_length'] == 1` as a cache miss, logging at INFO
  "stale pre-format-2 entry, rebuilding".
- The normal save path then rebuilds the graph and overwrites the file
  under the same key.
- Genuine multi-parameter entries, which are the expensive ones at the
  200k-600k scale target, are kept. No files are orphaned.

**Alternative: bump `PHASIC_CALLBACK_VERSION`** (`callback_hash.py:28`,
"1.0" → "1.1"). Simpler, but it rebuilds every cached graph once. It also
orphans the old files: `clear_from_cache` (`_graph_cache_mgmt.py:86-91`)
computes only the new key, so the old files are removed only by
`clear_graph_cache`'s glob.

## 5. Behaviour changes beyond the cache (corrected in v2)

v1 wrongly listed `__init__.py:3615/7403/8556` as "explicit theta_dim,
unchanged". They forward their own `theta_dim` argument, whose default is
`None`, so they reach the new branch (confirmed in source).

The table lists every default-`serialize()` site. "Before" and "after"
describe what happens when the site receives a constant graph. Parameterized
graphs are unchanged everywhere.

| Site | Before the fix | After the fix | Evidence |
|---|---|---|---|
| `graph_cache.py:133` | comes back parameterized | comes back constant (the fix) | measured |
| `__init__.py:7094` `moments_from_graph` | fake 1-parameter model | existing `ValueError` "Graph must have parameterized edges" | measured (both reviewers) |
| `__init__.py:7403` `pmf_and_moments_from_graph` | **silently θ-scaled**: through FFI at θ=2 the mean is 0.75, but the true mean is 1.5 | existing `ValueError`, for both `use_ffi` settings | measured (B) |
| `__init__.py:3615` `pmf_from_graph` | generated C++ writes `rate*theta[0]`, then the non-parameterized wrapper passes a dummy θ=[0.0] (`__init__.py:3836-3840`), which likely zeroes every rate | generated code uses the constant weights directly, so this is likely a REPAIR | codegen measured (B); not run end to end, because the JIT path needs the C/C++ source tree |
| `__init__.py:8556` `pmf_from_graph_joint_index` | joint graphs are always parameterized | unchanged | measured (B) |
| `__init__.py:3375` `reward_visit_probability`, JAX θ | **correct**: scaling every rate by one θ leaves hitting probabilities unchanged | `XlaRuntimeError` "Theta length mismatch: expected 0, got 1" (`graph_builder.cpp:196`). The error is loud, and `theta=jnp.zeros(0)` returns the correct value. Decision D2 | measured (A, B) |
| `zero_inflation.py:147` | same as the row above | same error; probably unreachable, because `pmf_and_moments_from_graph` raises first (not verified) | source (B) |
| `bffg.py:313`/`:487` | already raises, in `update_weights` (`bffg.py:316`) | unchanged | source (B) |
| `ffi_wrappers.py:142` | dead code (no callers) | none | source (B) |
| `__init__.py:527` | inside the disabled `pmf_from_graph_parameterized` | none | source |
| `distributed_scc.py:74` | synthetic graphs are parameterized | unchanged | measured (A, B) |

No consumer gives a silently wrong answer after the change. The C++
GraphBuilder accepts `param_length 0` with a non-empty `edges` array
(measured). No tutorial notebook cell feeds a constant graph to any of
these paths (B).

**Hash-keyed artefacts.** `compute_graph_hash` includes both the
`parameterized` flag and `param_length` (`phasic_hash.c:237-243`).
- Today a constant graph loaded from a warm cache hashes as parameterized.
  After the fix it hashes the same whether the cache is warm or cold.
- Stage-A2 / registry artefacts made from warm-cache constant graphs become
  unreachable orphans.
- Nothing is poisoned, because the key and the content change together.

## 6. Out of scope (recorded, not fixed)

- `from_serialized()` restores structure, not the current θ. A
  parameterized graph reloads with weights at θ = 1 until `update_weights`
  is called. Pre-existing, and apparently deliberate.
- **Paths that still treat a constant graph as θ-scaled** (B). The fix does
  NOT make all of them loud:
  - `Graph.svgd` on a constant graph runs and fits a θ multiplier, both
    before and after the fix (measured).
  - `probability_matching` defaults `theta_dim = graph.param_length()`
    itself (`probability_matching.py:196-197`).
  - `daisy_chain_joint_probs` keeps the old layout (`__init__.py:11175`).

  These are candidates for a separate "constant graph passed to an
  inference path" guard.
- `hierarchical_trace_cache.py:254` and `:334` call
  `serialize(theta_dim=graph.param_length())` explicitly, which is 1 for a
  constant graph, so such a work unit is rebuilt parameterized. Pre-existing
  and probably unreachable (the trace path serves parameterized graphs);
  unverified (diff review).
- An existing inconsistency (A): `laplace_transform` of a parameterized
  graph stays `parameterized=True`, while `reward_transform` of one becomes
  False.

## 7. Tests (new file `tests/pytest/test_serialize_constant_roundtrip.py`)

All tests use a temporary cache directory, so they never depend on the
user's cache. Values are compared with `rtol=1e-14`.

- **T1** A constant callback graph saved and then loaded through the real
  `Graph.__init__` cache-hit path: `parameterized()` is False,
  `update_weights` raises, and moments and pdf equal the freshly built
  graph.
- **T2** A constant graph through default `serialize()`: `param_length == 0`,
  `param_edges` empty, every edge in `edges`. The reload is constant and its
  values are equal.
- **T3** Genuine 1-parameter and 2-parameter graphs: default `serialize()`
  output is identical to `serialize(theta_dim=param_length())`, and the
  reload stays parameterized with the same `param_length`.
- **T4** A `discretize()`'d constant graph and a native DPH constant graph:
  the reload is constant, `is_discrete` / `was_dph` are preserved, and
  `pdf()` is equal. The discretized case raises today.
- **T4b** UNLOCKED graphs (`reward_transform` of a parameterized graph,
  `laplace_transform` of a constant one): the reload is constant, its
  values equal the original, and `update_weights` raises.
- **T4c** A constant graph that also has coefficient-less
  `add_edge_constant` edges, and a graph with `set_param_length(1)`
  followed by constant edges: round trip correct.
- **T5** Explicit `serialize(theta_dim=1)` on a constant graph keeps the old
  layout, documenting that the explicit argument wins.
- **T6** Stale entry. Write it with the old serialize behaviour and without
  the format marker (or under version "1.0", if D1 chooses the bump). The
  load must miss, rebuild, and give a constant graph. Also check that a
  stale-format entry with `param_length` 2 is still loaded (marker option
  only).
- **T7** `moments_from_graph` on a constant graph raises `ValueError` whose
  message matches "must have parameterized edges". Assert on the message:
  on this install, master fails earlier with a JIT-source `RuntimeError`.
- **T8** `pmf_and_moments_from_graph` on a constant graph raises the same
  `ValueError`, for both `use_ffi` settings.
- **T9** `reward_visit_probability` with a JAX θ on a constant graph
  behaves as D2 decides.

Before the fix is applied, check that T1, T2, T4, T4b, T6, T7, T8 and T9
fail on current master for the stated reason, and that T3, T4c and T5 pass
both before and after.

## 8. Gates

1. **New tests:** fail before and pass after, as stated in §7.
2. **Targeted files:**
   - `test_graph_cache.py`, `test_from_serialized_find_vertex.py`,
     `test_gate_persistent_graph_reuse.py`, `test_disk_param_cache_save_load.py`,
     `test_pull_push_cache.py`, `test_param_compute_graph_cache_survives.py`.
   - The 42 files reviewer B ran (421 passed with the fix monkeypatched in;
     list in `review_B/`).
3. **Full suite:** no new failures relative to `b3-test-baseline.md`.
   Pre-existing failures are not a gate.
4. **Adversarial review of the diff** before commit.
5. **Branch** `fix/serialize-constant-graphs`. Kasper merges and pushes.

## 9. Done means

- A graph with `parameterized()==False` comes back non-parameterized from
  any default serialize/load round trip, including the graph cache. This
  covers continuous, discretized, native-DPH and transform-output graphs.
- Parameterized graphs serialize byte-identically to today.
- Stale cache entries cannot reintroduce the bug.
- Every §5 site behaves as recorded.

## 10. Review record

**Reviewer A — correctness: SOUND-WITH-CORRECTIONS.** Root cause and fix
confirmed; no graph found that needs `param_edges` while `parameterized()`
is False; de-risk evidence reproduced. Corrections folded in:
- UNLOCKED graphs added to the scope (§1, §3, T4b).
- The `reward_visit_probability` row settled (§5).
- Tolerance-based comparison instead of exact equality (§3, §7).
- Invalidation cost and orphaned files: led to the marker option (§4b).
- T6 and T7 mechanics, plus T4c (§7).

**Reviewer B — blast radius: SOUND-WITH-CORRECTIONS.** No silent wrong
answer after the fix, and no legitimate reliance on the old layout found.
Corrections folded in:
- Three missed call sites, `pmf_and_moments_from_graph`, `pmf_from_graph`
  and `pmf_from_graph_joint_index` (§5, T8).
- `reward_visit_probability` is correct today, so its new error is a real
  trade-off (D2).
- The "loud everywhere" claim narrowed (§6).
- Hash-orphan note (§5) and orphaned files under the version bump (§4b).
- The stale demo assertion and comment (§4a).
- Clarifications: dead code in `ffi_wrappers`, `bffg` unaffected.

## 11. Decisions for Kasper

- **D1 — how to invalidate stale graph-cache entries.**
  - **Format marker (recommended).** Rebuilds only old entries with
    `param_length == 1`; nothing is orphaned.
  - **Version bump.** Rebuilds every cached graph once and leaves the old
    files orphaned.
- **D2 — `reward_visit_probability` with a JAX θ on a constant graph.** It
  is correct today and becomes a loud "theta length mismatch" error.
  - **(a) Accept the error (recommended).** Passing θ to a constant graph is
    a usage error.
  - **(b) Keep it working.** Add a branch that ignores θ for constant
    graphs. That adds behaviour that branches on graph type.

## 12. Decisions made (Kasper, 2026-10-11)

- **D1 (stale cache entries):** delete the cache and bump
  `PHASIC_CALLBACK_VERSION` "1.0" → "1.1". Deleting the cache removes the
  files the bump would otherwise orphan. The format marker is dropped.
  T6 becomes: an entry written under "1.0" is not loaded.
- **D2 (`reward_visit_probability` with θ on a constant graph):** raises,
  through the shared guard in `constant-parameterized-consistency-plan.md`
  Part 2, with its standard message.

This plan is now Part 1 of `constant-parameterized-consistency-plan.md`.

## 13. Implementation record (2026-10-11)

- Full suite, run file by file under a 6 GB / 25 min cap per file: 2025
  passed, 2 failed, 84 skipped, 26 xfailed. Two files were cut off by the
  caps and not counted: `test_notebook_model_recovery.py` (timeout at
  1500 s; no failure before the cut) and `test_svgd_divergence_robustness.py`
  (6.2 GB at its extreme-θ test, on a parameterized graph, which this
  change does not touch).
- The 2 failures were caused by this change and were not in §5:
  `inference/test_rewards_support.py::test_rewards_none_backward_compat`
  and `::test_vmap_with_rewards`. Their graph's only edge left the
  starting vertex, so it was an initial-probability edge and the graph
  was never parameterized (`parameterized()` False); it was silently
  treated as a 1-parameter model, and the tests only checked shapes.
  Fixed by adding a transient vertex so the rate edge is a real
  parameterized edge.
- Diff review (read-only): APPROVE-WITH-CHANGES; changes applied (T1
  checks the cache hit directly, T6 writes a real stale-layout entry,
  explicit `graph_cache=True`, `match=` on T4b, serialize docstring,
  hierarchical-cache note in §6).
