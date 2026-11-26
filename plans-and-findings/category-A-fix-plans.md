# Fix plans — the three reachable Category-A bugs (adversarially reviewed)

Each plan was drafted against the fix site, then subjected to an independent adversarial reviewer
(told to REFUTE, running code). Verdicts and corrections are folded in below.

---

## A1 — tied-slave parameter exported as a 0.0 sentinel (F-010, SVGD)

**Adversarial verdict: SOUND-WITH-CORRECTION** (core fix confirmed on a `tied=[(0,[0,1])]` fit; bug
reproduced: `theta_mean[slave]=0.693` vs master `8.3e-5`; DOF/AIC/summary unaffected).

### Root cause (grounded)
`_daisy_chain_svgd_model` locks each tied *slave* flat index in `broadcast_fixed` at sentinel `0.0`;
`_apply_tying` copies master→slave only **inside the forward pass** (a scatter on a copy), never
persisted. So `SVGD.particles`/`theta_mean` (svgd.py:6711–6712) keep `0.0` at slave columns. Only
`summary()` resolves them (via `self.model._tying_info['slave_to_master']`, svgd.py:9595); every other
surface reads the sentinel (→ `softplus(0)=0.693`, ~8000× wrong).

### Fix (corrected)
At the single result-finalization point in `SVGD.optimize()` (svgd.py ~:6711, confirmed the ONLY place
`self.particles` is assigned; `_tying_info` reachable there, no-op when absent), re-tie **both** the
particles **and the history**:
```python
ti  = getattr(self.model, "_tying_info", None) or {}
s2m = ti.get("slave_to_master", {})
if s2m:
    P = np.array(results["particles"])              # (n_particles, theta_dim)
    for slave, master in s2m.items():
        P[:, slave] = P[:, master]
    results["particles"] = P
    results["theta_mean"] = P.mean(axis=0)          # (redundant on the reported path -- get_results
    results["theta_std"]  = P.std(axis=0)           #  recomputes from particles under the default
                                                     #  transform -- but needed for the raw-attribute path)
    if results.get("history") is not None:          # [adv] SECOND surface the draft missed:
        H = np.array(results["history"])            # (n_steps, n_particles, theta_dim)
        for slave, master in s2m.items():
            H[:, :, slave] = H[:, :, master]
        results["history"] = H
```
(Copy is in UNCONSTRAINED space; softplus of the copied master value reproduces the master's rate —
verified. slave/master flat indices == particle columns == θ positions.)

### Files / tests
- `src/phasic/svgd.py` — result-export tail of `optimize()` only (NOT gradient/kernel/fixed handling).
- Test: after a tied daisy-chain fit assert `get_results()['theta_mean'][slave]==theta_mean[master]`,
  `get_results()['history'][:, :, slave]==[...,master]`, a θ round-trip yields the master's rate;
  `degrees_of_freedom`/AIC/BIC unchanged; `summary()` still prints `Tied→θ[k]`.

### Risks / notes
- **Touches SVGD code** (no-modify-SVGD rule) — export tail only; needs your approval to proceed.
- Out of scope (pre-existing, not addressable here): `map_estimate_with_optimization()` re-freezes fixed
  dims during ascent, so master moves and the copied slave doesn't — they diverge post-refinement. The
  model re-applies `_apply_tying` internally, so the *likelihood* is unaffected; only that one refined
  export would drift. Note it; don't fix it at the finalization point.

---

## A2 — FFI computes the combined PMF on the untransformed graph (F-007 / Q7.1)

**Adversarial verdict: SOUND-WITH-CORRECTION** (right handler confirmed via jaxpr =
`ptd_compute_pmf_and_moments`, no pure_callback; bug reproduced: FFI PMF-with-rewards bit-identical to
`g.pdf` (untransformed) not `reward_transform(g).pdf`). **But the draft's rationale is false.**

### Root cause (grounded)
`ComputePmfAndMomentsFfiImpl` computes the PMF as `g.dph_pmf`/`g.pdf` on the **untransformed** `g`
(graph_builder_ffi.cpp ~:562 batched / ~:611 non-batched); moments via
`compute_moments_impl(g, nr_moments, rewards_vec)` (~:581/:627). The pybind path reward-transforms `g`
first. Reachable on `use_ffi=True` + rewards; pinned strict-xfail in `test_gate_ffi_vs_pybind.py`.

### Fix
Mirror the pybind path: when `n_rewards > 0`, inside the existing `BatchError` try, build
```cpp
Graph g_transformed = is_disc
    ? g.reward_transform_discrete(GraphBuilder::rewards_to_int_or_throw(rewards_vec))
    : g.reward_transform(rewards_vec);
```
compute the PMF on `g_transformed`, and the moments via `compute_moments_impl(g_transformed, {})` +
the discrete correction. Both branches.

### Correction to the rationale [adv] — the fix repairs a SECOND latent bug
The draft claimed "the moment value is unchanged (math-equivalent)". **False for discrete:** the current
FFI feeds *real-valued* rewards to `compute_moments_impl(g, rewards)` (continuous reward-weighting) and
*then* applies the discrete correction — mathematically inconsistent. Measured: current FFI discrete
moments `[20, 530, 17870]` vs pybind `[20, 520, 17000]` (~2–5% off). Switching to
`reward_transform_discrete → empty-reward moments → correction` makes them match pybind — i.e. the fix
**also corrects the discrete moments**, not just the PMF. (Continuous moments were ~1 ulp apart and
become bit-identical.) Also: discrete + non-integer rewards currently returns a number; after the fix it
**raises** (via `rewards_to_int_or_throw`, matching pybind / no-silent-fallback) — the throw is inside
the batched `try`, so `BatchError` records + re-raises it safely.

### Files / tests
- `src/cpp/parameterized/graph_builder_ffi.cpp` (ComputePmfAndMoments, both branches).
- Flip the Q7.1 strict-xfail in `test_gate_ffi_vs_pybind.py` to a pass; update its stale line refs
  (actual sites ~562/611, not 492/741).
- **Add a discrete + integer-rewards FFI-vs-pybind MOMENT gate** — no such test exists today, so the
  ~2–5% discrete-moment correction would otherwise be unpinned. (`test_g1_..._moments_close` is
  continuous-only.)

### Risks / notes
- The "clone-first persistent-graph" risk in the draft is **moot** — `ComputePmfAndMomentsFfiImpl`
  builds a FRESH `g` per batch element (no per-thread cache), so `reward_transform` can't corrupt reuse.
- Keep the just-merged `is_disc` dispatch + moment correction intact.

---

## A3 — trace replay silently returns linear for a log-mode trace (F-004)

**Adversarial verdict: BROKEN as drafted.** The draft (bake `is_log` into the trace, default replay to
`trace.is_log`) is unsound because the trace is a **structure-only, mode-blind, cached** artifact.

### Why the draft is broken (grounded)
- **The cache key ignores `weight_mode`.** `compute_graph_hash(linear) == compute_graph_hash(log)`
  (verified identical; the trace records *unit weights* — log/linear is a replay-time interpretation,
  `phasiccpp.cpp:1130`). So the on-disk trace cache (keyed by that hash) cannot distinguish the two.
  Baking `is_log` into a cached trace ⇒ **cross-mode collision**: a log graph loads a cached *linear*
  trace (`is_log=False` → footgun persists); worse, a linear graph loads a cached *log* trace
  (`is_log=True`) → silently computes **log** — a NEW wrong-answer path in the DEFAULT case, triggered
  by an unrelated graph. Reproduced end-to-end.
- **Old pickles reload `is_log=False`** (dataclass class-default, no AttributeError) — cached log traces
  keep the footgun with no cache-version bump.
- **Two more construction sites** the draft never touches: the hierarchical **merge**
  (`hierarchical_trace_cache.py:1832`, the production provider via `_ensure_trace`) and the **C-JSON
  loader** (`trace_serialization.py:186`, tried first on every cache load; needs a new `is_log` field +
  getter in the C trace struct). Both would stamp `is_log=False`.
- Minor: `test_trace_requires_use_log` is a plain passing assertion (not `@xfail`), so post-fix it
  **fails** (still requires updating); use `getattr(graph,'_weight_mode','linear')` not a bare attr.

### Corrected fix — two options

**Option A (recommended: small, safe, no cache changes).** Keep the source of truth for `use_log` at the
**live graph's `weight_mode`**, threaded at the call site — which the production paths already do
(`_gate_backend`/`_trace` thread `getattr(g,'_weight_mode')=='log'`). Concretely:
1. Audit every *in-phasic* replay call site (`_ensure_trace` / moments-via-trace / any
   `instantiate_from_trace`/`evaluate_trace*` call that has the live graph) and thread
   `use_log = (graph._weight_mode == 'log')`.
2. Turn the low-level footgun **loud instead of silent**: change `use_log: bool = False` →
   `use_log: bool | None = None` on `instantiate_from_trace`/`evaluate_trace`/`evaluate_trace_jax`; when
   `None`, **raise** a clear error ("`use_log` must be given: True for weight_mode='log', False
   otherwise — the trace does not record the mode"). This converts a silent-wrong result into an
   immediate error for direct low-level callers, without any cache/serialization change.
   - Back-compat: no in-tree caller passes `use_log` positionally; explicit callers are unchanged.
     Callers that *omitted* it were either linear (correct) or log (already wrong) — the raise surfaces
     the ambiguity rather than guessing.

**Option B (complete but large: make the trace self-describing).** Only if you want replay to work with
no `use_log` at all: (1) include `weight_mode` in `compute_graph_hash` so linear/log get **distinct
cache keys** (kills the collision); (2) **bump the trace-cache version** to invalidate stale pickles +
C-JSON; (3) set `is_log` at **all three** sites — `record_elimination_trace`, the merge
(`hierarchical_trace_cache.py:1832`), and `_c_trace_to_python` (which also needs an `is_log` field +
getter added to the C trace struct + pybind binding); (4) `getattr` for `_weight_mode`. Substantially
larger than the draft implied.

**Recommendation:** Option A. It removes the *silent* failure (the actual F-004 harm) with a localized,
cache-safe change; Option B is a separate, larger initiative if fully self-describing traces are wanted.
