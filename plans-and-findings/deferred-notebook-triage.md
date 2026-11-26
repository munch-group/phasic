# Notebook triage — pre-Stage-3 blocker (runtime bugs in tutorial notebooks)

**Status:** triage complete. **1 of 2 genuine notebook bugs fixed** this pass
(`state-space`); the rest are enumerated below with root causes and fixes.
This document is self-contained for Stage-3 pickup.

**Why this exists.** `refactoring-review-prompt.md:35` makes fixing tutorial-
notebook runtime bugs a **hard blocker before any Stage-3 restructuring** —
"restructuring on top of already-broken behavior defeats the equivalence gates,
which would happily pin the *broken* result." The docs pipeline
(`scripts/docs-run-notebooks.sh`) only replaces a notebook's outputs **on
success**, so a failing notebook silently keeps its last-good version — binary
pass/fail with no error visibility. This triage re-executed every render-set
notebook with `--allow-errors` so the actual tracebacks are captured.

---

## 1. Method (reproducible)

Render set = the 24 notebooks uncommented in `docs/_quarto-default.yml` (what the
docs pipeline actually builds). Each executed from `docs/` with:

```bash
jupyter nbconvert --to notebook --execute --allow-errors \
  --ExecutePreprocessor.timeout=420 --ExecutePreprocessor.interrupt_on_timeout=True \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='{"skip-execution"}' \
  --output-dir <scratch> --output <name> pages/tutorial/<nb>.ipynb
```

- `--allow-errors` embeds error outputs instead of aborting (the fix for the
  pipeline's binary reporting). `--output-dir <scratch>` so **source notebooks
  are never modified**.
- Failures were classified by **root cause**, then every ambiguous case was
  **re-run serially with an isolated cache** (`PHASIC_COMPILATION_CACHE_DIR` →
  local dir) to strip parallelism artifacts (see §5).

Two harness caveats that produced **false positives** in the first parallel
(`-P3`) pass, both eliminated by the serial re-run — do not mistake them for
notebook bugs:
- **420s/cell cap** (my addition; the real docs pipeline sets *no* per-cell
  timeout) → long SVGD fits show as `XlaRuntimeError`-wrapping-`KeyboardInterrupt`.
- **`-P3` sharing NFS `~/.phasic_cache`** → `OSError: Device or resource busy:
  '.nfs…'` and OOM `DeadKernelError`s under memory contention.

---

## 2. Results (24 notebooks)

| Bucket | Count | Notebooks |
|---|---|---|
| ✅ Clean | 16 | getting_started, introduction, state-space-utils, visualize, properties, discrete, parametrization, laplace, joint-probability, svgd-priors-and-schedules, svgd-multi-feature, configuration, mpfr, scc-decomposition, model-hub, trace-and-jax-caching-c-path |
| 🐛 Genuine bug — **FIXED** | 1 | **state-space** |
| 🐛 Genuine bug — **OPEN** | 1 (2 failures) | **model-selection** |
| 🔧 Environment / robustness — OPEN | 2 | **time-inhomogeneous**, **distributed** |
| ⏱ Not a bug (420s cap; complete in real pipeline) | 4 | method-of-moments, svgd-basics, svgd-multi-param, svgd-joint-prob |

---

## 3. Genuine notebook bugs

### 3.1 `state-space.ipynb` — FIXED this pass ✅
- **Symptom:** `NameError: name 'cpp_state_spaces' is not defined` in the
  `%%timeit graph = Graph(cpp_state_spaces.coalescent(nr_samples))` benchmark
  (source cell 56).
- **Root cause:** the cell that imports the fragile `cpp_state_spaces` cppimport
  module (source cell 49) is tagged `skip-execution` (it needs `RTLD_GLOBAL`
  promotion for `ptd_err` and eager C++ compilation), but the dependent
  benchmark cell was **not** tagged — so the pipeline strips the import and then
  runs the benchmark.
- **Fix applied:** tagged the benchmark cell `skip-execution` too (matches its
  dependency; the C++ timing can't render in the automated build regardless).
  The paired Python `%%timeit` (cell 55) still runs. Verified clean
  (0 error outputs) after the fix.

### 3.2 `model-selection.ipynb` — OPEN (both failures are the deferred SVGD docs, live)
This notebook is where **both** deferred SVGD issues surface as real runtime
failures. Two independent failures:

**(a) LRT likelihood inversion — source cell 21**
```python
lrt = model_selection.likelihood_ratio_test(svgd, svgd_no_mig)
# ValueError: LL_nested (-1735.39) > LL_full (-1994.04). A true nested restriction
# cannot have higher likelihood than the full model; ...
```
- The nesting itself is set up correctly (the canonical pattern: `svgd_no_mig =
  SVGD(model=svgd.model, theta_dim=svgd.theta_dim, observed_data=svgd.observed_data,
  fixed=…)`, migration fixed to 0 as a strict superset — source cell 13). The
  inversion comes from **non-converged fits**: the tutorial uses tiny settings
  for speed (`n_iterations=5, n_particles=5`, cell 26), so the *full* fit doesn't
  reach the nested optimum, and strict `likelihood_ratio_test` rejects
  `LL_nested > LL_full`. Root cause = SVGD convergence quality, the domain of
  `deferred-svgd-divergece-fix.md` (non-convergence, though not the rate-blowup
  crash). **Fix options:** raise the tutorial's fit budget (uncomment the real
  `n_iterations`/`n_particles`), use `refine=True`, or `strict=False` with a
  narrated caveat.

**(b) `epoch_starts` TypeError — source cell 30 (→ cascade 31–35)**
```python
svgd_all_coal_free = SVGD(model=svgd_all_tied.model, observed_data=…,
                          epoch_starts=…, …)
# TypeError: SVGD.__init__() got an unexpected keyword argument 'epoch_starts'
```
- **This is exactly `deferred-svgd-lr-bug.md`.** The notebook reuses the tied
  epoch model to run a tied-vs-free LRT (cell 35:
  `likelihood_ratio_test(svgd_all_coal_free, svgd_all_tied)`), and hits the wall
  documented there: the direct `SVGD(model=…)` path rejects `epoch_starts`, and
  even without it the reused model has tying baked in. Cells 31–35 then
  `NameError` on `svgd_all_coal_free`.
- **Fix:** owned by `deferred-svgd-lr-bug.md` (Stage-3 WS-C). Until that lands,
  either guard cells 30–35 (`skip-execution` or simplify to a supported LRT) so
  the notebook executes cleanly, or drop the epoch tied-vs-free LRT example.

---

## 4. Environment / robustness (not phasic-notebook code bugs)

### 4.1 `time-inhomogeneous.ipynb` cell 3 — graphviz `dot` mis-registered
```
CalledProcessError: dot -Kdot -Tsvg → 'config8 is zero sized. There is no layout
engine support for "dot". Perhaps "dot -c" needs to be run to register the plugins?'
```
- A `.plot()` inline SVG render fails because the pixi env's graphviz plugin
  registry is broken. **Fix is environmental:** run `dot -c` (with install
  privileges) or reinstall graphviz in the pixi env. It only surfaced here
  because this notebook renders a *small* graph inline (larger graphs hit the
  "too many nodes" text path and skip rendering), so it likely lurks elsewhere.
  Re-triage the graph-rendering notebooks after fixing the env.

### 4.2 `distributed.ipynb` — kernel SIGABRT (thread exhaustion + hard crash)
```
libgomp: Thread creation failed: Resource temporarily unavailable
pybind11::handle::dec_ref() ... GIL is either not held or invalid
terminate called without an active exception     → kernel dies (~36s, even serially)
```
- The `pmap`/multi-device cells (source cells ~15, 20) exhaust the OpenMP
  thread/process limit, and the failure path then **hard-crashes** (pybind11
  `dec_ref` with an invalid GIL → `terminate`/SIGABRT) instead of raising a clean
  Python error. Part environment (thread `ulimit`), part **library robustness**.
- **Fix options:** (i) tag the offending cells `skip-execution` for the docs
  build (this notebook really wants a cluster), and (ii) file the
  pybind11-GIL-invalid-`dec_ref`-on-thread-spawn-failure as a separate
  robustness bug — a failed `libgomp` thread spawn should surface as an
  exception, not abort the interpreter.

---

## 5. Not blockers — false positives from the harness (confirmed)

- **`method-of-moments`, `svgd-basics`, `svgd-multi-param`, `svgd-joint-prob`** —
  each is `XlaRuntimeError` wrapping **`KeyboardInterrupt`** at
  `_compute_pmf_and_moments_cached`, i.e. my 420s/cell cap firing mid-`svgd()`;
  trailing `NameError: 'svgd'` cells are the cascade. **Verified none carry the
  rate-blowup divergence signature.** The real docs pipeline has no per-cell cap,
  so these complete. Note: `svgd-joint-prob` **no longer crashes** — the original
  sojourn allocation error is gone (`sojourn-fix.md`, committed `c340bedc`); it
  just runs long.
- **`trace-and-jax-caching-c-path`** — `OSError: Device or resource busy: '.nfs…'`
  in `clear_caches()` was the `-P3` NFS-cache race; **clean when run serially**
  with an isolated cache.

---

## 6. Recommended sequencing before Stage-3

1. **`state-space`** — done ✅ (this pass).
2. **graphviz env** — `dot -c` / reinstall in the pixi env (not a commit);
   re-triage `time-inhomogeneous` (and other inline-plot notebooks) after.
3. **`model-selection`** — its two failures *are* the deferred SVGD docs:
   - (b) epoch `epoch_starts` → resolve via `deferred-svgd-lr-bug.md` (WS-C), or
     guard cells 30–35 now to unblock the docs build.
   - (a) LRT inversion → raise the tutorial fit budget / `refine=True`, or
     `strict=False` with a caveat.
   Lower-churn path: **guard now, fix properly during WS-C** (which reshapes the
   `Graph.svgd`/`svgd.py` layer anyway).
4. **`distributed`** — `skip-execution` the cluster-only cells for the docs
   build; file the hard-crash robustness bug separately.
5. The 4 "timeout" notebooks need **nothing** — they pass in the real pipeline.

Net: the true pre-Stage-3 notebook blocker is **`state-space` (now fixed) +
`model-selection`**, and `model-selection` collapses into the two already-tracked
deferred SVGD docs plus a graphviz env fix. Small and well-scoped.

---

## 7. Reproduce / references

- Re-run one notebook with error capture: the `nbconvert` command in §1 (add
  `PHASIC_COMPILATION_CACHE_DIR=<local>` and run serially to avoid the NFS/OOM
  artifacts).
- Render list source: `docs/_quarto-default.yml`; pipeline:
  `scripts/docs-run-notebooks.sh` (replace-only-on-success — the reason errors
  are invisible).
- Related: `deferred-svgd-lr-bug.md` (model-selection cell 30/35),
  `deferred-svgd-divergece-fix.md` (model-selection cell 21 convergence),
  `sojourn-fix.md` (why `svgd-joint-prob` no longer crashes).
- Cleanup note: the parallel pass wrote a stray `island_model_derived_counts.csv`
  and an NFS `.nfs…` sillyrename under `docs/pages/tutorial/` — the CSV was
  removed; the `.nfs` file clears itself once its holder exits. Source notebooks
  were never modified by the triage (only `state-space` was edited, as the fix).
