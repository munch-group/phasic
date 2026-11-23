# Bug 1: SVGD's compiled-gradient cache returns another fit's gradient — fix plan

**Status: v1, 2026-11-23. Branch `fix/svgd-compiled-cache`, worktree
`/Users/kmt/phasic-bug1` (own pixi env, built from master `ca09e2e1`).
No source changed yet. Backward compatibility is NOT required (user
decision 2026-11-23): prefer the simplest correct design.**

Evidence: `experiments/bug1_compiled_cache/` (`b1_repro.py`, `b1.log`,
`b2_compile_cost.py`, `b2.log`), run on unmodified code in this worktree.

## 1. The defect (read from source + measured)

`SVGD._precompile_unified` (`src/phasic/svgd.py:6989-7112`) builds
`jax.jit(jax.grad(partial(self._log_prob_unified, ...)))` — a function
closing over the SVGD instance (`self`): its observed data, prior list,
parameter transform, fixed values, sample moments, exposure wrapper — and
stores it in the CLASS-level dict `SVGD._compiled_cache` (`svgd.py:5493`)
under the key `(id(self.model), theta_dim, obs_shape, nr_moments,
regularization, rewards_tuple[, 'value_and_grad'])` (`svgd.py:7032-7036`).
A later SVGD instance with the same model object and shapes gets the FIRST
instance's function, whatever else differs.

Measured (b1.log), same model object, cache cleared between "alone" runs:

| second fit differs in | second fit after first | second fit alone |
|---|---|---|
| data (same shape) | 9.3805 (= the first fit's result) | 2.1768 |
| prior | 6.7231 | 1.8094 |
| `fixed` only (FreeEpochModel full → nested, same prior on shared free slots) | identical to alone | — |

The last row is benign because `fixed` is applied outside the compiled
function (`run_svgd`, `svgd.py:4558-4567`). That is the one same-model
reuse the library itself does (`epoch_model.py:100`, the likelihood-ratio
fast path), so shipped library paths are not known to be wrong; user code
and tests that refit one model object with different data, priors,
transforms or `positive_params` are.

Related facts:

- **Leak:** each entry keeps the whole first SVGD instance alive
  (weakref still alive after `del` + `gc.collect()`; dead after clearing
  the cache).
- **Disk half is dead code:** `pickle.dump` of the compiled function
  always fails (`PicklingError ... _log_prob_unified`), silently
  (`svgd.py:7103-7110`). A file `~/.phasic_cache/compiled_svgd_<hash>.pkl`
  would be LOADED if it existed (`svgd.py:7046-7062`), with a hash built
  from `id(model)` — a process address — so any such file left by an older
  version would be a wrong function. (I have not looked in the home
  directory.)
- **Unused older path:** `_precompile_model`, `_save_compiled`,
  `_load_compiled`, `_get_cache_path` (`svgd.py:6875-6962`) have no caller
  (the call at `svgd.py:6438` is commented out); they use the same kind of
  key.
- **What a hit saves** (b2.log, 1-iteration fits): 2.9 s (pdf path,
  two-island 186 vertices), 2.9 s and 1.0 s (epoch path, 168 and 2,808
  vertices). It is JIT trace/compile time only.

## 2. Fix

Remove cross-instance sharing of compiled gradients entirely.

1. `_precompile_unified` compiles and returns the function; no class-level
   cache, no disk read, no disk write, no per-instance memo either (an
   instance's data/prior attributes are mutable, so a memo could go stale
   the same way; `optimize()` calls `_precompile_unified` once per call).
2. Delete `SVGD._compiled_cache`, `_get_cache_key_unified`, and the unused
   `_precompile_model` / `_save_compiled` / `_load_compiled` /
   `_get_cache_path`, plus imports that become unused.
3. Update the one textual mention in `tests/pytest/inference/_demo_svgd_jax.py:617`
   and any docstring/doc text describing the cache.

Cost: a refit on the same model object pays the 1-3 s compile again
(measured above). No behaviour change for a first fit.

Not done: keying the cache "correctly". The compiled function depends on
everything `_log_prob_unified` reads from `self`; enumerating and hashing
that (including data contents) is fragile, and making those explicit
arguments of the jitted function is a refactor of the log-probability
path — not justified by 1-3 s.

## 3. Tests (new file `tests/pytest/test_svgd_no_shared_compiled_grad.py`)

Each compares "second fit after a first fit on the same model object" with
"second fit in a fresh process state" (same seed), requiring exact
equality of the particles:

1. different data, same shape (the b1 case; fails on master);
2. different prior (fails on master);
3. different `positive_params` / transform on the same model;
4. different regularization targets (sample moments) with the same
   `regularization` value;
5. FreeEpochModel full → nested gives the same nested result as nested
   alone (passes on master; guards the LRT path);
6. no retention: a weakref to a fitted SVGD instance dies after `del` +
   `gc.collect()`;
7. `SVGD` has no attribute `_compiled_cache` (so nothing can silently
   start relying on it again).

## 4. Gate

- New tests pass; tests 1-2 demonstrated failing on master first.
- Full `tests/pytest` (not-slow), one file per process, memory-capped
  (`.git/claude-scratch/gate.sh`): the set of failing/erroring test ids in
  this worktree must be a subset of master's baseline ledger
  (`/Users/kmt/phasic/.gate_baseline/ledger.txt`). Any test that newly
  fails is investigated: a test that passed only because of the shared
  cache is a finding to report, not something to paper over.
- Wall time of the gate vs baseline reported (the fix removes cache hits).

## 5. Risks

- A test or user workflow that refits one model object many times gets
  slower by the compile time per fit.
- Code outside `svgd.py` reading `SVGD._compiled_cache` (grep: none in
  `src/`; one print string in a demo script).
- Stale `compiled_svgd_*.pkl` files in `~/.phasic_cache` are no longer
  read; they are not deleted by this change.
