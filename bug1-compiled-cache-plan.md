# Bug 1: SVGD's compiled-gradient cache returns another fit's gradient — fix plan

**Status: v3, 2026-11-23 (v2 + the re-review's gate and wording corrections, section 7). v1 was reviewed by three independent agents
(diagnosis; fix design; tests and gate). All three confirmed the fix and
required changes to the diagnosis, cost figures, test list and gate; v2
makes them (section 7). Branch `fix/svgd-compiled-cache`, worktree
`/Users/kmt/phasic-bug1`. No source changed yet. Backward compatibility
is not required (user decision).**

Evidence: `experiments/bug1_compiled_cache/` (`b1`, `b2`; reviewers'
scripts and logs in `review1/`, `review2/`, `review3/`), all on unmodified
code.

## 1. The defect

`SVGD._precompile_unified` (`src/phasic/svgd.py:6989-7112`) builds
`jax.jit(jax.grad(partial(self._log_prob_unified, ...)))`. The bound
method captures the SVGD instance — observed data, prior and prior list,
parameter transform, model (and with it the exposure wrapper and
zero-inflation term); sample moments and rewards come in through the
partial. The function is stored in the CLASS-level dict
`SVGD._compiled_cache` (`svgd.py:5493`) under
`(id(self.model), theta_dim, obs_shape, nr_moments, regularization,
rewards_tuple[, 'value_and_grad'])` (`svgd.py:7032-7036`). A later
instance with the same model object and shapes gets the first instance's
function.

Fits are bit-reproducible in one process on a freshly built model
(review 1 E0, review 3 r1/r2: ten pdf-path scenarios and the epoch path),
so "second fit after a first fit on the same model object" can be compared
exactly with "the same fit on a fresh model object":

| second fit differs from the first in | effect on master |
|---|---|
| data, same shape (with or without `return_history`, with regularization) | particles = the first fit's answer (9.38 vs 2.18; differences of 9.0) |
| prior | wrong (6.72 vs 1.81) |
| `positive_params` | wrong (1.1e-2) |
| mutated `svgd.observed_data` then `optimize()` again on the SAME instance | stale (9.38 vs 2.18) |
| FreeEpochModel: `fit(prior=P1)` then `fit(prior=P2)` | wrong (3.9e-2) |
| FreeEpochModel: non-nested `fixed` (`[(1,1.0)]` then `[(3,1.0)]`) | wrong (theta_mean[1] 0.456 vs 0.946; particles differ by 1.09) |
| FreeEpochModel: nested then full | wrong (5.7e-4) |
| FreeEpochModel: `prior=None`, same slots fixed at different values | wrong (6.0e-3; the default prior depends on the fixed values) |
| FreeEpochModel: full then nested (the likelihood-ratio order) | particles identical; recorded log posterior offset by the now-fixed slot's prior term (up to 2.85) |
| fixed VALUES only, explicit prior | identical |

Rule: `fixed` values are applied outside the compiled function
(`svgd_step`, `svgd.py:4558-4567`), but the prior list is inside it and
`FreeEpochModel.fit` re-masks the prior at fixed slots
(`epoch_model.py:139-148`). Particles are right only when the cached
function's prior covers every slot that is free in the later fit. So the
documented `FreeEpochModel` workflow IS affected on master.

Who is affected: direct `SVGD(model=same_object, ...)` refits and
`FreeEpochModel.fit` reuse (`epoch_model.py:150`). `Graph.svgd` builds a
new model per call (`__init__.py:7040`) and the exposure path re-wraps the
model per instance (`svgd.py:6419`), so they never get a hit — for them
the cache does nothing except retain one entry per call forever (10 calls
→ 10 entries; each entry keeps the SVGD instance, model, data and history
alive, which is also why `id(model)` is never recycled).

Disk half: `open(path, 'wb')` succeeds before `pickle.dump` fails
(`svgd.py:7103-7110`), so every compile miss leaves a 0-byte
`~/.phasic_cache/compiled_svgd_<hash>[_vg].pkl` (15 fits → 15 files under a
redirected HOME). Loading such a file raises and is ignored, but a
non-empty file at that path IS loaded and used (a planted
`{'grad': jnp.zeros_like}` changed the posterior mean from 8.88 to 8.38
under a redirected HOME). The hash is built from `id(model)`, a process
address.

Unused older path: `_precompile_model`, `_save_compiled`,
`_load_compiled` (`svgd.py:6875-6962`) and `_get_cache_path`
(`svgd.py:6826-6841`) have no caller; `self.compiled_model` /
`self.compiled_grad` (`svgd.py:6432-6434`) are written only by them.

Not affected: regularization schedules; `jit=False` TOGETHER WITH
`precompile=False` (measured: either one alone still leaves
`jit_enabled=True` and creates a cache entry, `svgd.py:5691-5707`);
parallel mode; tied parameters (baked into distinct models).

## 2. Fix

Remove cross-instance sharing of compiled gradients; add nothing in its
place.

1. `_precompile_unified(nr_moments, sample_moments, regularization,
   rewards, value_and_grad)` keeps its signature (an experiment script
   calls it positionally) and just compiles and returns the function: no
   class-level cache, no disk read, no disk write, no directory creation.
2. No per-instance memo: `optimize()` is called once per instance
   everywhere in `src/` and always restarts from `theta_init`; within one
   `optimize()` jax.jit's own cache already covers repeated calls. A memo
   would add nothing (and would preserve the mutate-then-refit staleness).
3. Delete: `SVGD._compiled_cache`; `_get_cache_key_unified`;
   `_precompile_model`, `_save_compiled`, `_load_compiled`,
   `_get_cache_path`; `self.compiled_model`, `self.compiled_grad` and the
   commented-out call at `svgd.py:6432-6438`; the imports `pickle`,
   `hashlib`, `pathlib` at `svgd.py:11-13` (the other `Path` use at
   `svgd.py:5711` has its own import).
4. Text: docstrings at `svgd.py:5215-5221` (precompile parameter), `6994`,
   `7011`, `7225`; the block at
   `tests/pytest/inference/_demo_svgd_jax.py:605-625` (then
   `python -m py_compile` it); `atlas/*.md` mentions of
   `_precompile_model` are historical notes and stay. The generated
   `docs/api/SVGD.qmd:794` is regenerated by `pixi run api`, not edited by
   hand (noted for the user).
5. Experiment scripts that call `SVGD._compiled_cache.clear()` stop
   working by design: `experiments/bug1_compiled_cache/*` (this branch)
   and `experiments/fd_step_guard/v5/review_grad/a5_compiled_cache.py`.
   They are evidence for the unfixed code; a header comment is added to
   those on this branch saying so.

### Cost (review 2, measured properly: refits on one model object, recompiled vs cache hit)

| path | vertices | extra time per same-model refit |
|---|---|---|
| pdf + moments | 43 / 1003 / 2437 | 0.23-0.32 / 0.37-0.49 / 2.5-2.9 s |
| joint-index | 168 / 2808 | 0.27 / 0.41-0.53 s |
| epoch, 2 epochs | 168 / 2808 | 0.44 / 0.93-1.11 s |
| epoch + 40 exposures | 168 / 2808 | 0.52 / 1.4-1.8 s |

What a hit skips is (a) trace + XLA compile — 0.2 s on the pdf+moments
path regardless of graph size; on the epoch/joint-index FFI paths the
serialized graph is a static attribute of the call, so the module is
~2.8 KB per vertex and compile grows with it — and (b) one single-particle
gradient evaluation (the warm-up call at `svgd.py:7096`). At 20k-60k
vertices this is NOT measured; extrapolating linearly from two points
gives +3-8 s compile on the epoch/joint-index paths. Either way it is one
gradient evaluation plus a compile per fit, against
n_particles x n_iterations gradient evaluations in the fit. v1's "1-3 s"
mixed first-touch costs into the figure.

## 3. Tests (new file `tests/pytest/test_svgd_no_shared_compiled_grad.py`)

Reference for every comparison: the same fit on a FRESHLY BUILT model
object (no cache manipulation; after the fix there is no cache). Each
test asserts finiteness, exact equality of particles
(`np.testing.assert_array_equal`), and — for power — that the first and
second fits differ materially. Separate prior objects per fit
(`SVGD.__init__` sets `_transform` on the prior objects it is given).
Small models, seeded (`random` and numpy), a few seconds each.

Fail on master, pass after the fix:
1. different data, same shape — default (`return_history` on, the
   value_and_grad entry);
2. the same with `return_history=False` (the grad entry);
3. the same with `regularization > 0` (sample moments differ with the data);
4. different prior;
5. mutate `svgd.observed_data` on one instance, `optimize()` again —
   WITHOUT regularization (with `regularization > 0` the result differs
   from a fresh instance even with no cache, because `sample_moments` is
   computed in `__init__` and not refreshed: a property of mutation, not
   of this bug);
6. FreeEpochModel: `fit(prior=P1)` then `fit(prior=P2)`;
7. FreeEpochModel: non-nested `fixed` sets;
8. FreeEpochModel: nested then full;
9. FreeEpochModel: `prior=None`, same slots fixed at different values;
10. no retention: a weakref to a fitted SVGD instance dies after `del` +
    `gc.collect()` (default, no-history and epoch paths);
11. under `monkeypatch.setenv('HOME', tmp_path)`: fits (pdf, epoch,
    `Graph.svgd`) create no `compiled_svgd_*` file, and a recorder on
    `pickle.load` sees no load of a `compiled_svgd_*` path (other caches
    may legitimately unpickle their own files); a planted non-empty
    `compiled_svgd_*.pkl` cannot be loaded because nothing computes its
    name any more — asserted by the recorder;
12. `SVGD` has no attribute `_compiled_cache`.

Pass on master and after (guards):
13. FreeEpochModel full then nested: particles AND `history_log_prob`
    equal to nested on a fresh model (the log-posterior part fails on
    master: offset up to 2.85 — so this one is in the first group for that
    assertion);
14. same instance, `optimize()` twice without changes: bit-equal;
15. two `Graph.svgd` calls on one graph with different data: each equals
    its fresh-graph reference.

(v1's "different positive_params" test is dropped: the GaussPrior `lower`
rule forces the prior to differ too, and `positive_params=False` drives
this model out of its domain.)

Before implementing, tests 1-9 and 13 are run on the unmodified build to
record that they fail there.

## 4. Known behaviour-changing existing test

`tests/pytest/inference/test_nan_observations_correctness.py::test_nan_correctness_univariate`
fits one model with NaN-masked data and then with full data of the same
shape, and asserts the posterior means are close. On master the second fit
reuses the first's gradient (its own verbose output says "Using cached
compiled gradient from memory"), so it compares the NaN fit with itself
(1.3566 vs 1.3309, differing only through unseeded initial particles).
After the fix it becomes a real comparison, and it is unseeded. It is run
in isolation on both builds before the gate, three times each; whatever it
does is reported, and it is not edited to pass.

## 5. Gate (hardened after two review rounds)

Order: edit source → `pixi run install-dev` in this worktree (the package
is an installed copy; without it the old code is tested) → new test file →
gate.

Runner `.git/claude-scratch/gate2.sh`: one pytest process per test file
(199 files + the new one), not-slow, memory-capped 8 GB, the project's own
600 s per-test timeout, `--junitxml` and exit code per file, a `DONE`
marker at the end. `gate_rerun.sh <tree> <tag> <cap> files...` reruns
named files into an existing result directory without wiping it (used for
killed files, with a larger cap, and for isolated reruns under separate
tags). `gate2.sh` wipes its output directory, so it is never re-invoked
with the tag of a finished run.

`.git/claude-scratch/gate_compare.py <baseline> <candidate> --new-file
tests/pytest/test_svgd_no_shared_compiled_grad.py` passes only if ALL hold:

- both directories have `DONE`;
- the file sets are equal apart from the declared new file;
- every file has a usable result on BOTH sides (pytest exit 0/1/5, xml
  present and parsable, not killed); a file unusable on both sides with
  the same exit code needs an explicit `--ack-unusable` (reported);
- every test that passed in the baseline passes in the candidate;
- no test disappeared;
- every new test passes.

It also prints every other outcome change, the tests failing on both
sides, and tests above 540 s. Follow-up, before merging:

- any regression is rerun three times in isolation on both builds
  (separate tags) and classified (flake vs caused by the fix);
- tests failing on both sides that construct `SVGD` or call `.fit(` are
  rerun in isolation on both builds the same way (a test that failed in
  the single baseline run for an unrelated reason and is ALSO broken by
  the fix would otherwise be invisible);
- `tests/pytest/inference/_demo_svgd_jax.py` is checked with
  `python -m py_compile`.

Baseline: master `ca09e2e1` (same source as this branch's base). Total
test time of both runs is reported.

## 6. Other caches checked (review 1) — not changed here

| cache | location | finding |
|---|---|---|
| on-disk graph cache | `graph_cache.py:126,183`, `callback_hash.py:96-170` | **separate defect**: key = callback AST + kwargs; it misses module globals, helper functions and (for `with_ipv` callbacks, `callback_hash.py:110-112`) closures. Measured: after changing such a dependency 10x the cached graph is returned (expectation 0.833333 instead of 0.083333), across processes. Tracked as its own bug; NOT fixed on this branch. |
| `_pmap_file_cache` | `hierarchical_trace_cache.py:471,542` | keyed on file path only; stale only if a work-unit file is rewritten at the same path in one process (SLURM path); not tested |
| `_work_unit_store`, `_metadata_cache`, `_lib_cache`, native builder/daisy caches, per-thread graphs, mcmc | — | no wrong-result path found (native JSON keys not audited field by field) |

## 7. Review of v1 — dispositions

Review 1 (diagnosis): M1 fixed-only not benign → section 1 table and rule,
tests 7-9, 13; M2 graph cache → section 6 and a separate bug; m1 leak, m2
0-byte files, m3 mutate-and-refit (test 5), m4 positive_params test
(dropped; separate prior objects), m5 citations, m6 cleanup list → done.
Review 2 (fix design): cost decomposition and extrapolation label; who is
affected; full deletion and text list; memo rationale → section 2.
Review 3 (tests and gate): non-nested fixed and prior-override tests
(6, 7); `test_nan_correctness_univariate` (section 4); hardened gate
(section 5); HOME-redirected disk test (11) and litter (section 1);
fresh-model reference, power, finiteness, `return_history=False` (section
3); test 3/4 of v1 reworked. Not adopted: "run the baseline's SVGD files
twice for a flake set" — replaced by triple isolated reruns of any
regression on both builds, which answers the same question only where it
arises.

Re-review of v2: reviewer 1 SIGN-OFF (notes adopted: the `precompile`
sentence; test 11 records paths). Reviewer 2: new tests must pass for the
gate to pass; an unusable baseline file fails the gate; reruns must not
wipe results; `install-dev` step; the `precompile` sentence → all done.
Reviewer 3: `DONE` on both sides and equal file sets; unusable-on-both
needs explicit acknowledgement; failing-on-both list and isolated reruns of
SVGD-touching ones; test 5 without regularization; other outcome changes
printed → all done (`gate_compare.py` rewritten, `gate_rerun.sh` added).

## 8. Risks

- Same-model refits get slower by a compile and one gradient evaluation
  (section 2).
- A test that passed only through a stale hit changes outcome (one known,
  section 4); the gate finds others.
- Files `compiled_svgd_*.pkl` already in `~/.phasic_cache` are no longer
  read or written; the user can delete them (not done by this change).
