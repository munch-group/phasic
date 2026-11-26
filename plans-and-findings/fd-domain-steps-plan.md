# Plan A: domain-aware finite-difference steps

**Status: v1, 2026-11-26. NOT reviewed. No code changed.** Split out of
[fd-step-and-slow-band-guard-plan.md](fd-step-and-slow-band-guard-plan.md)
v5 (its §3.2 and Batch 1), at the user's request, to be done BEFORE the
core-engine refactor ([core-engine-ssot-plan.md](core-engine-ssot-plan.md)).
The rest of v5 (true-uniformization forward, cost ceiling, `advance(dt)`)
becomes Plan B and waits until after the refactor, because it changes
forward values and the refactor is gated on bit identity.

Decisions taken (user, 2026-11-26, "go with the recommendations"):
v5's D2 (svgd/mcmc declare domains from their transforms; undeclared means
unconstrained means today's code) and D3 (step size fixed per forward
family in Batch 0, starting from s = 1e-4).

## 0. Copy-paste prompt for a fresh session

> Read `plans-and-findings/fd-domain-steps-plan.md` in full, and
> `OPEN-ITEMS.md`. Implement it batch by batch on branch
> `fix/fd-domain-steps` in a worktree `../phasic-fdsteps`. Gate each batch
> before the next. Memory-cap every run (`.git/claude-scratch/runcap.py`,
> or `scripts/` once moved there); one heavy run at a time; never run
> reviewers alongside test runs. Reviews: one Opus reviewer per round;
> after a clean sign-off, two more independent reviewers. Do not edit
> notebooks (list cells for the user). Run `pixi run install-dev` after
> every source change. Never `git add -A`. The user pushes.

## 1. Problem (measured; evidence in v5 §1 and `experiments/fd_step_guard/`)

Every finite-difference (FD) gradient in the model builders probes
theta ± 1e-7 in absolute terms:

- a rate below 1e-7 is probed NEGATIVE: the reported `NEGATIVE transition
  rate` crash on epoch fits, and silent gradient errors up to 4.6e15x on
  the default joint-graph path and 3.4e3x on the main pdf path;
- the mixed-scale pins (`inference/test_fd_gradient_mixed_scale.py`, strict
  xfail; OI-043) record the multi-epoch backward aborting for mu ~ 1e-8;
- fixed parameter slots are probed at four sites, which makes formula and
  log weight modes raise there.

## 2. Goal and definition of done

- A probe never leaves the declared domain of its parameter.
- Callers that declare nothing get exactly today's numbers (bit identical).
- `Graph.svgd` and `Graph.mcmc` declare domains from the transforms they
  already build, so fits on positive parameters no longer probe negative
  rates; the mixed-scale pins flip to passing.
- Fixed slots are never probed.
- No forward value (pdf, cdf, moments, sojourn, stop probability) changes
  anywhere.
- Suite comparison against a fresh master baseline shows no new failures
  apart from the test edits listed in §6, each justified.

## 3. Design

1. **Domains.** Each model builder takes `param_domains`: a sequence with
   one entry per theta slot, each `'positive'` (0, inf), `'unit'` (0, 1) or
   `None` (unconstrained). Default `None` for all = today's code path,
   kept byte for byte (including today's clamp sites).
2. **Rule, in the domain's coordinate, divided by the actual probe
   difference:**
   - positive: theta·e^(±s) (log step), never crosses 0;
   - unit: logit step of size s, never crosses 0 or 1;
   - unconstrained: theta ± 1e-7, as today.
3. **Transitional cap on the current continuous forward.** While a model
   runs on the Euler forward with automatic granularity (the default
   continuous pdf/cdf path, which has jumps where the step count changes),
   a positive parameter uses h = min(1e-7, s·theta) in the same log family:
   never larger than today's step, so it straddles no more jumps than today,
   and it never goes negative. Elimination-only forwards (moments, sojourn,
   joint index), discrete models and models with an explicit granularity
   have no jumps and use the full rule. The cap disappears when Plan B
   replaces the forward.
4. **Fixed slots** are skipped at every site.
5. **One helper module** (a leaf module, e.g. `src/phasic/_fd_steps.py`;
   it cannot live in `__init__.py` because of a circular import) computes
   probe points and divisors; every site calls it. This also removes the
   current copy-pasted step logic.
6. **Bound at construction.** The domains and the step rule are fixed when
   the model is built; a later `configure()` cannot split a forward from
   its backward.
7. **Declaration by the fitters.** `Graph.svgd` / `Graph.mcmc` map their
   per-slot transforms to domains (softplus → positive, sigmoid → unit,
   identity / custom transform → None) and pass them to the builder they
   call. A custom `param_transform` declares nothing (unknown domain).

Error model (v5 §3.2, measured): error ≈ s²|F'''|/6 + eps_fwd·|F|/s in the
domain coordinate. A component dwarfed by another parameter's effect on
the same output stays inaccurate in relative terms under any FD step; the
exact moments gradient (default on) covers moments.

## 4. FD sites (grep of master `f4eeea41`, 2026-11-26; Batch 0 confirms)

| Site | Entry point | Forward family |
|---|---|---|
| `__init__.py:4089`, `:4211` | `pmf_from_graph` (two backward rules) | continuous Euler / discrete |
| `__init__.py:4600` | `pmf_from_cpp` | continuous Euler / discrete |
| `__init__.py:5245`, `:5501` | `_daisy_chain_svgd_model` (epoch fits) | multi-epoch FFI |
| `__init__.py:8006` | `moments_from_graph` | elimination (generated C++) |
| `__init__.py:8661`, `:8943`, `:8998` | `pmf_and_moments_from_graph` (cdf-at-zero and main backward) | continuous Euler + elimination |
| `__init__.py:9888` | `pmf_from_graph_joint_index` | elimination (sojourn) |
| `__init__.py:12018` | `daisy_chain_joint_probs` | multi-epoch FFI |
| `ffi_wrappers.py:1296` | `reward_visit_probability` backward | elimination |
| `svgd.py:4017`, `:4121`, `:4220` | preconditioner `compute_scaling` (step 1e-5, theta space when no transform) | any |

v5 counted 15 builder sites; some constants above serve several loops.
`pmf_and_moments_from_graph_multivariate` inherits through its per-feature
calls. `mcmc.py`'s `epsilon = 1e-6` is a covariance regulariser, not FD.

## 5. Batches

### Batch 0: census and step size (no source change)

- List every FD loop (not only constants) with its forward family, which
  slots it probes, and whether it probes fixed slots. Confirm or correct §4.
- For each forward family, measure forward noise and pick s (start 1e-4)
  from the error model on: coalescent (elimination), two-island (cyclic),
  a reward-transformed model, the epoch model, a discrete model. Record
  in `plans-and-findings/fd-domain-steps-findings.md`.
- Census of tests that replicate the 1e-7 formula or compare exact vs FD
  at tight tolerance (v5 found three, §6).
- Fresh suite baseline on master (`gate2.sh`), since the bug 5 run is the
  last one.

### Batch 1: helper module and builders (bit-identical by default)

- `_fd_steps.py`; `param_domains` on every builder in §4; every site calls
  the helper; fixed slots skipped.
- Nothing declares domains yet, so the whole suite must be bit identical
  except the fixed-slot change (formula/log modes that raised now run).
- Tests (new `tests/pytest/test_fd_domain_steps.py`): undeclared =
  master bit for bit on each builder; probe points stay inside each
  domain (positive near 1e-12, unit near 0 and 1 - 1e-9); accuracy vs exact
  gradients or closed forms per forward family within the error-model
  tolerance; additive (rate = t0 + t1) and offset parameters under each
  declaration; formula and log weight modes with fixed slots.

### Batch 2: svgd and mcmc declare domains (fits change)

- `Graph.svgd` / `Graph.mcmc` pass domains derived from their transforms.
- Preconditioner sites use the same rule when no transform is set.
- Tests: the epoch fit from near-floor particles completes (the reported
  crash); the three mixed-scale pins flip (remove their xfail markers);
  fits on ordinary-sized parameters agree with master within a stated
  tolerance (multiplicative instead of additive probes, so not bit
  identical); fixed-value fits unchanged.

## 6. Existing tests expected to change (each justified, none loosened silently)

- `inference/test_fd_gradient_mixed_scale.py`: strict xfails flip to passes.
- Tests that replicate today's 1e-7 rule or compare exact vs FD at
  rtol 1e-8: `test_exact_grad_daisy_final.py::test_false_path_is_plain_fd`,
  `test_exact_grad_rewards.py` (two cases). New tolerances from the error
  model.
- `test_likelihood_correctness.py`: the `log(pmf + 1e-10)` regulariser
  artefact (v5).
- Batch 0's census may add more; each is listed before Batch 2 starts.

## 7. Gate per batch

`gate2.sh` on the worktree and `gate_compare.py` against Batch 0's master
baseline with `--new-file tests/pytest/test_fd_domain_steps.py`; no new
failures; regressions rerun three times in isolation; Batch 1 additionally
requires bit identity on undeclared paths. One Opus implementation review
per batch.

## 8. Risks and limits

- Fits change slightly (Batch 2) for everyone using svgd/mcmc with
  positive parameters; materially only for parameters below ~1e-3, where
  today's gradients are wrong. Notebook numbers shift on re-run (listed
  for the user, not edited).
- Dominated gradient components stay inaccurate in relative terms.
- The transitional cap keeps today's jump behaviour on the Euler forward;
  only Plan B removes it.
- `__init__.py` is also the refactor's territory; doing Plan A first
  avoids rebasing it later.

## 9. Out of scope

Plan B (true uniformization, cost ceiling, `advance(dt)`; OI-002), exact
PDF gradients (OI-005), the preconditioner grid's units (OI-016), the
rate-blowup forward/backward mismatch (OI-027).

## 10. Open items affected

Closes OI-043 and the FD-probe half of OI-002 (also Batch 1 of
[epoch-fit-prior-and-fd-floor-plan.md](epoch-fit-prior-and-fd-floor-plan.md)).
