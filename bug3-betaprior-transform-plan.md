# Bug 3: a BetaPrior's (0, 1) transform is not applied consistently — fix plan

**Status: v1, 2026-11-23. Branch `fix/single-betaprior-transform`,
worktree `/Users/kmt/phasic-bug3` (own pixi env, built from master
`ca09e2e1`). No source changed yet. Backward compatibility is not required
(user decision). Depends on bugs 1 and 2 being merged first (same region of
`SVGD.__init__` / `MCMC.__init__`); this branch is rebased onto master
after they merge and implemented then. Experiments use a fresh model
object per fit (bug 1).**

Evidence: `experiments/bug3_single_betaprior/` (`b1_scope.py`, `b1.log`),
unmodified code.

## 1. The defect (read from source + measured)

`BetaPrior` declares `_natural_transform = sigmoid` (clipped to
[1e-9, 1-1e-9]; `src/phasic/svgd.py:1587`, `416`) so that its parameter
lives in (0, 1). Who applies it:

- `SVGD.__init__` builds the MODEL-facing transform per dimension only when
  `prior` is a list containing a prior with a natural transform
  (`svgd.py:5937-5969`). For a single prior object it falls through to
  softplus (`svgd.py:5982`).
- The PRIOR's own transform is wired separately (`svgd.py:6000-6023`): a
  single prior with a natural transform gets that natural transform.
- `MCMC.__init__` always uses softplus and hands softplus to every prior
  (`src/phasic/mcmc.py:271-287`); natural transforms are never consulted.
  Initial points come from `prior.sample`, which for a BetaPrior returns
  the inverse SIGMOID of a Beta draw (`svgd.py:1694`).

Measured (b1.log; "model saw" is the range of theta the model received):

| call | model transform | prior transform | model saw |
|---|---|---|---|
| SVGD, `prior=BetaPrior(...)`, dim 1 or 2 | softplus | sigmoid | theta up to **12.5** |
| SVGD, `prior=[BetaPrior]`, `[Beta, Beta]`, `[Gauss, Beta]` | per-dim, correct | same | Beta slots in (0, 1] |
| SVGD, BetaPrior (single or list), `positive_params=False` | none | none | theta in **[-2, 12.5]** |
| MCMC, single BetaPrior | softplus | softplus | samples up to **1.31**, mean 0.83 (prior mean 0.5) |
| MCMC, `[BetaPrior]` | softplus | softplus | samples up to **1.74**, mean 1.07 |
| MCMC, `[GaussPrior, BetaPrior]` | softplus | softplus | Beta slot up to **1.77** |

So a probability parameter given a BetaPrior is only confined to (0, 1) in
one of the ways of calling SVGD, and never in MCMC.

Also read from source (not yet measured): with a CUSTOM `param_transform`
and a list prior, a BetaPrior's own `_transform` becomes its sigmoid
(`svgd.py:6014-6017`) while the model gets the custom transform — the same
inconsistency.

## 2. Fix

**Principle: the transform a parameter goes through is decided in one
place, from its prior, and the model, the prior and the initial points all
use that one transform. A prior that needs a bounded domain is either
honoured or refused — never silently ignored.**

1. One function in `svgd.py`, used by both `SVGD.__init__` and
   `MCMC.__init__`:
   `_resolve_param_transform(prior, prior_list, positive_params,
   param_transform, theta_dim, where)` → `(transform_or_None,
   per_dim_fns_or_None)`, which also sets each prior's `_transform`.
   Rules:
   - `positive_params=True`:
     - list prior: slot i uses its prior's natural transform if it has one,
       else softplus (today's SVGD behaviour, now also MCMC's);
     - single prior WITH a natural transform: every slot uses that natural
       transform (new for SVGD and MCMC);
     - single prior without one, or no prior: softplus (unchanged).
     The prior's `_transform` is the same function the model side uses for
     that slot.
   - `positive_params=False` (with or without a custom `param_transform`):
     a prior with a natural transform raises `ValueError` — it cannot be
     honoured without the library's transform, in line with the existing
     rule that positive-support priors require `positive_params=True`
     (`_check_gauss_prior_lower`, `svgd.py`).
   - Everything else as today.
2. `MCMC` drops its own softplus-only wiring (`mcmc.py:262-287`) and calls
   the function; its result export (`mcmc.py:846-890`) and proposals
   already go through `self.param_transform`, so they need no change.
3. `BetaPrior`'s docstring ("pass it inside a per-dimension list") is
   updated: a single BetaPrior now applies to every parameter.

Not changed: the transform functions themselves, the clipping constants,
`BetaPrior`'s density, fixed-value handling (bug 2's wrapper sits on top of
whatever transform this function returns).

## 3. Tests (new file `tests/pytest/test_betaprior_transform_consistency.py`)

Spy model (records theta), fresh model object per fit.

Fail on master:
1. SVGD, single BetaPrior, dim 1 and dim 2: every theta the model receives
   is in [1e-9, 1-1e-9]; `svgd.param_transform(phi)` equals
   `prior._transform(phi)` slot by slot for a grid of phi.
2. SVGD, single BetaPrior is equivalent to the list form: dim 1 fit with
   `BetaPrior(...)` vs `[BetaPrior(...)]`, and dim 2 vs `[Beta, Beta]`
   (separate prior objects), same seed: particles array-equal.
3. MCMC with a flat likelihood and BetaPrior (single; `[BetaPrior]`;
   `[GaussPrior, BetaPrior]`): all samples of Beta slots in (0, 1); a
   Kolmogorov-Smirnov test against the Beta distribution (several seeds,
   median p > 0.1); the Gauss slot still matches its truncated Gaussian.
4. MCMC initial points for a Beta slot are inside (0, 1) after the
   transform.
5. BetaPrior with `positive_params=False` raises (SVGD and MCMC; single and
   list; with and without a custom `param_transform`).

Guards (pass on master and after):
6. SVGD list forms (`[BetaPrior]`, `[Gauss, Beta]`): particles
   bit-identical to master's behaviour is asserted indirectly — the model
   input stays in range and `param_transform` equals the per-dim functions
   (no stored reference numbers).
7. Priors without a natural transform (Gauss, LogGauss, Gamma, HalfCauchy;
   single and list) still get softplus on the model and on the prior, in
   SVGD and MCMC.
8. A real fit: a discrete geometric model whose parameter is a probability,
   single BetaPrior: posterior particles in (0, 1) and the truth inside
   the 95% interval (prior chosen to exclude the truth's neighbourhood
   only mildly — a recovery smoke test, not a pin).

## 4. Existing tests and documents that change

- Any existing test or notebook that uses a BetaPrior with
  `positive_params=False`, with a custom transform, or as a single prior
  in SVGD/MCMC changes behaviour (raises, or is now confined to (0, 1)).
  The reviewers are asked to list them; the gate confirms.
- MCMC fits with a BetaPrior in a list change (the Beta slot is now
  confined).

## 5. Gate

Same procedure and tools as bugs 1 and 2 (`gate2.sh`, `gate_compare.py
--new-file tests/pytest/test_betaprior_transform_consistency.py`,
`gate_rerun.sh`), after rebasing onto master-with-bugs-1-and-2 and
`pixi run install-dev`; baseline = the previous bug's gate run.

## 6. Risks

- MCMC and SVGD constructors are edited in the same region as bug 2's fix:
  rebase first, then implement.
- A user who relied on a single BetaPrior being softplus-transformed on
  the model side (an accident) gets different, now consistent, results.
