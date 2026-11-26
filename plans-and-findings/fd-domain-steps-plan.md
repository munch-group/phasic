# Plan A: domain-aware finite-difference steps

**Status: v2, 2026-11-26. v1 (`b9178dab`) went through three independent
adversarial reviews (Opus; all CHANGES REQUIRED, no REJECT); their
findings are folded in here (dispositions in §11). v2 is NOT yet
re-reviewed. No code changed.** Split out of
[fd-step-and-slow-band-guard-plan.md](fd-step-and-slow-band-guard-plan.md)
v5 (its §3.2 and Batch 1), to be done BEFORE the core-engine refactor
([core-engine-ssot-plan.md](core-engine-ssot-plan.md)). The rest of v5
(true-uniformization forward, cost ceiling, `advance(dt)`) becomes Plan B
and waits until after the refactor, because it changes forward values and
the refactor is gated on bit identity.

Decisions (user, 2026-11-26): fitters declare domains from their
transforms, undeclared callers keep today's numbers (v5 D2); the step size
is fixed per forward family in Batch 0, starting from s = 1e-4 (v5 D3;
review 3 measured s = 1e-4 as sound: 2e-9..5e-9 relative error on the
epoch model, `experiments/fd_domain_steps/review3/p1_daisy.log`).

## 0. Copy-paste prompt for a fresh session

> Read `plans-and-findings/fd-domain-steps-plan.md` in full (including §11,
> the review dispositions, and the three review reports under
> `experiments/fd_domain_steps/review{1,2,3}/REPORT.md`) and `OPEN-ITEMS.md`.
> Implement it batch by batch on branch `fix/fd-domain-steps` in a worktree
> `../phasic-fdsteps`. Gate each batch before the next. Memory-cap every
> run (`.git/claude-scratch/runcap.py`); one heavy run at a time; never run
> reviewers alongside test runs. One Opus implementation review per batch.
> Do not edit notebooks (list cells for the user). `pixi run install-dev`
> after every source change. Never `git add -A`. The user pushes.

## 1. Problem (measured)

Every finite-difference (FD) gradient in the model builders probes
theta ± 1e-7 in absolute terms (one site uses max(|theta|·1e-6, 1e-10)):

- a rate at or below 1e-7 is probed at or below zero. On the multi-epoch
  (daisy-chain) path this raises `NEGATIVE transition rate`, the reported
  crash: reproduced on master for theta = [1, 1e-8, 1, 1e-8],
  [1, 1e-9, ...], [1e-9, 1e-4, ...], including with
  `exact_final_grad=True`, whose earlier-epoch slots still use FD
  (review3/p1_daisy.log);
- silent gradient errors where the dependence on a small rate is
  multiplicative (v5 §1: up to 4.6e15x on the default joint-graph path);
- with `positive_params=False`, two sites clamp the minus probe at 1e-9
  (`__init__.py:8662-8669`, `:9068-9077`), so a parameter that may be
  negative gets a secant over [theta, 1e-9]: rate = t0 + t1 at
  theta = [6, -5] gives -0.49382 instead of -0.50049 for t1
  (review2/probe_clamp.py);
- fixed slots are probed at sites that have no fixed-slot information.

## 2. Goal and definition of done

1. **No crash from a probe leaving a parameter's valid range, on every
   route** (Graph.svgd, `Graph.epoch_model(...).fit` / `FreeEpochModel`,
   direct `SVGD(model=...)`, model selection, direct `jax.grad` on a
   built model).
2. **Undeclared callers keep today's numbers bit for bit wherever today
   does not crash** (each site's own probes, divisor, clamp and guard).
3. Declared domains (from the fitters, `epoch_model`, or the caller) give
   probes inside the domain and the documented accuracy.
4. Fixed slots are never probed; tied slaves keep flowing to their master.
5. No forward value (pdf, cdf, moments, sojourn, stop probability)
   changes anywhere.
6. The one strict-xfail mixed-scale pin
   (`inference/test_fd_gradient_mixed_scale.py:229`) passes and its marker
   is removed; the "FD is broken on an undeclared builder" tests in the
   same file (`:112`, `:132`) stay unchanged and passing.
7. Suite comparison against Batch 0's master baseline: no new failures
   apart from the test edits in §7, each justified.

## 3. Domains and how they are declared

**Values per theta slot:**

- `None` (undeclared) = legacy: today's code, plus only the crash
  fallback of §4.4;
- `'positive'` (0, inf);
- `'unit'` (0, 1);
- `'unconstrained'`: plain ±1e-7, divided by the literal 2·eps, NO clamp
  (the fix for signed parameters);
- `'fixed'`: never probed.

**Side-effect-free resolver.** One function
`_resolve_param_domains(prior, positive_params, param_transform,
fixed, layout)` maps the fitter's settings to a domain list:

- softplus (positive_params) → `'positive'`;
- sigmoid (BetaPrior slot) → `'unit'`;
- identity (positive_params=False) → `'unconstrained'`;
- custom `param_transform` → `None` (unknown);
- truly fixed slots → `'fixed'`.

It never writes to the prior objects (`_resolve_param_transform`,
`svgd.py:5265`, writes `prior._transform`, `:5316-5322`).
`_resolve_param_transform` is refactored to call it, so the two cannot
drift.

**Where it runs:**

- **`Graph.svgd`**: before the model is built (the model is built at
  `__init__.py:7095` (epoch), `:7225`, `:7253`, `:7266`, `:7278`, before
  `SVGD(...)` at `:7288`). The domains go to the builder.
- **The epoch route**: domains per local slot are broadcast over the flat
  n_epochs × param_length vector. Tied slaves take their master's domain.
  With `prior=None`, the default epoch prior (GaussPrior with lower=0)
  means `'positive'`.
- **`Graph.epoch_model` / `FreeEpochModel`** build one model and fit it
  many times (`epoch_model.py:72-160`; `full.model is nested.model` is the
  likelihood-ratio-test fast path). They take `param_domains`, default
  `'positive'` for every flat slot (rates). `fit()` checks that its
  `positive_params`/prior agree with the model's domains and raises a
  clear error otherwise ("rebuild with `epoch_model(...,
  param_domains=...)`"). Fit-time fixed slots stay probed; that is
  harmless because of §4.5.
- **The model carries `_param_domains`.** `SVGD.__init__` checks it
  against its own transforms and raises on a conflict (for example a
  `'unit'` model fitted with softplus, where logit(theta > 1) would be a
  silent NaN).
- **Graph.mcmc: no change.** MCMC is gradient-free (`mcmc.py:536`, `:657`;
  no gradient call), so declaring domains there would be a no-op.
- **Direct callers** (`SVGD(model=...)`, model-selection examples,
  likelihood-ratio-test MAP refinement `svgd.py:7875`) get the crash
  fallback (§4.4) automatically and full accuracy by passing
  `param_domains`; documented.

## 4. Rules

### 4.1 Forward families (each named; Batch 0 confirms)

| Family | Jumps in theta? | Rule |
|---|---|---|
| Continuous pdf/cdf, Euler forward with automatic granularity (`pmf_from_graph`, `pmf_from_cpp`, the pdf half of `pmf_and_moments_from_graph`) | yes, when max exit rate > ~500 (`phasic.c:12963-13006`) | capped |
| Multi-epoch / daisy chain with granularity 0 (`_daisy_chain_svgd_model`, `daisy_chain_joint_probs`; `stop_probability(dt, 0)` at `graph_builder_ffi.cpp:1940`, `:2168`; `__init__.py:4729-4731`, `:5195`) | yes (measured: full rule up to 330x wrong at rate scale 200, review3/p4_daisy_jumps.log) | capped, all slots |
| Native discrete (DPH; `was_dph` false) | no jumps, but the row sum <= 1 + 1e-9 wall (`phasic.c:6698-6770`) | capped (the full step crashes within ~1e-4 of the wall, review3/p5_discrete_edge.log) |
| Elimination-only (moments, `moments_from_graph`, joint-index sojourn, `reward_visit_probability`) | no | full |
| Daisy chain with an explicit user granularity | no | full |

`pmf_and_moments_from_graph` runs one FD loop for pdf and moments together
(`:9068-9108`); with `exact_moment_grad=False` the moments inherit the
pdf's capped step (harmless; stated).

### 4.2 Probes and divisors

| Domain | Full rule | Capped rule | Divisor |
|---|---|---|---|
| positive | theta·e^(±s) | theta·e^(±u), u = min(log1p(1e-7/theta), s) | actual difference, with zero guard |
| unit | logit step of s | logit step of size min(s, 1e-7/(theta(1-theta))) | actual difference, with zero guard |
| unconstrained | — | theta ± 1e-7 | literal 2·eps, no clamp |
| `None` | — | today's code at that site, unchanged | today's (2·eps, or actual difference at the clamp sites) |

The capped up-probe is never more than 1e-7 above theta (review 1 m1
corrected v1's formula), and no probe is ever negative.

### 4.3 Error model

In the domain coordinate, error ≈ s²|F'''|/6 + eps_fwd|F|/s (for log
steps, s²·theta·(F''/2 + theta·F'''/6) in theta terms). Measured:
- at ordinary theta (>= 1e-3), the change from today is ~1e-8 relative,
  i.e. rounding level (review3/p1_daisy.log);
- for parameters whose effect is linear near 0, the relative error in
  theta terms grows like 1/theta: 1e-3 at 1e-9, ~100% at 1e-13
  (review3/p3_nearzero.log).

That is harmless for fits, where softplus floors theta at 1e-9
(`svgd.py:5254`) and the absolute error in phi is ~1e-11, but it is a
documented limit for direct theta-space callers.

### 4.4 Crash fallback for undeclared slots (fixes every route)

At sites whose forward REFUSES a non-positive rate (at least the
multi-epoch FFI; Batch 0 lists all of them), an undeclared slot with
0 < theta <= 1e-7 uses the capped positive rule instead of today's probe.
Everywhere else undeclared slots keep today's code. This changes results
only where today raises, so goal 2 holds. At sites whose forward accepts
a slightly negative probe (the Euler pdf path is ~1e-8 accurate there,
review3/p3_nearzero.log), undeclared slots are unchanged.

### 4.5 Degenerate values (no silent NaN, no silent zero)

- **Declared positive with theta <= 0, or unit with theta outside (0, 1):**
  that slot uses today's probe, selected with `jnp.where` (no Python
  branching on tracers). This happens for fit-time fixed values of 0
  (allowed since bug 2, `svgd.py:5196-5247`) and exposure-scaled values.
- **`'unit'` is refused on `exposure_param_index`** at construction.
- **A zero probe difference** (e.g. theta = 1e10 with a capped u of 1e-17)
  gives a zero gradient for that slot, only where today's site would also
  give zero. Batch 0 checks the cases.

### 4.6 Fixed and tied slots

The skip set is the truly fixed slots only, never tied slaves. Slave
partials must keep flowing to the master through `_apply_tying`'s
scatter-VJP (`__init__.py:5352-5357`); tied slaves are in SVGD's `fixed`
list (`:4964-4970`) but not in `fd_skip_indices` (`:4954`). Sites without
fixed-slot information get it as the `'fixed'` domain value:
- `pmf_from_graph`, signature `:3928`, loops `:4089`, `:4211`;
- `pmf_from_cpp`, `:4600`;
- `moments_from_graph`, `:8006`;
- `reward_visit_probability`, `ffi_wrappers.py:1296`. This is a
  module-level `custom_vjp` with `nondiff_argnums=(0,)`, so domains must
  be a hashable nondiff argument. It is reached via `zero_inflation.py:176-196`
  (attached by `Graph.svgd` at `:7292`) and `Graph.reward_visit_probability`
  (`:3751`).

### 4.7 One helper module

`src/phasic/_fd_steps.py`, a leaf module (`__init__.py` would make a
circular import). It returns probe points and divisors only. Each site
keeps its own arithmetic and expression order, so undeclared paths stay
bit identical. Everything is in jnp with no Python branching on theta, so
it runs under `vmap(jit(grad))`. Domains and rules are bound when the
model is built.

## 5. FD sites (master `b9178dab`; Batch 0 confirms and lists every loop)

| Site (constant → loops) | Entry point | Family |
|---|---|---|
| `__init__.py:4089`, `:4211` | `pmf_from_graph` | Euler / discrete |
| `:4600` | `pmf_from_cpp` | Euler / discrete |
| `:5245` → loops `:5306`, `:5344` (exact_final override) | `_daisy_chain_svgd_model`, no exposure | multi-epoch |
| `:5501` → loops `:5647`, `:5698` | `_daisy_chain_svgd_model`, exposure | multi-epoch |
| `:8006` | `moments_from_graph` | elimination |
| `:8661`, `:8943` (cdf at zero, clamp), `:8998` → `:9068-9108` (main, clamp) | `pmf_and_moments_from_graph` | Euler + elimination |
| `:9545-9570` (step max(abs(theta)·1e-6, 1e-10), numpy, probes all slots; goes negative below 1e-10) | joint-index host fallback for `exact_grad_decline='fd'` (forwarded by `Graph.svgd` at `:7229-7237`) | elimination; gets the positive rule and the fixed skip |
| `:9888` | `pmf_from_graph_joint_index` | elimination |
| `:12018` | `daisy_chain_joint_probs` | multi-epoch |
| `ffi_wrappers.py:1296` | `reward_visit_probability` backward | elimination |

`pmf_and_moments_from_graph_multivariate` inherits through its per-feature
calls.

**Out of scope, stated:**
- the SVGD preconditioner's `compute_scaling` (`svgd.py:4017`, `:4121`,
  `:4220`): it differentiates in the unconstrained phi coordinate, so it is
  already inside any domain;
- MCMC (gradient-free);
- the scipy `least_squares` numeric Jacobians in
  `probability_matching.py:447` and `method_of_moments.py:480` (bounded at
  1e-10);
- the BetaPrior moment solver (`svgd.py:1778`);
- the hand-written FD loop inside `test_fd_gradient_mixed_scale.py:141`
  (a test oracle).

## 6. Batches

### Batch 0: census, step sizes, baseline (no source change)

- Every FD loop with its family, slots probed, divisor/clamp/guard, and
  whether its forward refuses non-positive rates (the §4.4 list).
- s per family on the five model types (coalescent, two-island,
  reward-transformed, epoch, discrete).
- The degenerate cases of §4.5.
- The census of tests that replicate today's rule.
- A fresh master suite baseline with `gate2.sh`.
- Record in `plans-and-findings/fd-domain-steps-findings.md`.

### Batch 1: helper, builders, crash fallback (bit identical where today works)

- `_fd_steps.py`; `param_domains` on every builder in §5 plus
  `Graph.epoch_model` (default `'positive'`); the `'fixed'` value; the
  §4.4 fallback; the §4.5 guards.
- **Tests** (new `tests/pytest/test_fd_domain_steps.py`):
  - per-site bit identity for undeclared slots where today works, under
    eager, `jit`, `vmap(grad)` and the `custom_vmap` epoch forward;
  - probes stay in the domain (positive near 1e-12; unit near 0 and
    1 - 1e-9);
  - accuracy against exact gradients or closed forms per family, at theta
    in {1e-6, 1, 1e3}, on models whose slots have distinct gradients,
    with fixed and tied slots interleaved, using absolute tolerances in
    the domain coordinate;
  - the tied-master gradient equals the sum of per-epoch partials (extend
    `test_daisy_chain_c_path.py::test_tied_gradient_sums_per_epoch_partials`);
  - theta = 0 and fixed-0 cases;
  - discrete near the row-sum wall;
  - signed parameters under `'unconstrained'`;
  - formula and log weight modes with fixed slots;
  - the crash cases of §1 run on every route (Graph.svgd, `epoch_model`,
    direct `SVGD(model=...)`, direct `jax.grad`).
- The mixed-scale pin flips here, because `epoch_model` now declares
  `'positive'`.

### Batch 2: Graph.svgd declares domains (fits change slightly)

- The resolver in `Graph.svgd`; `_param_domains` on the model; the
  consistency check in `SVGD.__init__`; `positive_params=False` →
  `'unconstrained'`.
- **Tests:**
  - the flat epoch domain list for mixed Beta/Gauss priors with tying and
    fixed slots;
  - gradients at fixed particles equal master's within 1e-6 relative for
    theta >= 1e-3 (measured ~1e-8);
  - one coarse fit-level check per route;
  - the reported epoch fit from near-floor particles completes.

## 7. Existing tests expected to change

- `inference/test_fd_gradient_mixed_scale.py:229`: the strict xfail flips
  (Batch 1); remove the marker. `:112`, `:132`: unchanged.
- `test_exact_grad_daisy_final.py` (direct builder, `:68-70`),
  `test_exact_grad_rewards.py` (`:106-111`), `test_daisy_chain_c_path.py:340`,
  `:797` (replicate the 1e-7 rule): must stay BIT EXACT; these are not
  edits (v1 §6 was wrong).
- `test_likelihood_correctness.py`: the `log(pmf + 1e-10)` regulariser
  artefact (v5), only if Batch 0 confirms it is affected.
- Fit-level strict xfails in `test_notebook_model_recovery.py` (`:484`,
  `:553`, `:659`, `:699`; `:699` is the "particles stuck at the 1e-9
  floor" pin) may XPASS after Batch 2. The Batch 2 gate runs them
  explicitly. A flip is reported to the user, who decides whether to
  remove the marker; it is not edited silently.
- Batch 0's census may add more; each is listed before Batch 1 starts.

## 8. Gate per batch

`gate2.sh` on the worktree and `gate_compare.py` against Batch 0's master
baseline (`--new-file tests/pytest/test_fd_domain_steps.py`). No new
failures; regressions rerun three times in isolation. Batch 1
additionally passes the per-site bit-identity tests. One Opus
implementation review per batch.

## 9. Risks and limits

- **Fits shift** at rounding level for ordinary parameters (measured
  ~1e-8 in gradients); a chaotic fit trajectory can amplify that.
  Notebook numbers shift on re-run (listed for the user, not edited).
- **Dominated gradient components** stay inaccurate in relative terms
  under any FD step.
- **Linear-near-zero parameters** in theta-space direct calls: the §4.3
  limit.
- **The cap keeps today's jump behaviour** on Euler forwards; only Plan B
  removes it.
- **Overlap with the refactor:** `__init__.py` is also the refactor's
  territory; doing Plan A first avoids rebasing later.

## 10. Out of scope

Plan B (OI-002 forward replacement), exact PDF gradients (OI-005), the
preconditioner grid's units (OI-016), the rate-blowup forward/backward
mismatch (OI-027), and the items listed at the end of §5.

## 11. Review dispositions (v1 → v2)

Reports: `experiments/fd_domain_steps/review{1,2,3}/REPORT.md`.

- **Unreachable definition of done.** All three reviews (R1 B1, R2 B1,
  R3 B1): one pin, not three, and it is undeclared. Fixed by §2.6, §3
  (`epoch_model` declares `'positive'`) and the §4.4 crash fallback on
  every route.
- **Domains cannot reach the builders; FreeEpochModel reuse.** R1 B2,
  R2 B2 and M8, R3 M4 → the §3 resolver, the epoch broadcast and tying
  rules, the `fit()` check, and `_param_domains` checked by
  `SVGD.__init__`.
- **Missing sites.** R1 M1, R2 M3: §5 now includes the joint-index host
  fallback, lists loops by line, and adds the rvp threading.
- **Fixed and tied slots.** R1 M2, R2 M3c → §4.6.
- **Bit identity.** R1 M3, R2 M1, R3 m6 → §4.2 divisors, §4.7, and
  per-site tests.
- **Degenerate values.** R1 M4, R2 M4, R3 m6 → §4.5.
- **Forward classification.** R1 M5, R2 M5, R3 M2 → §4.1, including the
  capped logit.
- **Discrete row-sum wall.** R1 M6, R2 M6, R3 M3 → discrete is capped.
- **Section 6 contradictions.** R1 M7 → §7 rewritten.
- **The fit-level xfails.** R2 M7 → §7.
- **MCMC.** R1 M8, R2 m1 → dropped.
- **Preconditioner.** R1 M9, R2 m2, R3 m7 → out of scope.
- **Signed parameters.** R2 M2 → the `'unconstrained'` value, with no
  clamp.
- **Cap formula.** R1 m1 → §4.2.
- **Error-model coordinate and the linear-near-zero limit.** R1 m2,
  R3 m5 → §4.3.
- **Testing at fixed particles.** R1 m3, R3 Q3 → Batch 2.
- **Hashable nondiff argument.** R1 m4 → §4.6.
- **Out-of-scope solvers.** R1 m5, R2 M3 → §5.
- **Shared pdf and moments loop.** R1 m6 → §4.1.
- **Direct callers.** R2 m3 → §3.
- **Unit near 1.** R2 m4: the tolerance is noted in the Batch 1 tests.
- **jnp only.** R2 m5 → §4.7.
- **Gate strength.** R3 m6 → the Batch 1 and Batch 2 tests.
