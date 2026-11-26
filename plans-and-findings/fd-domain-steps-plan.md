# Plan A: domain-aware finite-difference steps

**Status: v2.3, 2026-11-26. v1 (`b9178dab`) went through three independent
adversarial reviews, v2 (`a4d2b3ad`) a fourth, v2.1 (`f27561fa`) a fifth and
sixth, v2.2 (`74ef880b`) a seventh (Opus; all CHANGES REQUIRED, no
REJECT); their findings are folded in here (dispositions in §11). v2.3 is
NOT re-reviewed. No code changed.** Split out of
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
> the review dispositions, and the review reports under
> `experiments/fd_domain_steps/review{1..7}/REPORT.md`) and `OPEN-ITEMS.md`.
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
2. **Undeclared callers keep today's numbers bit for bit**, with one
   stated exception: on models whose forward refuses a negative rate (§4.4: multi-epoch,
   discrete, or non-linear weight mode), an undeclared slot with
   0 <= theta <= 1e-7 uses the §4.4 fallback even when
   today's probe happened not to crash (today's crash depends on the edge
   weight sum c·theta, not on the slot alone: theta = 1e-7 does not crash,
   review4/p1_edges.log; an exact trigger is impossible under jit). Bit
   identity tests avoid that band.
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
  With `prior=None` and `positive_params=True`, the default epoch prior
  (GaussPrior with lower=0) means `'positive'`; with
  `positive_params=False` it means `'unconstrained'`
  (`__init__.py:7116-7126`).
- **`Graph.epoch_model` / `FreeEpochModel`** build one model and fit it
  many times (`epoch_model.py:72-160`; `full.model is nested.model` is the
  likelihood-ratio-test fast path). They take `param_domains`, default
  `'positive'` for every flat slot (rates). `fit()` checks that its
  `positive_params`/prior agree with the model's domains and raises a
  clear error otherwise ("rebuild with `epoch_model(...,
  param_domains=...)`"). The check is CONTAINMENT, not equality: a fit's
  domain must lie inside the model's (unit ⊂ positive; everything ⊂
  unconstrained), so a BetaPrior fit on a positive model works as today;
  only a widening raises (e.g. a softplus fit on a 'unit' model). `fit()`
  reads the domains stored on the `FreeEpochModel` object, not on
  `em.model` (tests replace `em.model` with spies,
  `test_fixed_values_exact.py:341-354`). Rejecting `positive_params=False`
  on a default (positive) `epoch_model` is a listed behaviour change (§7).
  The epoch_model default is introduced in Batch 2, together with the
  `Graph.svgd` resolver, so the two routes keep their parity
  (`test_epoch_model.py:83`, `:161`, rtol 1e-9). Fit-time fixed slots stay probed; that is
  harmless because of §4.5 (declared-positive slots at theta <= 0, such as
  a rate fixed at 0 for the nested likelihood-ratio model, get a one-sided
  step).
- **The model carries `_param_domains`.** `SVGD.__init__` checks it
  against its own transforms and raises on a conflict (for example a
  `'unit'` model fitted with softplus, where logit(theta > 1) would be a
  silent NaN).
- **Custom `param_transform`** resolves to `None` (unknown); `None` passes
  the containment check (on refusing models the primal already refuses
  invalid theta).
- **Length check:** the domain list's length is checked against the traced
  `theta.shape[-1]` at first trace (formula and callback models resolve
  theta_dim late); a mismatch raises.
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
| Discretized / joint graphs (`was_dph` true, from `discretize()` or `joint_prob_graph(discrete=True)`) | the discretize slot enters every row's normalisation; a negative probe on it raises (review6/p4_discrete.log) | capped; the step rule for the discretize slot is fixed by a Batch 0 jump/accuracy check |
| Elimination-only (moments, `moments_from_graph`, joint-index sojourn, `reward_visit_probability`) | no | full |
| Daisy chain with an explicit user granularity | no | full |

The upper row-sum wall of native discrete graphs (a row summing above
1 + 1e-9, e.g. a probability slot at 1 - 5e-8 or 1.0) already crashes today
on both pmf sites (review6/p4_discrete.log). It is not a per-parameter
range (a row can hold several slots), so no per-slot rule can protect it:
OUT OF SCOPE, and the Batch 1 test asserts today's clear error there.

`pmf_and_moments_from_graph` runs one FD loop for pdf and moments together
(`:9068-9108`); with `exact_moment_grad=False` the moments inherit the
pdf's capped step (harmless; stated).

### 4.2 Probes and divisors

| Domain | Full rule | Capped rule | Divisor |
|---|---|---|---|
| positive | theta·e^(±s) | theta·e^(±u), u = min(log1p(1e-7/theta), s) | actual difference, with zero guard |
| unit | logit step of s | logit step of size min(s, 1e-7/(theta(1-theta)))/(1 + s) (the extra factor keeps the up-probe at most ~1e-7 above theta to second order) | actual difference, with zero guard |
| unconstrained | — | theta ± 1e-7 | literal 2·eps, no clamp |
| `None` | — | today's code at that site, unchanged | today's (2·eps, or actual difference at the clamp sites) |

The capped up-probe is never more than 1e-7 above theta (review 1 m1
corrected v1's formula), and no probe is ever negative.

### 4.3 Error model

In the domain coordinate, error ≈ s²|F'''|/6 + eps_fwd|F|/s (for log
steps, s²·theta·(F''/2 + theta·F'''/6) in theta terms). Measured:
- at theta >= 1e-3 the change from today is ~1e-8 relative
  (review3/p1_daisy.log); at population-genetics rates (1e-4 to 1e-5),
  where most test fixtures sit, about 1e-7 to 1e-5 relative (review 4 m4),
  because today's absolute 1e-7 step is then a large relative step;
- for parameters whose effect is linear near 0, the relative error in
  theta terms grows like 1/theta: 1e-3 at 1e-9, ~100% at 1e-13
  (review3/p3_nearzero.log).

That is harmless for fits, where softplus floors theta at 1e-9
(`svgd.py:5254`) and the absolute error in phi is ~1e-11, but it is a
documented limit for direct theta-space callers.

### 4.4 Crash fallback (fixes every route)

**Whether a forward refuses a negative probe is a property of the built
MODEL, not of the site** (reviews 5 and 6, measured on master). A model
refuses when ANY of these holds:

- it is multi-epoch (`graph_builder_ffi.cpp:2000`, `:2242`; threshold
  -1e-12 at `:1620`);
- it is discrete, native (`phasic.c:6716-6745`) or `was_dph`
  (review6/p4_discrete.log);
- its weight mode is `log` (`graph_builder.cpp:613-622`,
  `phasic.c:5739-5757`), `formula` (`phasic.c:5706-5718`) or `callback`
  (measured);
- it is built by `pmf_from_cpp` (user C++ code may transform theta
  arbitrarily; the add-edge check refuses a negative weight,
  `phasic.c:4938`; measured, review7/p1_cpp.log): ALWAYS refusing,
  continuous and discrete;
- it is CALLED WITH REWARDS (`rewards is not None`): the reward-transform
  rebuild goes through the same add-edge check, so a linear continuous
  `pmf_and_moments_from_graph` model refuses a negative probe when called
  with rewards and accepts it without (review7/p3_rewards_neg.log). Whether
  `rewards` is None is static at trace time, so this part is decided
  inside the `custom_vjp` closure at call time. The zero-inflation
  cdf-at-zero site (`:8943`), which evaluates on reward rows
  (`zero_inflation.py:160-174`), counts as refusing.

The flag is NOT made unconditional for `pmf_and_moments_from_graph`: its
no-rewards linear form must keep today's behaviour (the "FD is still
broken" tests at `test_fd_gradient_mixed_scale.py:112`, `:132`, goal 6).

Only linear-weight continuous models accept a slightly negative probe (the
check is commented out, `phasic.c:5767-5790`; the Euler pdf path is then
~1e-8 accurate, review3/p3_nearzero.log), when called without rewards and
not built by `pmf_from_cpp`. The model builder computes the build-time part
of this flag once and passes it to `_fd_steps` with the domains; the
rewards part is added at trace time. For the direct
`Graph.reward_visit_probability` call, which has no model, the rvp backward
derives the flag from its `structure_json` (weight mode and discreteness).

The flag is exposed through ONE named function, `_fd_steps.model_refuses(
build_info, has_rewards)`, which every site calls. The refactor plan
([core-engine-ssot-plan.md](core-engine-ssot-plan.md), phase 5) later
replaces its implementation with a C query on the graph-state record
(`refuses(record, has_rewards)`, with `pmf_from_cpp` refusing by
constant), so the swap happens in one place. The Batch 0 refusal grid is
the starting point of the refactor's phase-1 refusal matrix, and the two
must agree.

On a refusing model, at EVERY FD site of that model (including the
`pmf_and_moments_from_graph` clamp sites and the `reward_visit_probability`
backward used by zero inflation):

- an undeclared or `'unconstrained'` slot with 0 < theta <= 1e-7 uses the
  capped positive rule;
- one with theta exactly 0 uses the one-sided step of §4.5.

The `'unconstrained'` case covers `positive_params=False` fits, which today
survive only through the 1e-9 clamp. Removing that clamp under
`'unconstrained'` would otherwise crash log, formula, callback and
discretized fits (review6/p1_modes.log).

On a non-refusing (linear continuous) model, undeclared slots keep today's
code everywhere, and `'unconstrained'` is plain ±1e-7.

Goal 1 still cannot cover non-monotone formulas (e.g. `c0*t0 - c1*t1`),
which can produce a negative edge weight with every slot large, nor
`'unconstrained'` slots in log or formula mode whose valid domain is
theta < 0 (a negative coefficient): today crashes there too. Both are the
model's own domain, stated in §9.

### 4.5 Degenerate values (no silent NaN, no silent zero)

- **Declared positive with theta == 0:** a one-sided difference over
  [0, 1e-7] (both probes non-negative, so forwards that refuse NEGATIVE
  weights accept them; for theta < 0 the forward itself is invalid and
  today's probe is kept). Exception: log mode refuses a product that is
  exactly 0 (`graph_builder.cpp:617-621`, `phasic.c:5745-5756`), so there
  the primal itself raises at 0 and today's clear error is the expected
  outcome, selected with `jnp.where`. This is the case of a rate fixed
  at 0 in a nested model (allowed since bug 2, `svgd.py:5196-5247`; the
  epoch-route crash is reproduced on master, review4/p1_edges.log, and
  named in `test_fixed_values_exact.py:362-365`).
- **Declared unit with theta outside (0, 1):** that slot uses today's
  probe, selected with `jnp.where` (exposure-scaled values).
- **The exposure slot is never `'unit'`:** the resolver widens a BetaPrior
  (sigmoid) on `exposure_param_index` to `'positive'` (the wrapper
  multiplies it by alpha, `svgd.py:315`, so values above 1 are valid;
  unit ⊂ positive, so the containment check still passes). On the epoch
  route `exposure_param_index` is a per-epoch local index (`svgd.py:5685`);
  the widening applies to every flat slot e·P + k. `SVGD.__init__` uses the
  same containment check (not equality), so the widened slot fitted with a
  sigmoid passes.
- **A zero probe difference** (theta above ~2e9, where the capped u falls
  below ~5.5e-17; at 1e9 the down-probe still moves, review6/p1_modes.log)
  gives a zero gradient for that slot and a host-side counter plus a
  once-per-model log line, so it is not silent. Batch 0 measures the band.

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
  (`:3751`). The domain tuple (a tuple of str, plus the refusing flag) is
  read from `model._param_domains` and passed as a second nondiff
  argument; the non-FFI branch uses `cdf_zero_fn` (site `:8943`) and
  inherits the builder's domains.

### 4.7 One helper module

`src/phasic/_fd_steps.py`, a leaf module (`__init__.py` would make a
circular import). It returns probe points (selected with `jnp.where`, which
is exact). Each site then computes TODAY's quotient verbatim and, where a
rule other than today's applies, the fallback quotient separately, and
selects between the two QUOTIENTS, never between divisors: selecting the
divisor with `jnp.where` changed 58,870 of 200,000 elements under jit,
selecting the quotient changed none (review7/p2_divisor.log). On
non-refusing models with undeclared slots there is no traced selection at
all. Bit-identity tests compare each mode (eager, jit, vmap(grad)) against
master in the SAME mode, never eager against jit (XLA already rewrites
x / (2·eps) under jit). At traced sites everything is jnp with no Python branching
on theta, so it runs under `vmap(jit(grad))`; the joint-index host fallback
(`__init__.py:9551-9563`) runs on concrete numpy arrays and gets a numpy
variant of the same rule. Domains and rules are bound when the
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
| `:9545-9570` (step max(abs(theta)·1e-6, 1e-10), numpy, probes all slots; goes negative below 1e-10) | joint-index host fallback for `exact_grad_decline='fd'` (forwarded by `Graph.svgd` at `:7229-7237`) | elimination; undeclared slots keep today's rule bit for bit (plus the §4.4 fallback if the model refuses); declared slots use the numpy variant of §4.2; only `'fixed'` slots are skipped |
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

- **Enumeration (mechanical):** every `custom_vjp` / `defvjp` in
  `src/phasic/*.py` (review 5 counts 37 lines in `__init__.py` plus
  `ffi_wrappers.py:1278`, `:1316`). Each is listed as an FD site (with its
  loops, slots probed, divisor/clamp/guard) or as resolved-not-FD.
- **Refusal grid** for every FD site:
  - builder {graph-based, `pmf_from_cpp`};
  - rewards {none, 1-D, 2-D multivariate};
  - weight mode {linear, log, formula (incl. sqrt and division),
    callback};
  - discreteness {continuous, native DPH, `was_dph`};
  - slot value {1, 5e-8, 0, 1 - 5e-8}.

  It confirms the §4.4 model property.
- **Step size per family**, on coalescent, two-island, reward-transformed,
  epoch, native discrete and discretized models. Acceptance criterion: the
  smallest s whose median relative error against an exact or Richardson
  reference is within 2x of the best s on the scan grid {1e-3 ... 1e-6},
  on both ordinary (theta ~ 1) and population-genetics (1e-4 to 1e-5)
  values.
- The degenerate cases of §4.5; the census of tests that replicate today's
  rule; a fresh master suite baseline with `gate2.sh`.
- Record in `plans-and-findings/fd-domain-steps-findings.md`.

### Batch 1: helper, builders, crash fallback (bit identical outside the §2 band)

- `_fd_steps.py`; `param_domains` on every builder in §5 (accepted but not
  yet defaulted anywhere); the `'fixed'` value; the §4.4 fallback; the §4.5
  guards.
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
    direct `SVGD(model=...)`, direct `jax.grad`);
  - slot = 5e-8 and slot = 0 for each weight mode × {`pmf_from_graph`,
    `pmf_and_moments_from_graph` with and without rewards,
    `reward_visit_probability`, `pmf_from_cpp`} and on a discretized model's
    discretize slot: no crash (except log mode at exactly 0, where today's
    primal error is expected);
  - a **probe-recording spy** through a named seam: `_fd_steps` gets a
    test-only probe hook that each loop feeds the ACTUAL probe vector it
    passes to the forward (wrapping `_fd_steps` alone would not see a
    loop/slot misalignment, and the multi-epoch forward is an FFI call
    under `custom_vmap` in a closure). Record every probe vector,
    and assert per slot that only slot i changed, the probe lies inside that
    slot's domain, fixed slots are never probed and tied slaves are. Use a
    heterogeneous list (unit / positive / fixed / tied slave / exposure)
    across >= 2 epochs;
  - native discrete at the upper row-sum wall: today's clear error is kept
    (out of scope, §4.1).
- The mixed-scale pin flips here through the §4.4 fallback (its small
  slot is 1e-8, undeclared, at a refusing site).
- Also tested: epoch-route fits with a rate fixed at 0 (the nested
  likelihood-ratio model); an exposure case with small alpha inside the
  fallback band.

### Batch 2: Graph.svgd declares domains (fits change slightly)

- The resolver in `Graph.svgd`; `Graph.epoch_model` defaults to
  `'positive'` (moved here from Batch 1 so the two routes stay in parity);
  `_param_domains` on the model; the containment check in `SVGD.__init__`
  and `FreeEpochModel.fit`; `positive_params=False` → `'unconstrained'`.
- **Tests:**
  - the flat epoch domain list for mixed Beta/Gauss priors with tying and
    fixed slots;
  - gradients at fixed particles equal master's within 1e-6 relative for
    theta >= 1e-3 (measured ~1e-8), and within a tolerance set from Batch
    0's measurement at theta = 1e-4 and 1e-5 (expected 1e-7 to 1e-5);
  - one coarse fit-level check per route;
  - the reported epoch fit from near-floor particles completes;
  - `positive_params=False` × {log, formula, callback, discretized,
    linear with rewards} × theta = 5e-8: no crash (the §4.4 fallback
    replaces the 1e-9 clamp).

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
- `test_epoch_model.py:83` and `:161` (epoch_model reproduces Graph.svgd
  at rtol 1e-9): must keep passing; they do because both routes declare in
  the same batch (Batch 2).
- **Behaviour changes (Batch 2), listed for the user:** `epoch_model`
  gradients change for every user (declared 'positive'), by ~1e-8 relative
  at theta >= 1e-3 and up to ~1e-5 at 1e-4 to 1e-5; `epoch_model(...).fit`
  with `positive_params=False` on a default model raises with a rebuild
  message.
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
- **The §2 band:** undeclared slots with 0 <= theta <= 1e-7 on refusing
  models change even where today did not crash (theta == 0 included: the
  one-sided step replaces today's clamp secant or central difference; such
  points are excluded from the bit-identity tests).
- **Out of scope by nature:** the native-discrete upper row-sum wall, and
  non-monotone formulas that make an edge negative with every slot large.
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

### Review 4 (of v2) → v2.1

Report: `experiments/fd_domain_steps/review4/REPORT.md`. B1 (fixed 0 on the
epoch route still crashed) → §4.5 one-sided step, tests. M1 (slot-level
trigger versus edge-level refusal) → §2 goal 2 restated with the band,
§9. M2 (Batch 1 parity break) → epoch_model default moved to Batch 2, §7.
M3 (over-strict check) → containment, storage on FreeEpochModel, listed
behaviour change. m1 → fallback also for 'unconstrained' at refusing
sites; prior=None wording. m2 → numpy variant at the host site. m3 →
capped logit factor. m4 → population-genetics regime in §4.3 and Batch 2.
m5 → zero-guard claim replaced by a measurement.

### Reviews 5 and 6 (of v2.1) → v2.2

Reports: `experiments/fd_domain_steps/review{5,6}/REPORT.md`.

- **Refusal depends on weight mode and discreteness** (R5 B1, R6 B1, R6
  M3). Measured: log, formula, callback and discretize slots crash at
  5e-8. Fix: §4.4 makes refusal a property of the model computed at build
  time; was_dph row in §4.1; the Batch 0 refusal grid; Batch 1 tests per
  weight mode.
- **The Batch 2 crash for `positive_params=False`** (R6 M2, R5 B1.2) →
  §4.4 applies the fallback to `'unconstrained'` on refusing models,
  including the clamp sites; Batch 2 test.
- **The joint-index host fallback** (R5 M1) → §5 row.
- **BetaPrior on the exposure slot** (R5 M2) → §4.5 widening to
  positive, per flat epoch slot.
- **Misaligned domains** (R5 M3) → the probe-recording spy test.
- **The upper row-sum wall** (R6 m4) → out of scope, with an explicit test
  expectation.
- **Wording.** R6 m5: refusal qualified by weight mode; non-monotone
  formulas. R5 m2: theta == 0.
- **Census.** R5 m5, R6 m6 → Batch 0 enumeration, grid and acceptance
  criterion.
- **Smaller items.** R5 m1: None passes containment. R5 m3: the zero
  difference is counted and logged. R5 m4: the length check at first
  trace. R5 m6: prompts updated. R5 m7: the rvp domain tuple.

### Review 7 (of v2.2) → v2.3

Report: `experiments/fd_domain_steps/review7/REPORT.md`.

- **Rewards make a linear model refuse** (B2) → §4.4: the flag is the
  build-time property OR rewards present, decided in the `custom_vjp`
  closure; the cdf-zero site counts as refusing; a rewards axis in the
  Batch 0 grid; tests.
- **`pmf_from_cpp`** (B1) → always refusing; tests.
- **Divisor selection breaks bit identity** (M1) → §4.7 selects
  quotients, not divisors; same-mode comparisons.
- **Minors.**
  - m1: the band includes theta == 0.
  - m2: log mode at exactly 0 keeps the primal error.
  - m3: the spy's probe hook seam.
  - m4: the rvp flag from `structure_json`.
  - m5: negative-domain slots out of scope.
- **Part A note** → containment stated for `SVGD.__init__` as well.
