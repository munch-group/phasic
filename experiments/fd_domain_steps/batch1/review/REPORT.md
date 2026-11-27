# Plan A Batch 1: implementation review (Opus, 2026-11-27)

The reviewer's harness refused its own write of this file, so the main
session saved the report as delivered. Probe scripts and logs r1-r9 are in
this directory. Every run was memory-capped (6 GB), one at a time; nothing
under src/ or tests/ was edited.

**Verdict: SIGN-OFF.** No blocker, no major; six minors. m1 and m2 are worth
deciding before Batch 2 declares domains.

## Requirements and how they were checked

1. **Undeclared slots bit-identical outside the band: verified.**
   - Today's probe and quotient lines are verbatim. The two clamp-site
     `actual_diff` rewrites reduce to today's expression for every non-plain
     rule.
   - r3 compares against an emulated master: forcing `slot_rule` to give
     TODAY for undeclared slots reproduces master's traced program.
     - Four modes: eager, jit, vmap(grad), and jit(vmap(grad)), SVGD's real
       mode, which the goldens do not cover.
     - 14 cells, plus thetas just above the band where today's 1e-9 clamp
       fires (1.0001e-7 to 1.01e-7).
     - vmap batches mixing one in-band particle with out-of-band ones.
     - Result: 0 mismatches.
   - r4 shows the goldens are sensitive to a wrong band: widened to every
     t >= 0, the bitwise cells fail; with the band edge off by 2x, the three
     2e-7 cells fail.
   - No extra serialize(): both rvp callers pass a dict. The new attributes
     are read nowhere else in src/.
2. **Probes stay in the domain: verified.** No zero crossing on refusing
   models; 'fixed' slots are never probed; tied slaves are probed. The
   exceptions are m1 and m6.
3. **Refusal per site matches v2.10 §4.4: verified.**
   - r1: serialize() exposes weight_mode, is_discrete and was_dph for every
     mode and for native-discrete and discretized graphs.
   - The flag is right on pmf_from_graph, pmf_and_moments_from_graph, the
     rvp backward and the joint index.
   - pmf_from_cpp, moments_from_graph, both cdf-at-zero closures, the epoch
     model and daisy_chain_joint_probs always refuse.
   - Rewards add refusal at trace time through `_rewards_provided`.
4. **Families, s = 3e-5 and §4.5: verified.** The exceptions are m4 and m6.
5. **jit, vmap(grad), the custom_vmap epoch forward and SVGD: verified.** No
   Python branching on traced values; non-refusing undeclared models trace
   no new ops. Cost: m3.
6. **Import time and forward values unchanged: verified.** The logger is
   created lazily; test_14_import_order passes.
7. **The tests test what they claim: mostly.**
   - Forcing the alternative quotient to 0 fails 55 of 109 non-golden tests.
   - Halving it fails 45 of 68 selected tests.
   - Some value assertions cannot fail (m5).

**Missed sites: none.** Every FD loop is wired. The only remaining ±1e-7
loops are the out-of-scope preconditioner at svgd.py:4032, :4136 and :4238.

## Findings (all minor)

**m1. A slot declared 'unit' crashes at exactly 0 on a refusing model; an
undeclared slot does not.**
- Where: `_fd_steps.probes` (and `host_probes_np`). The unit rule uses
  `(t > 0) & (t < 1)`, so at 0 it falls back to today's ±1e-7 probe.
- Evidence (r5 E1/E1b), epoch model at [1, 0, 1, 0.5]:
  - undeclared and all-'positive' both work;
  - declaring 'unit' raises "NEGATIVE transition rate";
  - native-discrete `pmf_from_graph` declared 'unit' raises "negative
    weight".
- It follows §4.5's wording but contradicts goal 1: a declared slot gets
  less crash protection than an undeclared one.
- After Batch 2 it becomes reachable through sigmoid underflow (phi below
  about -745), or an explicit 'unit' declaration on epoch_model with a value
  fixed at 0. Master crashes too: a gap, not a regression.
- Suggested fix (plan amendment): at t == 0 a 'unit' slot takes the
  one-sided [0, 1e-7] step that 'positive' already uses.

**m2. With exact_final_grad=True, a 'fixed'-domain final-epoch slot gets a
nonzero exact gradient.**
- Where: the efg override loops (`__init__.py` ≈5444-5453, ≈5824-5833). The
  'fixed' skip comes after the exact branch.
- Evidence (E2): 0 without efg, 20.80 / 20.70 with it, against the
  docstrings' "zero gradient". The joint index zeros 'fixed' on its exact
  path, so the two are inconsistent.
- Graph.svgd cannot reach it today (truly fixed slots are already skipped).
  A direct param_domains does, and so would epoch_model (bake_fd_skip=False)
  if Batch 2 declares 'fixed' there.
- Suggested fix: add `_fd_steps.fixed_slots(...)` to `fixed_set_local`, or
  test 'fixed' before the exact branch.

**m3. The tie/zero-difference report makes one host callback per particle
under vmap, unrolled into the compiled program.**
- Cause: JAX 0.7.2 unrolls `jax.debug.callback` across the batch axis
  (debugging.py:112).
- Evidence (r2_cost.log), refusing epoch model under jit(vmap(grad)),
  emulated master against the branch:

  | particles | callbacks compiled (master / branch) | first call (master / branch) | per call after (master / branch) |
  |---|---|---|---|
  | 8 | 0 / 8 | 0.28 s / 0.31 s | 18.9 ms / 25.0 ms |
  | 64 | 0 / 64 | 0.21 s / 0.47 s | 108 ms / 108 ms |
  | 256 | 0 / 256 | 0.48 s / 1.17 s | 369 ms / 404 ms |

- Gradients are bit-identical. The cost is small next to production
  forwards, but it grows with the particle count on every refusing model,
  including the default epoch path.
- Findings §10.1's "one callback per backward" is wrong under vmap.
- Suggested fix: `jax.experimental.io_callback(..., ordered=False)`, which
  batches as a rolled `lax.map` (callback.py:474-489), or correct the wording.

**m4. The rvp counter is per process, not per model**
(`ffi_wrappers._RVP_FD_STATS`).
- Evidence (E3): two graphs fed one counter (tie = 2), logged once per
  process. §4.5 says once per model.
- Suggested fix: pass the model's counter through zero inflation, or
  document the exception.

**m5. Some value assertions in test_fd_domain_steps.py cannot fail** (r7,
r8):
- **rvp cells in the band crash test:** with REWARDS = [0, 1, 2, 0], vertex a
  is rewarded and always visited, so p ≡ 1 and the gradient is 0. Only "no
  crash" is tested. Reward vertex b instead.
- **pam-log cells:** the exact moments gradient (about -3.6e19) dominates the
  loss, so the FD pdf term is never checked. Use exact_moment_grad=False.
- **Epoch band test:** for dcjp at every band value, and for "first-1e-9" on
  the three epoch cells, the tolerance allows |g| up to 1e2-1e3, so a zero
  gradient passes.
  - The dc_* cells agree to about 1e-10 relative, so they could assert a
    relative tolerance.
  - dcjp is noise-dominated (-0.232 against -0.046 at 1e-9) and should be
    "finite only".
- **Tied-master test:** compares two outputs of the same rule, so 0 == 0
  passes under the mutation. That is inherent to a consistency test.
- **_fd_golden_cells.py's rtol 1e-6 reason ("orders of magnitude") is
  refuted by r4:** with the band wrongly applied everywhere, cdf_zero and
  multivar at [0.7, 1.3] move by less than 1e-6 and still pass, and
  pam_linear_rw at [0.7, 1.3] fails only barely (1.26e-6). Only the 2e-7
  cell protects.
- **Not exercised:** the joint-index host fallback loop
  (exact_grad_decline='fd') at the site itself.
- **Platform gating:** the 74 bit-identity cells skip off darwin-arm64, so
  Linux CI does not guard goal 2. An emulated-master check like r3 would be
  platform-independent.

**m6. 'unit' probes near 1 hit float resolution** (E4).
- At 1 - 2.2e-16 the probe difference is 0, and the log message ("theta too
  large ... above ~1.8e9") is misleading there.
- At nextafter(1, 0) the up-probe is exactly 1.0, outside the open domain.
  This is harmless for the row-sum wall (1 + 1e-9).
- Suggested fix: a domain-neutral message, and optionally clamp the
  up-probe.

## Fine

- **Probe arithmetic:** the cap, the one-sided step, the unit (1+s) factor,
  the zero guard, and the tie definition with its NaN exclusion and moments
  mask.
- **Quotient handling:** the quotient is selected, never the divisor; the
  exact-moments swap is kept in the alternative branch.
- **'unconstrained':** literal 2·eps and no clamp; the signed parameter at
  [6, -5] gives the right gradient.
- **Plumbing:** the length check, the multivariate passthrough, the
  zero-inflation domains, and the rvp nondiff argument under vmap and jit.
- **Existing-test edits:** the test_discrete_row_sum.py rewrite asserts the
  geometric closed form at rel 1e-6, and the mixed-scale xfail removal keeps
  its rtol 1e-2 value check.
- **Pickling:** models were already unpicklable, so the counter's lock
  changes nothing (r6).
