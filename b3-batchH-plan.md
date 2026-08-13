# Batch H plan — daisy-chain FINAL-epoch exact gradient (de-risk first)

**Status: DRAFT v1, pending adversarial plan review. Per master plan §10
this batch "must be de-risked and adversarially reviewed on its own" — it
introduces the first Jacobian-shaped (not gradient-shaped) B3 primitive.
The implementation section here is a sketch to be re-detailed from the
de-risk findings.** Branch: `derisk/batchH-final-epoch` then
`b3/batchH-final-epoch`. Baseline: ledger @ `d2cca7ab`.

## Goal

`final_read='sojourn'` (the shipped default) already reads the final epoch
via `joint_sojourn_graph()` — an exact, granularity-free ELIMINATION solve
— but the gradient still bulk-FDs the whole chain. This batch gives the
FINAL epoch's contribution an exact gradient by composing the shipped
sojourn adjoint (`ptd_sojourn_grad_theta_subset`) across ONE epoch/IPV
boundary, leaving intermediate epochs on FD. No granularity-pinning and no
backprop-through-time needed (master plan §10) — those stay Deferred-2's
problem.

## The mathematical structure (to be VERIFIED in de-risk, not assumed)

The final read is `final_jp[c] = r_v · sojourn(v; theta_final, ipv_in) ·
handoff_mass` (r_v read per-vertex since `9a80ac45`; sojourn normalizes the
IPV and the mass rescale undoes it — net effect: **final_jp is LINEAR in
the raw handoff IPV** [reasoned from `alpha @ (-S)^{-1}` linearity in
alpha; de-risk H0 verifies]). Consequences for the two Jacobian blocks the
epoch-boundary composition needs:

- **d(final_jp)/d(theta_final)** at fixed ipv: the SHIPPED
  `ptd_sojourn_grad_theta_subset` on a clone with `update_ipv(handoff)` —
  reuse, not new C. (Scope note: the same probe-and-commit /
  raise-on-decline semantics Batch F just established should govern here —
  decided at implementation planning.)
- **d(final_jp)/d(ipv_in)**: by linearity, column j is the sojourn read
  with the basis IPV e_j — i.e. n_ipv extra sojourn evaluations (primal
  only, no gradient), or better a single transposed solve. De-risk H0
  measures which; n_ipv is tens-to-hundreds on real fixtures.

The upstream cotangent then chains: theta_bar_final = J_theta^T·g;
ipv_bar = J_ipv^T·g flows into the FD gradient of the earlier epochs
(the existing bulk-FD `custom_vjp` keeps computing d(chain-up-to-
handoff)/d(theta) — the composition point is INSIDE the current
`_autodiff_bwd`, whose external interface is unchanged).

## De-risk phase (branch-only, experiments/)

- **H0 — pure-JAX one-hop oracle.** Small daisy fixture (the
  `test_lrt_at.py` epoch model): implement final-epoch read as a dense
  differentiable solve in JAX; verify (i) value parity vs production
  `final_read='sojourn'` (~1e-12); (ii) LINEARITY of final_jp in raw ipv
  (exact, by construction check); (iii) `jax.jacobian` w.r.t.
  (theta_final, ipv_in) as the oracle for both blocks; (iv) the composed
  chain gradient (FD intermediate + exact final) vs full-FD and vs the
  full-JAX-oracle gradient — quantifying how much of FD's error the final
  epoch owned.
- **H1 — primitive-shape decision (the master-plan §10/§16-risk-6 E/H
  question).** Options: (a) NO C change — J_theta from the shipped
  function per epoch call; J_ipv via n_ipv primal sojourn reads (or the
  FFI's existing batched read); (b) a new C entry for the transposed
  ipv-solve. De-risk measures (a)'s cost on realistic n_ipv first;
  (b) only if (a) is prohibitive. Batch E's consumer relationship: E uses
  the shipped function at different index granularity — option (a) leaves
  it untouched (no interface collision); option (b) is additive-new.
  Record the decision + evidence.
- **H2 — wiring-point study.** Locate the exact composition point inside
  `_daisy_chain_svgd_model`'s two `custom_vjp` sites (no-exposure and
  exposure branches — the Batch-F/D2-review lesson: SVGD reaches THESE,
  not `daisy_chain_joint_probs`'s wrapper) + the public wrapper; decide
  how the FD loop's perturbation set shrinks (final-epoch theta slots
  leave the FD loop; earlier slots keep FD through the full chain
  INCLUDING the final read — or the ipv_bar composition replaces that
  too: exactness/cost trade recorded with numbers from H0(iv)).

## Implementation sketch (re-detailed after H0-H2)

- I1: host callback for the final-epoch Jacobian blocks (clone +
  update_ipv + shipped sojourn adjoint; probe-and-commit semantics per
  Batch F precedent); I2: composition inside the daisy `custom_vjp`s
  (static dispatch, no lax.cond; exact_grad-style opt-in kwarg, default
  False, name/semantics coordinated with the D.3/G plumbing story);
  I3: tests (oracle parity incl. mixed-scale; exposure branch parity;
  FD-only path byte-identity; decline/raise cases); gates G0-G5 per
  process with the chunked G3.

## Risks / notes

1. The linearity claim is load-bearing and cheap to verify — H0(ii) is
   the very first check; if it fails the ipv-block needs the full
   Jacobian treatment (cost model changes).
2. Exposure branch: theta pre-scaling happens OUTSIDE the FFI (per the
   Deferred-2 review's verified read) — the final-epoch theta slice must
   compose with that scaling; H2 maps it precisely.
3. The E/H interface question (master risk 6) closes with H1's decision —
   option (a) means no collision at all.
4. CC-2 pinning: unaffected (moments family).
