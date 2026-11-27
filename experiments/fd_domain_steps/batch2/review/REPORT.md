# Plan A Batch 2: implementation review (Opus, 2026-11-27, commit b0b501d3)

The report was delivered as the reviewer's final message and saved by the
main session. Probes r1-r4 are in this directory.

**VERDICT: SIGN-OFF.** No blocker, no major; six minors.

- **m1.** Reusing a Graph.svgd model in a direct `SVGD(model=s.model, ...)`
  call with a widening fit now raises. Two cases:
  - `positive_params=False`, or a softplus prior where the first fit had a
    BetaPrior;
  - a nested model reused for the full fit with its 'fixed' slot freed.
    Before, that slot's gradient was silently zero, so this is an
    improvement.

  Neither case is listed for the user, and the remedy text assumes the
  user chose param_domains. Fix: list both in §7 and findings §11.3, and
  improve the remedy.
- **m2.** epoch_model and Graph.svgd(epoch_starts) are no longer
  bit-identical with a BetaPrior: epoch_model declares 'positive', Graph.svgd
  'unit'. Learnable-slot gradients differ by 9.1e-9 and 4.2e-10 relative.
  Fix: document it.
- **m3.** The second assertion at scales 1e-4 and 1e-5 is implied by the
  first (triangle inequality). Replace it with "declared no worse than today
  + 1e-6"; the epoch margin is 1.4x.
- **m4.** Missing real-path tests:
  - exposure widening through Graph.svgd with a BetaPrior on the exposure
    slot;
  - the containment check read before the exposure wrapper (a raising
    case);
  - fit() checking before the default prior factory runs.
- **m5.** The parity-test edit is justified (p4 log), but the test should
  also call `random.seed(0)`. Note in §7 that a direct caller must declare
  param_domains to reproduce Graph.svgd.
- **m6.** Move the D-NZ note in FreeEpochModel.fit to a Notes block. The
  doc test's `"tie"` check passes trivially, because the docstring contains
  "tied"; assert "forward tie" instead.

**Verified fine:**

- the resolver split is behaviour-identical (MCMC included);
- every leaf is declared before it is built (mask form of fixed,
  theta_init-only, a custom transform with fixed slots, a tied slave, and
  formula/callback with a decoupled theta dimension);
- the containment rule and where it is read;
- no internal path reuses a model with a widening;
- the epoch_model forms;
- undeclared behaviour unchanged;
- the §12 reachability claim;
- the D-NZ text in all four places;
- test tolerances match the logs;
- the host-site test really reaches the host fallback.
