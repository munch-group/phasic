# Plan A v2 review (review 4, Opus, 2026-11-26, HEAD 256c788a) — CHANGES REQUIRED

Part A: resolved — missing sites (gap: :9545 host numpy vs "jnp only"), fixed/tied, forward classification, row-sum, fit xfails, MCMC, preconditioner, out-of-scope solvers, shared loop, nondiff arg, direct callers, unit near 1, fixed-particle testing, gate strength, signed params (gap m1), cap formula for positive (unit first-order only), error model (regime gap m4). Partial — done criterion (B1), domains/FreeEpochModel (B1, M3), bit identity (M1), degenerate values (B1), section-6 contradictions (M2).

BLOCKER B1: fit-time fixed 0 on epoch_model(...).fit crashes: jax.grad of epoch_model model at θ=[1e-4,0,1e-4,1e-4] -> NEGATIVE transition rate (p1_edges.log). FreeEpochModel probes fixed slots (epoch_model.py:17-21); fit() accepts fixed 0 (:84-88); §4.5 sends θ<=0 to today's probe θ-1e-7 refused (graph_builder_ffi.cpp:1919,:2157,:2203; kMinEdgeWeight=-1e-12 :1620). Standard nested LRT (test_epoch_model.py:100-120); test_fixed_values_exact.py:362-365 already names it. Fix: declared-positive θ<=0 -> one-sided [θ, θ+1e-7] via jnp.where (or zero gradient on fit-fixed slots only); test θ=0 fixed-slot epoch fits.

MAJOR
- M1 §4.4 trigger slot-level, refusal edge-level: θ1=1e-7 and 1e-7-1e-11 don't crash today (p1_edges.log); crash needs Σc_k·θ_k < -1e-12 (sums with larger slots, small coefficients/exposure α — probe before α scaling, unused in an epoch). Exact trigger infeasible under jit. Fix: goal 2 = bit identical except undeclared 0<θ<=1e-7 at refusing sites; list in §9; bit-identity tests avoid band.
- M2 Batch 1 epoch_model declared, Graph.svgd not -> test_epoch_model.py:83 (reproduces_graph_svgd) and :161 (exposure_parity) at rtol 1e-9, θ≈1e-4 -> fail (reasoning). epoch_model numbers change at all θ, unstated. Fix: move epoch_model default to Batch 2; pin flips in Batch 1 via §4.4; list parity tests + behaviour change in §7/§9.
- M3 fit() check over-strict: use containment (unit ⊂ positive; anything ⊂ unconstrained), raise only on widening; read domains from FreeEpochModel object not em.model (spy in test_fixed_values_exact.py:341-354 copies only _tying_info/_precondition_output); positive_params=False rejection on epoch_model = listed behaviour change.

MINOR
- m1 epoch route positive_params=False -> 'unconstrained' without fallback -> positive particle in (0,1e-7] still crashes at refusing sites; apply §4.4 to unconstrained at refusing sites or scope goal 1. Wording: prior=None -> 'positive' only under positive_params=True (__init__.py:7116-7126).
- m2 joint-index host fallback (:9551-9563) numpy -> numpy variant or scope "jnp only" to traced sites.
- m3 capped logit up-step θ(1-θ)v(1+(1-2θ)v/2+...) slightly > 1e-7 for θ<1/2: say ≈ or scale.
- m4 regime 1e-4..1e-5: declared rule changes gradients ~1e-7..1e-5 relative; add θ=1e-4 to Batch 2 comparison, state expected size.
- m5 zero-guard equivalence with today's rounding not bitwise guaranteed; drop phrase or test 1e9-1e10 band.
Sound: §4.4 under jit/vmap (jnp.where selects probes before one forward; daisy loop static, __init__.py:5294-5306); refusing forwards = daisy FFIs (graph_builder_ffi.cpp:2000,:2242) and native DPH (phasic.c:6716-6745); joint-index and Euler pdf accept sub-zero (p2_jix.log); LRT fast path unaffected; resolver inputs known before build (__init__.py:7095).
Add gates: epoch-route fits with fixed 0; exposure with small α in fallback band.
