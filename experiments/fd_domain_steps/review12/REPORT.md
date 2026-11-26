# Plan A v2.7 review 12 (Opus, HEAD d3e9c090) — CHANGES REQUIRED (1 MAJOR, 4 MINOR)

Part A: review 11 M1, m3-m5 and leftovers resolved; m2 (primal reuse) claim wrong (MAJOR 1); D-NZ consistent (host site the stated exception).
MAJOR 1: tie repair needs a THIRD forward: capped u = min(log1p(1e-7/θ), s) = s in band (log1p ≥ log 2), probes θ(1±1e-4), never θ+1e-7. Measured (p1_tie_forwards.log), offset slot beside O(1) rate at θ=1e-13: probes tie, f(θ)==f(up); f(θ+1e-7) repairs (-8.175955e-02 vs exact -8.175954e-02). Under vmap(jit(grad)) tie predicate batched -> third forward never skipped (cf. Batch F comment __init__.py:9888-9902) -> ~1.5x FD backward on refusing models (epoch route). Fix: (a) drop the repair: counted, logged zero, documented limit, no primal saved; or (b) restrict, select θ+1e-7 probe with jnp.where, measure cost, list in §9.
MINORS
1 save primal only where used (residuals today inputs only: :4084, :4205, :4595, :5291, :8656, :8938, :8991, :9874, ffi_wrappers.py:1290); outputs per observation/entry/target, not per vertex; only on refusing models (moot under (a)).
2 tie definition: all-NaN outputs vacuously "equal" (daisy NaN-fills at mass==0 handoff) -> require at least one entry finite on both sides; exclude entries NaN on one side; with exact moments active (:9061-9105) compare only FD-used outputs (pdf).
3 D-NZ docs: Graph.mcmc gradient-free -> replace with epoch_model param_domains docstring and FreeEpochModel.fit; align §9 (l.602).
4 host site can be refusing: joint-index exact scope includes native DPH (__init__.py:9409-9416, 9433-9470); direct pmf_from_graph_joint_index(exact_grad_decline='fd') on native DPH; svgd blocked by R31; host callback (:9545-9566) has no primal -> ties counted only.
Sound: containment/fixed; quotient selection; refusal flag static incl rewards (:9021); D-NZ wording; no new crash route.
