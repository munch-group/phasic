# Plan A review 2 (Opus, 2026-11-26, plan at b9178dab) — CHANGES REQUIRED

BLOCKERS
- B1 Only ONE strict xfail (test_fd_gradient_mixed_scale.py:229), built via jpg.epoch_model(...).model (~:209) + jax.grad -> undeclared -> cannot flip. Tests at :112, :132 assert FD broken on undeclared builder; must stay. Fix: param_domains on Graph.epoch_model / FreeEpochModel; edit pin; list in section 6.
- B2 FreeEpochModel (epoch_model.py:72-160) builds once (__init__.py:5907/6060), fit() varies fixed/prior/positive_params; full.model is nested.model (LRT fast path) -> contradicts construction-bound domains and "fixed never probed". Fix: declare e.g. positive for all flat slots at epoch_model time; fit() rejects conflicting positive_params/prior; accept fit-time fixed slots still probed (safe only if fixed 0 cannot crash, M4).

MAJORS
- M1 Bit identity: most sites divide by literal 2*eps (__init__.py:4096,4222,4612,5306,5647,9905; ffi_wrappers.py:1308); (t+1e-7)-(t-1e-7)!=2e-7 (probe); sites differ in expression order. Helper returns probes/divisors only; undeclared returns today's divisor; per-site bit-identity test.
- M2 Clamp max(theta-eps,1e-9) at __init__.py:9068-9077, 8662-8669 kept for positive_params=False -> signed params wrong: rate=t0+t1 at [6,-5] today [-0.50049,-0.49382] vs symmetric [-0.50049,-0.50049] (probe_clamp.py). Fix: undeclared = legacy incl. clamp; declared 'unconstrained' = plain ±1e-7 no clamp; svgd(positive_params=False) declares 'unconstrained'.
- M3 Missing: (a) __init__.py:9545-9570 joint-index exact_grad_decline='fd' host fallback (h=max(|θ|·1e-6,1e-10), all slots incl fixed, negative for θ<1e-10), svgd forwards at 7237; (b) ffi_wrappers.py:1296 reached via zero_inflation.py:176-196 (svgd 7292, use_ffi=True) and Graph.reward_visit_probability (3751) — no domain/fixed threading; (c) pmf_from_graph/pmf_from_cpp/moments_from_graph lack fixed argument -> name mechanism (e.g. 'fixed' domain value). Out of scope list: scipy least_squares (MoM / probability_matching), BetaPrior solver svgd.py:1778.
- M4 θ=0 positive rule -> 0/0; reachable at fixed/undeclared slots on reused models; raise at construction for declared slot fixed <=0 else skip; under jit NaN not silent 0; test θ=0.
- M5 Daisy forward stop_probability granularity=0 auto (__init__.py:4729-4731,5088,5195) has jumps -> cap or Batch 0 evidence; specify unit slots on capped forwards.
- M6 Discrete row-sum wall phasic.c:6698-6770 (>1+1e-9 error): s=1e-4 relative probes crash near bound -> keep native DPH capped or row-sum back-off; test.
- M7 Fit-level strict xfails test_notebook_model_recovery.py:484,553,659,699 (699 = particles stuck at 1e-9 floor) may XPASS -> list and decide handling.
- M8 Graph.svgd builds model (7095,7225,7253,7266,7278) before SVGD.__init__ resolves transforms (svgd.py:5265, side effect prior._transform). Fix: side-effect-free _resolve_param_domains shared with transform resolver; define broadcast/tying on epoch flat vector; model carries _param_domains, SVGD.__init__ checks consistency (unit-declared + softplus fit -> logit(θ>1) NaN).

MINORS
- m1 mcmc never differentiates -> drop or justify.
- m2 preconditioner: no-op/contradiction; remove.
- m3 SVGD(model=...) direct users, model_selection examples, LRT MAP refinement (svgd.py:7875) get no fix unless they pass param_domains; document.
- m4 unit rule near 1-1e-9 (sigmoid clip svgd.py:413): probes differ ~2e-13, ~1e-3 relative; tolerance.
- m5 helper jnp-only, no Python branching on θ (tracers); vmap+jit test per site.
- m6 line numbers cite f4eeea41 but match at b9178dab.
WHAT HOLDS: fixed-slot skip pattern exists (5300,5641,9897,9069); exposure strictly positive (svgd.py:6065-6083), probed before alpha scaling (5629); exact paths bypass FD only for own components.
Probe: review2/probe_clamp.py
