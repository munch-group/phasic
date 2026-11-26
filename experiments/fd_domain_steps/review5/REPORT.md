# Plan A v2.1 review 5 (Opus, 2026-11-26, HEAD 900b8016) — CHANGES REQUIRED

BLOCKER B1: refusal depends on weight mode, not only site. log mode refuses non-positive products (graph_builder.cpp:620); formula refuses negative/non-finite weights (phasic.c:5700-5717). Measured (p1_weightmode_refusal.log), jax.grad of pmf_from_graph at θ=[1,5e-8],[1,1e-9]: linear and formula c0*t0+c1*t1 finite; formula c0*t0+c1*sqrt(t1) RAISES; log RAISES. pmf_and_moments survives only via 1e-9 clamp (__init__.py:9068-9077). Consequences: goal 1 fails for direct jax.grad on pmf_from_graph/pmf_from_cpp under log/formula; Batch 2 'unconstrained' drops clamp -> new crash for log/formula svgd fits with 0<θ<=1e-7 (inferred); census would repeat linear-only error. Fix: (a) per site × weight mode census (incl callback) or (b, recommended) §4.4 band fallback at EVERY FD site regardless of forward; for 'unconstrained' keep fallback wherever today's clamp existed; Batch 1 tests on log and sqrt-formula graphs.

MAJORS
- M1 joint-index host fallback (:9545-9570): plan applies positive rule + fixed skip unconditionally -> changes undeclared numbers at every θ, NaN for negative θ (positive_params=False via exact_grad_decline='fd', :7229-7237). Fix: undeclared keeps today's rule bit for bit; declared uses numpy §4.2 rule; fixed skip only 'fixed'; per-site bit-identity test.
- M2 BetaPrior on exposure slot -> 'unit' refused at construction (§4.5) — new hard error not in §7 (no current rejection found, guess); SVGD exposure wrapper multiplies θ[k] by α (svgd.py:315). Fix: resolver widens exposure slot to 'positive'; epoch route exposure_param_index is per-epoch local (svgd.py:5685) -> apply to every flat slot e·P+k.
- M3 tests can't see misaligned domains (FD loop on expanded flat vector after _apply_tying __init__.py:5354-5357; fixed via static set :5246-5248,:5298). Fix: probe-recording spy test: only slot i changes, probe inside slot's domain, fixed never probed, slaves probed; heterogeneous list across >=2 epochs.

MINORS
- m1 custom param_transform -> None; define containment for None (passes) or list.
- m2 §4.5 "both probes non-negative" true only at θ==0.
- m3 zero-difference case contradicts "no silent zero": reword or make loud (host counter/log).
- m4 domain-list length checked against traced theta.shape[-1] at first trace (formula/callback late theta_dim).
- m5 census mechanical: enumerate every custom_vjp/defvjp in src/phasic (37 lines in __init__.py + ffi_wrappers.py:1278/1316); sites not in §5 (:4845,:8486,:8798,:9976,:11834) spot-checked not FD loops but list as resolved; refusal per weight mode; acceptance criterion for s.
- m6 prompts stale (§0 review{1,2,3}; §11 heading).
- m7 rvp domain tuple from model._param_domains as second nondiff arg (tuple of str); non-FFI branch uses cdf_zero_fn (:8943).
Sound: daisy site implementable (static per-slot domains, jnp.where before one forward, custom_vmap ok; exact_final_grad override loop needs same edit); _lib_cache only compiled C libs (:330,:4411,:4551); no pickling; gate scripts exist; pin unaffected; positive control :218 valid after Batch 2.
