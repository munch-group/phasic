# Plan A review 1 (Opus, 2026-11-26, plan at b9178dab) — CHANGES REQUIRED

BLOCKERS
- B1 Done-criterion "mixed-scale pins flip" unreachable: one strict xfail (test_fd_gradient_mixed_scale.py:229) built via epoch_model(...).model (:197) / pmf_and_moments_from_graph (:53), not svgd -> stays undeclared. Fix: epoch_model/FreeEpochModel declare domains, or rewrite pin to declare; count is one.
- B2 Domains cannot be "passed to the builder" with current order: Graph.svgd builds model first (__init__.py:7225,:7253,:7266,:7278; epoch :7095), transform resolved later inside SVGD (svgd.py:6195 _resolve_param_transform, side effects svgd.py:5316-5322); epoch route returns prior from builder (:6023,:7095); FreeEpochModel builds once, fits with varying positive_params/prior (epoch_model.py:72,:131,:154) vs construction-time binding. Fix: side-effect-free domain helper from (prior, positive_params, param_transform) before build; epoch default prior GaussPrior lower=0 -> softplus; FreeEpochModel rebuild per fit or one model per domain signature.

MAJORS
- M1 Inventory misses __init__.py:9552-9563 (joint-index exact_grad_decline='fd' host fallback, _h=max(|theta|*1e-6,1e-10), numpy, probes fixed slots, negative at theta<1e-10); shared-constant loops :5245->:5306,:5344; :5501->:5647,:5698; test FD loop test_fd_gradient_mixed_scale.py:141.
- M2 Fixed-slot skip underspecified: tied slaves are in SVGD fixed (:4964-4970) but FD skips only fd_skip_indices (:4954,:5246-5248); slave partials flow to master via _apply_tying scatter-VJP (:5352-5357) -> using svgd fixed gives silently wrong tied gradients. Sites without fixed info: pmf_from_graph (:3928; :4089,:4211), pmf_from_cpp (:4600), moments_from_graph (:8006), ffi_wrappers.py:1296; plus zero-inflation term (:7287). Fix: skip = truly fixed only; name sources; tied-sum test (test_daisy_chain_c_path.py::test_tied_gradient_sums_per_epoch_partials).
- M3 Bit identity vs actual-difference division: 11 loops divide by 2*eps; only pmf_and_moments divides by actual_diff with min_theta=1e-9 clamp and _ok guard (:8668-8678,:8950-8961,:9076-9097). (theta+1e-7)-(theta-1e-7) != 2e-7 exactly (probe). Fix: keep each site's divisor/clamp/guard for undeclared; test under jit, vmap(grad), custom_vmap.
- M4 NaN on degenerate probes: theta=0 log step -> identical probes; cap at theta=1e10 -> u=1e-17 -> actual_diff 0; logit at/outside (0,1) NaN; fixed values may be 0 (svgd.py:5196-5247); exposure-scaled BetaPrior >1 (svgd.py:315). Fix: _ok guard everywhere; out-of-domain theta falls back to today's rule (or loud raise); forbid unit on exposure_param_index.
- M5 Jump classification: auto granularity max(1000,int(2*max_rate)) (phasic.c:12963-13006); daisy/multi-epoch uses stop_probability granularity=0 (__init__.py:4676,:4729-4731,:5195) -> needs cap; explicit granularity only exists for daisy.
- M6 Native DPH row-sum upper bound: test_discrete_row_sum.py:357-381; full rule s~1e-4 relative would raise near bound. Fix: keep 1e-7-capped rule on native DPH or check; test.
- M7 Section 6 contradicts section 2: test_exact_grad_daisy_final.py (:68-70) and test_exact_grad_rewards.py (:106-111) call builders directly -> undeclared -> must not change; test_daisy_chain_c_path.py:340,:797 replicate 1e-7 and must stay bit-exact.
- M8 MCMC is gradient-free (mcmc.py:536,:657; no grad) -> declaring domains is a no-op; drop.
- M9 Preconditioner compute_scaling FDs in phi coordinate (svgd.py:4021-4033,:4125-4137,:4224-4238); domain rule there is no-op or wrong (dm/dtheta vs dm/dphi). Drop.

MINORS
- m1 "never larger than today's step" false for u=min(1e-7/theta,s) (up-probe 1.00005e-7 at theta=1e-3); use u=min(log1p(1e-7/theta),s) or symmetric absolute theta±min(1e-7,s*theta).
- m2 Error model in theta terms for log step: s^2*theta*(F''/2+theta*F'''/6); state coordinate for tolerances.
- m3 Compare gradients at fixed particles + one coarse fit check, not fit-level tolerance.
- m4 ffi_wrappers.py:1296 rvp custom_vjp nondiff_argnums=(0,): domains must be hashable nondiff arg.
- m5 scipy least_squares in probability_matching.py:447, method_of_moments.py:480 does own FD within bounds (1e-10,inf): out of scope, mention.
- m6 pmf_and_moments shares one FD loop for pdf+moments (:9068-9108): with exact_moment_grad=False moments FD inherits pdf cap; state.
Probe: review1/cap_arith.py, cap_arith.log
