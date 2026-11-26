# Plan A review 3 (Opus, 2026-11-26, plan at b9178dab, empirical) — CHANGES REQUIRED

Measurements (logs in this folder):
1 p1_daisy.log: master crashes (NEGATIVE transition rate) on pin epoch model at θ=[1,1e-8,..],[1,1e-9,..],[1e-9,1e-4,..]; exact_final_grad model also crashes (earlier slots FD).
2 p1_daisy.log: plan rule (outside package) no crash; full s=1e-4 2e-9..5e-9 rel err; capped 1e-9..5e-5.
3 p1_daisy.log: ordinary θ: master 1e-11..3e-9, full 6e-10..6e-9, cap 2e-10..1e-9; difference ~1e-8 (rounding level).
4 p1_daisy.log: s scan 1e-2:5e-5, 1e-3:5e-7, 1e-4:5e-9, 1e-5:5e-11, 1e-6:1e-10.
5 p2_pdf.log: pdf Euler auto-granularity at max rate 3600: today median 1.6e-7/max 6.9e-7; cap 1.3e-7/7.0e-7; full s=1e-4 median 0.20 max 3.5.
6 p4_daisy_jumps.log: daisy earlier-epoch at scale 200: today=cap; full s=1e-4 median 0.13 max 330x; at scale 30 all agree.
7 p3_nearzero.log: linear-near-0 params: log rule rel err (θ terms) 1e-3 at 1e-9, ~1 at 1e-11, ~100% at 1e-13; master pdf path ~1e-8 (Euler accepts negative probe); daisy crashes there on master.
8 p5_discrete_edge.log: discrete near row-sum 1: θ·e^1e-4 raises at 0.99995 and 0.9999999; θ+1e-7 does not at 0.9999999.

BLOCKER 1: done unreachable (one strict xfail test_fd_gradient_mixed_scale.py:229 via epoch_model(...).model + jax.grad, undeclared); crash stays on epoch_model(...).fit (epoch_model.py:154 -> SVGD(model=...)), direct SVGD(model=...) (model_selection.py:362), custom param_transform. Fix: fallback only where today crashes — undeclared slot whose θ-1e-7 <= 0 uses capped log step (bit-identical elsewhere, all routes fixed); or epoch builders default 'positive'.
MAJOR 2: daisy earlier epochs stop_probability granularity 0 (graph_builder_ffi.cpp:1940,2168) has jumps -> must be capped (330x errors otherwise); 'unit' logit needs capped variant on Euler forwards (~250x today's step at 0.5).
MAJOR 3: discrete full rule widens row-sum crash band 1e-7 -> 1e-4 -> cap discrete unless 'unit'.
MAJOR 4: domains needed before model build; transforms resolved after (svgd.py:5265, side effect p._transform); default epoch prior built inside builder -> side-effect-free pre-resolution + consistency check.
MINOR 5: error model in domain coordinate only; absolute tolerances in domain coordinate; state limit for θ-space callers.
MINOR 6: test θ in {1e-6,1,1e3} with distinct gradients, fixed/tied interleaved; inspect flat epoch domain list (mixed Beta/Gauss, tying); θ=0 positive -> NaN zeroed silently by SVGD -> must raise; unconstrained divisor literal 2*eps.
MINOR 7: preconditioner bullet no-op; drop.
